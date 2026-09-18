"""GAN-type controller (spec §10; audit-2 §B7 / §D4): co-adaptive generator / discriminator over
*whole* acoustic compositions, with a small reference distribution, a joint low-rank proposal
basis and lineage.

  r_b     ∝ exp[ -J_ref(Xi_b^ref, F, H, o) / tau_ref ],  J_ref = E_form + 0.2 E_hist   (40)
  Xi_hat  = Xi_b^parent + B_ref w,   p_Theta(b, w) = alpha_b N(w; mu_b, diag sigma_b^2) (41)
  D_phi(Xi) = sigmoid(phi^T psi(Xi)),  0 < D < 1                                       (42)
  L_D     = -E_ref log D(Xi^ref) - E_gen log[1 - D(Xi^actual)] + lambda_D ||phi||^2     (43)
  l(b, w) = -log D(Xi^actual) + lambda_fit <d_xi^2(xi^actual, xi_hat)> + lambda_f E_form(44)
  grad_Theta E l = E[(l - b_past) grad_Theta log p_Theta(b, w)]                         (45)

Audit-2 revision.  The engine works in commit steps: it asks for window references
(`prepare_reference`), FREEZES one of them, improves the *gain trajectory* against it, commits the
first block and hands the realized composition back through `observe_committed`.  Therefore:

  * one reference = one sample (b, w) from p_Theta (kept in `Target.meta`), built on the window
    rows from the realized current composition (bounded continuity offset) and never touched
    again while it is being scored;
  * the adversarial update is one bounded update unit per committed block, with the reference set
    and the committed composition FIXED: discriminator step(s) first (positives = the reference
    compositions restricted to the same committed rows, weighted by r_b; negative = the committed
    realized composition), then generator step(s) by REINFORCE (eq. 45) on the stored sample.
    D and G alternate only inside this update unit, never inside the realization;
  * `update()` is not called by the engine any more (a reference must not move while it is scored).

Basis change (audit B7 / D4.1): `B_ref` is rebuilt for every unit from that unit's reference set,
so a displacement coefficient w has no fixed acoustic meaning across units.  The parent acoustics,
the parent ids, the discriminator phi (with its frozen input standardisation), the mixture logits
and the common history are inherited; mu and log_sigma of the displacement coefficients are
RE-INITIALISED (mu = 0, sigma = sigma_init) whenever the basis hash changes.  What was inherited
and what was re-initialised is recorded in the trace and in `history.mode_state['gan']`.
Coordinate transport (option 2 of the audit) is deliberately not implemented.

Fragment-vocabulary revision (docs/FRAG_CONTRACT.md, 2026-09-16, user-authorised).  When the
analyzer carries the fragment bank (`hires.fragment_vocabulary`) and the engine has handed the PCM
to the mode (`engine.setup: mode.sources = job.sources`), the reference distribution of eq. (40) is
no longer a set of Bank gain plans but a set of random FRAGMENT COMPOSITION TRAJECTORIES: a new
random fragment composition (positions + material levels) every `min_clip_seconds` along the unit,
built with `Analyzer.random_fragment_composition` on hop-spaced offsets, the goal level following
the form's exposure policy per phase, GOAL_HOLD rows pinned to xi_goal.  Every reference is
therefore an *exact* mixture of clips the realizer can actually play (its jump rate is the same
`min_clip_seconds`), so the ideal side lives inside the enlarged reachable set instead of being
tethered to continuous-clock probes.  Everything else - basis, parents, generator, discriminator,
REINFORCE, the basis-change re-initialisation - is unchanged; with no fragment bank the mode falls
back to the private Bank reference distribution (baseline configs behave exactly as before).

The discriminator only ever sees *realized* compositions (computed from the actual summed PCM) and
the reference compositions; ideal proposals are never scored by D.  Positive and negative examples
come from the same unit, the same rows, the same phase grid and the same analysis scales, and only
free (non GOAL_HOLD) windows enter psi.  No candidate id / origin label is a feature.

This is the tool's lineage / mutation / contraction cycle, not standard GAN inference: nothing
here claims convergence, an optimal equilibrium or a match to a real acoustic distribution
(spec §10.5 / structural contract G)."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..bank import Bank
from ..curves import MotionLimits
from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, fix_hold_rows


try:      # the engine is fully imported before any mode module is; no cycle at runtime
    from ..engine import HardConstraintFailure as _EngineFailure
except Exception:                                                         # noqa: BLE001
    _EngineFailure = RuntimeError

_BASES = (_EngineFailure,) if issubclass(_EngineFailure, RuntimeError) else (_EngineFailure, RuntimeError)


class NoFeasibleReference(*_BASES):    # type: ignore[misc]
    """No legal reference trajectory exists for this unit: a hard feasibility problem, not a soft
    criterion (spec §10.2).  A RuntimeError, and also the engine's HardConstraintFailure so the
    job reports HARD_CONSTRAINT_FAILURE / NO_FEASIBLE_PLAN_FOUND instead of crashing."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic; strictly inside (0, 1) after the clip."""
    z = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class GANMode(ModeController):
    cli_name = "gan"
    internal_id = "gan_coadaptive"

    # origin classes of a reference plan (hold mode).  The generator's categorical lives on these
    # SLOTS, not on the per-hold plan index, so that what it learns carries across holds and units.
    HOLD_SLOTS = ("levels_only", "committed_positions", "fragment_candidates", "random_positions")
    HOLD_SLOT_SHARE = (0.2, 0.3, 0.3, 0.2)
    # polyphony cap (hires.max_active_materials > 0): a silent material can only enter by replacing a
    # sounding one, so SWAP is its own origin class - the generator's categorical then learns directly
    # how often the law should change WHO sounds, which a flag on the other slots could not express
    # separately from where the jump comes from.  With no cap the 4 slots above are used unchanged.
    HOLD_SLOTS_CAP = ("levels_only", "committed_positions", "fragment_candidates", "random_positions", "swap")
    HOLD_SLOT_SHARE_CAP = (0.15, 0.2, 0.25, 0.15, 0.25)

    # ------------------------------------------------------------------ construction
    def __init__(self, cfg, analyzer, fs, rng, objective):
        super().__init__(cfg, analyzer, fs, rng, objective)
        md = cfg["mode_defaults"]
        self.p = dict(md["gan"])
        g = self.p.get
        self.n_ref_target = int(md.get("gan_reference_target", 8))
        self.n_hist_parents = max(1, int(md.get("gan_components", 2)))
        self.tau_ref = max(1e-6, float(g("reference_temperature", 0.5)))
        self.l2 = float(g("l2", 1e-4))
        self.lr_D = float(g("lr_D", 0.001))
        self.lr_G = float(g("lr_G", 0.001))
        self.lambda_fit = float(g("lambda_fit", 0.25))
        self.lambda_f = float(g("lambda_f", 0.5))
        self.C_fail = float(g("C_fail", 100.0))
        self.sigma_init = float(g("sigma_init", 0.2))
        self.sigma_min = float(g("sigma_min", 0.01))
        self.sigma_max = float(g("sigma_max", 1.0))
        self.basis_rank = max(1, int(g("basis_rank", 3)))
        self.samples_per_round = max(1, int(g("samples_per_round", 4)))
        self.inner_steps_D = max(1, int(g("inner_steps_D", 40)))
        self.inner_steps_G = max(1, int(g("inner_steps_G", 10)))
        self.n_fresh_parents = max(1, int(g("fresh_parents", 2)))
        self.step_clip = float(g("generator_step_clip", 0.05))
        self.grad_clip = float(g("generator_gradient_clip", 10.0))
        self.hint_min_corr = float(g("hint_min_correlation", 0.3))
        self.max_hints = int(g("max_hints", 4))
        self.ess_min_fraction = float(g("importance_ess_min_fraction", 0.5))
        # not in config.py DEFAULTS (shared file, not edited): read with .get + documented default.
        # the adversarial update unit now runs once per *committed block* instead of once per
        # search round, so the inner-step counts per update unit are capped accordingly.
        self.inner_steps_D_commit = max(1, int(g("inner_steps_D_per_commit", min(8, self.inner_steps_D))))
        self.inner_steps_G_commit = max(1, int(g("inner_steps_G_per_commit", min(4, self.inner_steps_G))))
        self.g_batch_max = max(1, int(g("generator_batch_max", self.samples_per_round)))
        self.baseline_rate = float(g("baseline_ema_rate", 0.2))
        self.continuity_seconds = float(g("reference_continuity_seconds", 2.0))
        self.min_block_free_rows = max(1, int(g("min_block_free_rows", 3)))

        self.lim = MotionLimits.from_config(cfg["motion"], cfg["numerics"])
        self.hop_seconds = analyzer.hop / float(fs)
        self.hop_frames = int(analyzer.hop)
        self.commit_seconds = float(cfg["realization"]["commit_seconds"])
        if bool(cfg.get("hires", {}).get("enabled", False)):
            # the engine commits in hires steps when the hires realizer is active, so the blocks D
            # is standardised on are the blocks D is actually applied to (audit §8.1 intent)
            self.commit_seconds = float(cfg["hires"].get("commit_seconds", self.commit_seconds))
        self.block_rows = max(1, int(round(self.commit_seconds / max(1e-9, self.hop_seconds))))
        # ---------------------------------------------------------------- fragment vocabulary
        # FRAG_CONTRACT: none of these keys exist in config.py DEFAULTS (a shared file that is not
        # edited here), so they are read with .get and the chosen default is documented in the
        # trace; a project file cannot set them until they are added to DEFAULTS (reported).
        hz = dict(cfg.get("hires", {}) or {})
        self.frag_n_refs = max(12, int(g("frag_references", 12)))       # contract: >= 12 per unit
        self.frag_segment_seconds = float(g("frag_segment_seconds", float(hz.get("min_clip_seconds", 2.0))))
        self.frag_offset_rows = int(g("frag_offset_rows", 0))           # 0 = the whole segment
        self.frag_d_samples = max(1, int(g("frag_discriminator_samples", 2)))
        # the D statistic is a measurement, not part of the generation: it draws from its own
        # seeded stream so that recording it (or changing how many draws it takes) cannot move the
        # job rng and therefore cannot change the music
        self._stat_rng = np.random.default_rng(int(cfg.get("seed", 0)) + 977)
        self.sources: Optional[Sequence[np.ndarray]] = None   # engine.setup: mode.sources = job.sources
        self.goal_rise_seconds = float(hz.get("goal_rise_seconds", 12.0))
        ge = dict(cfg["form"]["goal_exposure"])
        self.goal_free = (ge.get("policy") == "free")
        self.goal_open_cap = 1.0 if self.goal_free else float(ge.get("open_max", 0.0))
        self.model_step_seconds = float(cfg["analysis"].get("model_step_seconds", self.hop_seconds))
        self.model_step_rows = max(1, int(round(self.model_step_seconds / max(1e-9, self.hop_seconds))))
        self.reference_source = "uninitialised"
        self._frag_goal_levels: Optional[np.ndarray] = None
        self._frag_bounds: List[Tuple[int, int]] = []
        # ---------------------------------------------------------------- held realizable ideals
        # HOLD_CONTRACT (2026-09-16, user item c): in hold mode the reference distribution becomes
        # exact PLANS that continue from the current state and stay close to the committed history.
        # None of these keys exist in config.py DEFAULTS (shared file, not edited here), so they are
        # read with .get and the defaults are documented in the trace and in the report.
        self.hold_move_scale = float(g("hold_move_scale", 1.0))          # R4: scales the requested move
        self.hold_candidates = max(2, int(g("hold_reference_candidates", 14)))
        self.hold_band_lo = float(g("hold_band_lo", 0.0))                # band in units of the hold's flutter
        self.hold_band_hi = float(g("hold_band_hi", 16.0))
        self.hold_ratio_min = float(g("hold_move_over_flutter_min", 3.0))  # R4: move vs own flicker
        self.hold_ratio_max = float(g("hold_move_over_flutter_max", 12.0))
        self.hold_level_move = float(g("hold_level_move", 0.7))          # level gesture (x o x scale x mag)
        self.hold_level_floor = float(g("hold_level_floor", 0.3))        # lowest drawn target level
        self.hold_magnitude_range = (float(g("hold_magnitude_min", 0.3)), float(g("hold_magnitude_max", 1.5)))
        self.hold_jump_probability = float(g("hold_jump_probability", 0.35))
        self.hold_search_rows = max(2, int(g("hold_search_rows", 4)))    # extra rows used while screening
        self.hold_keep_min = max(1, int(g("hold_keep_min", 2)))
        self.hold_keep_max = max(1, int(g("hold_keep_max", 3)))
        self.hold_fragment_pool = max(1, int(g("hold_fragment_pool", 4)))
        self.hold_steady_factor = max(1, int(g("hold_steadiness_factor", 4)))
        self.hold_max_jumps = max(1, int(g("hold_max_jumps_per_plan", 3)))  # cap for M = 9 ... 25
        self.hold_w_sigma_init = float(g("hold_w_sigma_init", 0.08))     # level-space displacement w
        self.hold_w_sigma_min = float(g("hold_w_sigma_min", 0.01))
        self.hold_w_sigma_max = float(g("hold_w_sigma_max", 0.30))
        self.hold_flutter_floor = float(g("hold_flutter_floor", 1e-3))
        self.hold_slots = tuple(self.HOLD_SLOTS)          # replaced by HOLD_SLOTS_CAP when capped
        self.hold_slot_share = tuple(self.HOLD_SLOT_SHARE)
        self.hold_cap = 0
        self.hold_seconds = 0.0           # length of one hold (from realizer_state)
        n_slot, n_w = len(self.hold_slots), max(1, self.M - 1)
        self.h_logits = np.zeros(n_slot)
        self.h_mu = np.zeros((n_slot, n_w))
        self.h_log_sigma = np.full((n_slot, n_w), np.log(max(self.hold_w_sigma_init, 1e-9)))
        self.h_inheritance = "uninitialised"
        self._hold_seen = False           # hold mode has been active at least once in this job
        self._hold_active = False         # hold mode is active in this unit
        self._hold: Optional[Dict[str, Any]] = None
        self._hold_counter = 0
        self._hold_records: List[Dict[str, Any]] = []
        self.hold_traces: List[Dict[str, Any]] = []
        self._committed_positions: List[List[int]] = [[] for _ in range(self.M)]
        self._flutter_ema: Optional[float] = None
        self._realized_flutter: Optional[float] = None
        # ------------------------------------------------ real data (docs/FIDELITY_CONTRACT.md, C)
        # The positives of D are no longer plans at all: they are the exact composition rows of
        # UNMANIPULATED playback - every track plays on continuously from a random position at a
        # constant level (levels drawn from the committed ones, the goal track on its schedule).
        # D therefore learns how the actual recordings MOVE, and the feature map carries explicit
        # block-to-block motion terms.  `hold_positive_source = "plans"` restores the previous law.
        self.hold_positive_source = str(g("hold_positive_source", "recordings"))
        if self.hold_positive_source not in ("recordings", "plans"):
            self.hold_positive_source = "recordings"
        self.hold_real_positives = max(2, int(g("hold_real_positives", 8)))      # >= 8 windows / hold
        self.hold_psi_blocks = max(2, int(g("hold_psi_blocks", 4)))              # commit blocks in psi
        self.hold_psi_scale_samples = max(4, int(g("hold_psi_scale_samples", 24)))
        # positives are drawn mostly NEAR what is playing, so that "where" (mean spectrum,
        # contributions) stops giving the answer away and D has to judge how the sound MOVES
        self.hold_real_near_share = float(g("hold_real_near_share", 0.8))
        self.hold_real_offset_seconds = float(g("hold_real_offset_seconds", 4.0))
        self.hold_logit_clip = float(g("hold_logit_clip", 6.0))
        self.hold_d_l2 = float(g("hold_d_l2", 1e-3))
        # the hold reward is bounded (a clipped logit, order 1) instead of -log D (order 13),
        # so the REINFORCE step size of the hold generator is scaled up to stay effective
        self.hold_lr_G_scale = float(g("hold_lr_G_scale", 20.0))
        self.hold_real_cap_redraw = float(g("hold_real_cap_redraw", 0.5))  # cap only
        self.hold_psi_scale_min_var = float(g("hold_psi_scale_min_var", 0.25))
        hn = ([f"mean_phi[{k}]" for k in range(self.d_phi)]
              + [f"sd_phi[{k}]" for k in range(self.d_phi)]
              + [f"dblock_phi[{k}]" for k in range(self.d_phi)]
              + [f"mean_c[{i}]" for i in range(self.M)]
              + [f"dblock_c[{i}]" for i in range(self.M)]
              + ["mean_R", "sd_R", "n_eff", "hop_dist2", "block_dist2",
                 "within_block_dist2", "span_dist2", "bias"])
        self.h_psi_names = hn
        self.d_psi_hold = len(hn)
        self.h_phi = np.zeros(self.d_psi_hold)
        self.h_psi_mean = np.zeros(self.d_psi_hold)
        self.h_psi_std = np.ones(self.d_psi_hold)
        self.h_psi_scale_source = "uninitialised"
        self.h_psi_scale_n = 0
        self.h_psi_provisional = False
        self.h_psi_scale_varying_fraction = 0.0
        self._committed_levels: List[np.ndarray] = []
        self._hist_blocks: List[Tuple[np.ndarray, np.ndarray]] = []   # rolling committed blocks
        # psi layout (eq. 42): fixed for the whole job, so phi carries over between units
        oi, oj = [], []
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    oi.append(i)
                    oj.append(j)
        self.pair_i = np.array(oi, dtype=int)
        self.pair_j = np.array(oj, dtype=int)
        iu = analyzer.iu
        names = ([f"mean_phi[{k}]" for k in range(self.d_phi)]
                 + [f"halfdiff_phi[{k}]" for k in range(self.d_phi)]
                 + [f"mean_c[{i}]" for i in range(self.M)]
                 + [f"mean_R[{int(a)},{int(b)}]" for a, b in zip(iu[0], iu[1])]
                 + [f"lag_c[{i}]xc[{j}]" for i, j in zip(oi, oj)]
                 + ["bias"])
        self.psi_names = names
        self.d_psi = len(names)
        self.phi = np.zeros(self.d_psi)              # discriminator parameters (eq. 42)
        self.n_parents = self.n_hist_parents + self.n_fresh_parents
        self.r = self.basis_rank
        # generator parameters Theta (eq. 41) - re-shaped per unit, initialised here for safety
        self.logits = np.zeros(self.n_parents)
        self.mu = np.zeros((self.n_parents, self.r))
        self.log_sigma = np.full((self.n_parents, self.r), np.log(max(self.sigma_init, 1e-6)))
        self.parent_ids: List[str] = []
        self.parent_origins: List[str] = []
        self.parent_xi = np.zeros((self.n_parents, 1, self.d_xi))
        self.psi_mean = np.zeros(self.d_psi)
        self.psi_std = np.ones(self.d_psi)
        self.psi_scale_source = "uninitialised"
        self.psi_scale_blocks = 0
        self.B_ref = np.zeros((self.r, 1, self.d_xi))
        self.basis_hash = ""
        self.basis_source = "uninitialised"
        self.basis_sv: List[float] = []
        self.basis_norms: List[float] = []
        self.inheritance: Dict[str, str] = {}
        self.inheritance_detail: Dict[str, Any] = {}
        self.extra_candidate_evaluations = 0
        self.loss_ema: Optional[float] = None       # baseline b_past of eq. (45), across units
        # traces
        self.generator_traces: List[Dict[str, Any]] = []
        self.discriminator_traces: List[Dict[str, Any]] = []
        self.update_counts: List[Dict[str, Any]] = []
        self.reference_summaries: List[Dict[str, Any]] = []
        self.statistics: List[Dict[str, Any]] = []       # per-unit D / displacement statistics
        self._unit_stats: List[Dict[str, Any]] = []      # per-committed-step statistics
        self.lineage: List[Dict[str, Any]] = []
        self._g_buffer: List[Dict[str, Any]] = []
        self._pending: List[Dict[str, Any]] = []
        self._commit_records: List[Dict[str, Any]] = []
        self._unit_counts = {"D": 0, "G": 0, "D_rejected": 0, "update_units": 0, "skipped_blocks": 0}
        self._total_counts = {"D": 0, "G": 0, "D_rejected": 0, "update_units": 0, "skipped_blocks": 0}
        self._step_index = 0

    # ================================================================== psi / D (eq. 42)
    def _psi_raw(self, xi_rows: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Interaction features of one composition on the given unit rows (free windows only):
        band/energy means, half-differences, contribution means, relation means, lagged products."""
        x = np.asarray(xi_rows, dtype=np.float64)
        if rows is None:
            x = x[self.fm_idx]
        else:
            fm = self.unit.free_mask[np.asarray(rows)]
            if fm.any():
                x = x[fm]
        phi_b, c, Rup = self.analyzer.split(x)
        n = len(x)
        mean_phi = phi_b.mean(axis=0)
        if n >= 2:
            h = max(1, n // 2)
            half = phi_b[:h].mean(axis=0) - phi_b[h:].mean(axis=0)
            lag = np.einsum("ti,tj->ij", c[1:], c[:-1]) / float(n - 1)
        else:
            half = np.zeros(self.d_phi)
            lag = np.outer(c[0], c[0])
        inter = lag[self.pair_i, self.pair_j]
        return np.concatenate([mean_phi, half, c.mean(axis=0), Rup.mean(axis=0), inter, [1.0]])

    def _psi(self, xi_rows: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        return (self._psi_raw(xi_rows, rows) - self.psi_mean) / self.psi_std

    def _unit_blocks(self, unit: UnitContext) -> List[np.ndarray]:
        """The unit grid cut into commit-length blocks (the rows D actually ever sees)."""
        out = []
        for s in range(0, unit.J, self.block_rows):
            blk = np.arange(s, min(s + self.block_rows, unit.J))
            if int(unit.free_mask[blk].sum()) >= 2:
                out.append(blk)
        return out

    def discriminator_score(self, xi_rows: np.ndarray, rows: Optional[np.ndarray] = None) -> float:
        """D_phi(Xi) in (0, 1) for a *realized* composition on the given rows (eq. 42)."""
        return float(_sigmoid(float(self.phi @ self._psi(xi_rows, rows))))

    def _d_block_mean(self, xi_full: np.ndarray) -> float:
        """Mean D over the unit's commit-length blocks (the scale D is trained on)."""
        blocks = getattr(self, "unit_blocks", None) or []
        if not blocks:
            return self.discriminator_score(xi_full)
        return float(np.mean([self.discriminator_score(xi_full[b], b) for b in blocks]))

    def _log_p(self, b: int, w: np.ndarray) -> float:
        """log p_Theta(b, w) = log alpha_b + log N(w; mu_b, diag sigma_b^2)."""
        a = self.alphas()
        sig = np.maximum(np.exp(self.log_sigma[b]), 1e-12)
        z = (np.asarray(w, dtype=np.float64) - self.mu[b]) / sig
        return float(np.log(max(float(a[b]), 1e-300)) - 0.5 * float((z * z).sum())
                     - float(np.log(sig).sum()) - 0.5 * self.r * np.log(2.0 * np.pi))

    def _d_loss(self, Psi_ref: np.ndarray, w_ref: np.ndarray, Psi_gen: np.ndarray,
                phi: np.ndarray, l2: Optional[float] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """Eq. (43) with the reference weights r_b on the positive term."""
        D_ref = _sigmoid(Psi_ref @ phi)
        D_gen = _sigmoid(Psi_gen @ phi) if len(Psi_gen) else np.zeros(0)
        pos = -float((w_ref * np.log(np.clip(D_ref, 1e-12, 1.0))).sum())
        neg = -float(np.log(np.clip(1.0 - D_gen, 1e-12, 1.0)).mean()) if len(D_gen) else 0.0
        lam = self.l2 if l2 is None else float(l2)
        return pos + neg + lam * float(phi @ phi), D_ref, D_gen

    # ================================================================== 10.2 reference set
    def fragment_mode(self) -> bool:
        """True when the fragment vocabulary is available: the analyzer carries the fragment bank
        (`hires.fragment_vocabulary`) and the engine has handed the PCM to the mode
        (`engine.setup`: `mode.sources = job.sources`; FRAG_CONTRACT)."""
        return (getattr(self.analyzer, "frag_f", None) is not None
                and getattr(self, "sources", None) is not None)

    def _goal_levels(self, unit: UnitContext) -> np.ndarray:
        """The goal track's level on this unit's rows under the form's exposure policy - the same
        deterministic schedule the hires realizer plays (hires.run_unit_hires): descend from the
        inherited 1.0 to `open_max` over goal_rise_seconds when the unit opens in REOPEN, hold,
        rise smoothly (Q5) to 1 over the last goal_rise_seconds before the goal arrival, 1 inside
        GOAL_HOLD.  A reference must not expose the goal where the realizer cannot."""
        c = unit.centers.astype(np.float64)
        rise = max(1.0, self.goal_rise_seconds * float(self.fs))

        def q5(s: np.ndarray) -> np.ndarray:
            s = np.clip(s, 0.0, 1.0)
            return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))

        reopen = unit.phase_frames.get("REOPEN")
        if reopen is not None and int(reopen[0]) <= int(unit.start):
            # the unit inherits the goal at 1.0 from the preceding GOAL_HOLD and descends
            desc = max(1.0, min(rise, float(reopen[1] - reopen[0])))
            gl = 1.0 + (self.goal_open_cap - 1.0) * q5((c - float(unit.start)) / desc)
        else:
            gl = np.full(unit.J, 0.0 if not self.goal_free else float(self.goal_open_cap))
        if unit.goal_arrival is not None:
            t1 = float(unit.goal_arrival)
            r0 = t1 - rise
            up = c >= r0
            if up.any():
                gl[up] = gl[up] + (1.0 - gl[up]) * q5((c[up] - r0) / rise)
        gl[unit.hold_mask] = 1.0
        return np.clip(gl, 0.0, 1.0)

    def _segment_bounds(self, unit: UnitContext) -> List[Tuple[int, int]]:
        """The unit cut into clip-length row blocks: one new random fragment composition per block
        (the realizer may jump a track once per `min_clip_seconds`, so this is its own time grid)."""
        n = max(1, int(round(self.frag_segment_seconds / max(1e-9, self.hop_seconds))))
        return [(a, min(a + n, unit.J)) for a in range(0, unit.J, n)]

    def _fragment_reference(self, unit: UnitContext, bounds: List[Tuple[int, int]],
                            gl: np.ndarray) -> Dict[str, Any]:
        """One reference trajectory: a sequence of random fragment compositions along the unit.

        Per block the exact mixture rows of "every track plays from its sampled position at its
        sampled level" are computed for `k` hop-spaced offsets after the position (what actually
        sounds after a jump); the remaining rows of the block are filled by linear interpolation
        toward the next block's first sampled row (held after the last one).  GOAL_HOLD rows are
        pinned to xi_goal.  Returns the (J, d) trajectory, the parts E_form needs and the record of
        what was sampled."""
        an = self.analyzer
        xi = np.zeros((unit.J, self.d_xi))
        E = np.zeros(unit.J)
        filled = np.zeros(unit.J, dtype=bool)
        segs: List[Dict[str, Any]] = []
        for (a, b) in bounds:
            k = (b - a) if self.frag_offset_rows <= 0 else min(b - a, self.frag_offset_rows)
            k = max(1, k)
            offsets = np.arange(k, dtype=np.int64) * self.hop_frames
            xi_s, parts_s, meta = an.random_fragment_composition(
                self.sources, self.rng, offsets, goal_level=float(gl[a]))
            xi[a:a + k] = xi_s
            E[a:a + k] = np.asarray(parts_s["E"], dtype=np.float64)
            filled[a:a + k] = True
            segs.append({"rows": [int(a), int(b)], "sampled_rows": int(k),
                         "positions": [int(p) for p in meta["positions"]],
                         "levels": [round(float(v), 4) for v in meta["levels"]],
                         "goal_level": round(float(gl[a]), 4)})
        idx = np.where(filled)[0]
        miss = np.where(~filled)[0]
        if len(miss) and len(idx):
            for d in range(self.d_xi):
                xi[miss, d] = np.interp(miss, idx, xi[idx, d])
            E[miss] = np.interp(miss, idx, E[idx])
        xi = fix_hold_rows(unit, xi)
        gp = getattr(an, "goal_parts_all", None)
        if gp is not None and unit.hold_mask.any():
            E[unit.hold_mask] = np.asarray(gp["E"])[unit.idx][unit.hold_mask]
        parts = {"c": an.split(xi)[1], "E": E}
        return {"xi": xi, "parts": parts, "segments": segs, "sampled_rows": int(filled.sum()),
                "interpolated_rows": int(len(miss))}

    def _draw_references(self, unit: UnitContext, history) -> List[Dict[str, Any]]:
        """The reference distribution of eq. (40).

        Fragment mode: >= `frag_references` random fragment composition trajectories (seeded from
        the job rng, recorded).  Otherwise: the private Bank reference distribution of the baseline
        (legal random gain plans) - unchanged."""
        if self.fragment_mode():
            gl = self._goal_levels(unit)
            bounds = self._segment_bounds(unit)
            out: List[Dict[str, Any]] = []
            for b in range(self.frag_n_refs):
                r = self._fragment_reference(unit, bounds, gl)
                r["e_form"] = float(self.objective.e_form(unit, r["xi"], r["parts"]))
                r["e_hist"] = float(self.objective.e_hist(unit, r["xi"], history))
                r["origin"] = "random_fragment_composition_sequence"
                r["index"] = int(b)
                out.append(r)
            self._frag_goal_levels = gl
            self._frag_bounds = bounds
            # one reference = one fragment composition per block: that is what was evaluated
            self.extra_candidate_evaluations = int(sum(len(r["segments"]) for r in out))
            return out
        bank = Bank(unit, self.lim, self.cfg, self.rng, self.objective, history)
        refs: List[Candidate] = []
        attempts = 0
        max_attempts = 4 * self.n_ref_target + 8
        while len(refs) < self.n_ref_target and attempts < max_attempts:
            attempts += 1
            c = bank.random_candidate(None, origin="gan_reference")
            if c is not None:
                refs.append(c)
        if not refs:
            # legal reference trajectories are a hard feasibility problem, not a soft criterion
            raise NoFeasibleReference(
                f"NO_FEASIBLE_PLAN_FOUND: unit {unit.index}: no legal reference composition "
                f"trajectory for the GAN reference distribution in {attempts} attempts "
                f"(bank rejected {bank.n_failed} illegal plans)")
        if len(refs) < min(4, self.n_ref_target):
            self.warnings.append(
                f"unit {unit.index}: only {len(refs)} legal reference trajectories "
                f"(target {self.n_ref_target}); proceeding with a small reference distribution")
        self._frag_goal_levels = None
        self._frag_bounds = []
        self.extra_candidate_evaluations = len(refs)
        return [{"xi": c.xi, "e_form": float(c.e_form), "e_hist": float(c.e_hist),
                 "origin": "bank_random_legal_plan", "index": int(k), "segments": [],
                 "sampled_rows": int(unit.J), "interpolated_rows": 0} for k, c in enumerate(refs)]

    def _reference_motion(self, unit: UnitContext) -> Dict[str, Any]:
        """How fast the reference distribution itself moves, in d_xi units (eq. 22 per row pair):
        mean d_xi^2 between free rows one model step apart and one analysis hop apart."""
        fm = unit.free_mask
        out: Dict[str, Any] = {"model_step_seconds": self.model_step_seconds,
                               "model_step_rows": int(self.model_step_rows)}
        for tag, s in (("step", int(self.model_step_rows)), ("hop", 1)):
            vals = []
            if unit.J > s:
                m = fm[s:] & fm[:-s]
                if m.any():
                    vals = [float(self.analyzer.dist2(x[s:], x[:-s])[m].mean()) for x in self.ref_xi]
            out[f"reference_{tag}_dist2_mean"] = float(np.mean(vals)) if vals else None
            out[f"reference_{tag}_displacement_mean"] = float(np.mean(np.sqrt(vals))) if vals else None
        return out

    # ================================================================== hold mode (HOLD_CONTRACT)
    # The engine holds one ideal for `hires.reference_hold_seconds` and publishes
    # `unit.realizer_state` with an exact plan evaluator.  The ideal must then be the exact rows of
    # a REALIZABLE plan, and (user item c) the reference distribution must be narrowed to
    # continuations that stay close to the committed history instead of unrelated random mixes.
    #
    #   reference plan  = level moves on the commit grid of the hold (scale ∝ hold_move_scale ∝ o)
    #                     plus, for tracks whose minimum clip length allows it, ONE jump to a
    #                     position of its origin class (slot);
    #   narrowing       = keep the plans whose rows lie in a distance band [d_min, d_max] (d_xi²)
    #                     around the anchor, the mean of the last committed block - i.e. neither
    #                     the unchanged sound nor an unrelated mix.  The band is expressed in units
    #                     of the hold's OWN flutter (measured from the unchanged continuation), so
    #                     it follows the material instead of a hard-coded number;
    #   weights r_b     = eq. (40) among the kept plans (J_ref = E_form + 0.2 E_hist at tau_ref);
    #   generator       = p_Theta(b, w): a categorical over the slots (logits carried across holds
    #                     and units) times the eq. (40) weights inside the drawn slot, and a
    #                     Gaussian displacement w of the MATERIAL LEVELS of the plan's steps
    #                     (clipped into [0, 1] - realizable by construction).  The xi-space
    #                     displacement B_ref w of eq. (41) is NOT added here: it is not realizable.
    def _restore_hold_generator(self, history) -> None:
        """Generator parameters of the hold law at the first hold of a unit.

        Unlike the coefficients w of eq. (41) - whose basis B_ref is rebuilt per unit, so they are
        re-initialised - the hold displacement lives in LEVEL space, which has the same acoustic
        meaning in every unit (the material gains).  Slot logits, mu and log_sigma are therefore
        inherited from `history.mode_state['gan']['hold_generator']` when they are usable."""
        n_slot, n_w = len(self.hold_slots), max(1, self.M - 1)
        self.h_logits = np.zeros(n_slot)
        self.h_mu = np.zeros((n_slot, n_w))
        self.h_log_sigma = np.full((n_slot, n_w), np.log(max(self.hold_w_sigma_init, 1e-9)))
        got: List[str] = []
        st = history.mode_state.get("gan") if hasattr(history, "mode_state") else None
        hg = st.get("hold_generator") if isinstance(st, dict) else None
        if isinstance(hg, dict):
            lg = np.asarray(hg.get("logits", []), dtype=np.float64)
            mu = np.asarray(hg.get("mu", []), dtype=np.float64)
            ls = np.asarray(hg.get("log_sigma", []), dtype=np.float64)
            if lg.shape == (n_slot,) and np.all(np.isfinite(lg)):
                self.h_logits = lg
                got.append("logits")
            if mu.shape == (n_slot, n_w) and np.all(np.isfinite(mu)):
                self.h_mu = mu
                got.append("mu")
            if ls.shape == (n_slot, n_w) and np.all(np.isfinite(ls)):
                self.h_log_sigma = np.clip(ls, np.log(self.hold_w_sigma_min), np.log(self.hold_w_sigma_max))
                got.append("log_sigma")
        self.h_inheritance = ("inherited: " + ", ".join(got) if got else
                              "cold (logits 0, mu 0, sigma = hold_w_sigma_init)")

    # ---------------------------------------------------------------- real data / motion features
    def _hold_psi_raw(self, xi_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Hold-mode feature map psi_h: WHERE the sound is and HOW it moves.

        The rows of one update window are cut into commit blocks; besides the usual level of
        description (band/energy means, contributions, relation and energy scalars) the map carries
        explicit MOTION terms - the mean absolute block-to-block change of the band features and of
        the contributions, the hop and block distances, the within-block motion (the flutter that
        splices and ramps produce) and the span of the window.  Every term is either fixed-length in
        the bands or linear in the number of tracks, so it survives M = 9 ... 25 unchanged."""
        an = self.analyzer
        x = np.asarray(xi_rows, dtype=np.float64)
        rows = np.asarray(rows)
        fm = self.unit.free_mask[rows]
        if fm.any() and not fm.all():
            x, rows = x[fm], rows[fm]
        n = len(x)
        phi, c, Rup = an.split(x)
        nb_ = max(1, int(self.block_rows))
        edges = list(range(0, n, nb_))
        Bm = np.stack([x[a:min(a + nb_, n)].mean(axis=0) for a in edges]) if n else x
        bphi, bc, _bR = an.split(Bm)
        if len(Bm) >= 2:
            dphi = np.abs(np.diff(bphi, axis=0)).mean(axis=0)
            dc = np.abs(np.diff(bc, axis=0)).mean(axis=0)
            blk_d2 = float(an.dist2(Bm[1:], Bm[:-1]).mean())
            span = float(an.dist2(Bm[-1][None, :], Bm[0][None, :])[0])
        else:
            dphi = np.zeros(self.d_phi)
            dc = np.zeros(self.M)
            blk_d2 = span = 0.0
        hop_d2 = float(an.dist2(x[1:], x[:-1]).mean()) if n >= 2 else 0.0
        wit = [float(an.dist2(x[a:min(a + nb_, n)], x[a:min(a + nb_, n)].mean(axis=0)[None, :]).mean())
               for a in edges if min(a + nb_, n) - a >= 2]
        within = float(np.mean(wit)) if wit else 0.0
        nong = c[:, 1:].sum(axis=1)
        p = c[:, 1:] / (nong[:, None] + 1e-12)
        n_eff = float(np.where(nong > 1e-9, 1.0 / (np.sum(p * p, axis=1) + 1e-12), 0.0).mean())
        # (log_energy == mean_phi[0] and goal_c == mean_c[0] were redundant and are gone)
        return np.concatenate([phi.mean(axis=0), phi.std(axis=0), dphi, c.mean(axis=0), dc,
                               [float(Rup.mean()), float(Rup.std()), n_eff,
                                hop_d2, blk_d2, within, span, 1.0]])

    def _hold_psi(self, xi_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return (self._hold_psi_raw(xi_rows, rows) - self.h_psi_mean) / self.h_psi_std

    def _draw_levels_like_committed(self, rng, levels_now: np.ndarray) -> np.ndarray:
        """A constant level vector drawn like the committed ones, so that the level alone does not
        tell a real window from the committed sound (the whole committed vector is re-used, which
        keeps the joint distribution over the tracks)."""
        if self._committed_levels:
            k = int(rng.integers(0, len(self._committed_levels)))
            lv = np.asarray(self._committed_levels[k], dtype=np.float64).copy()
            if len(lv) == self.M:
                return np.clip(lv, 0.0, 1.0)
        return np.clip(np.asarray(levels_now, dtype=np.float64).copy(), 0.0, 1.0)

    def _hold_real_windows(self, unit: UnitContext, rs: Dict[str, Any], rows: np.ndarray,
                           n: int, rng, anchor: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """`n` exact composition trajectories of UNMANIPULATED playback on `rows`: every track plays
        on continuously from ONE position of its own source at a constant level, the goal track keeps
        its position and its scheduled gain.  Computed with the analyzer's exact Gram machinery
        (`grams_at_positions` / `material_features_at` / `composition_from_grams`), exactly as the
        realized composition is computed - nothing here is a model of the sound.

        The positions are drawn mostly NEAR what is playing (the current position continued, a
        recently committed position, or a fragment that `fragment_candidates_mix` ranks close to the
        recent committed mixture - each shifted by a random offset of a few seconds) and only a
        minority uniformly over the source.  Otherwise D could answer "real or not" from the mean
        spectrum alone instead of from how the sound moves.  One extra leading row is evaluated and
        dropped so that the first row carries a real spectral flux and not the 0 of a sequence
        start - the same artefact the engine now removes on the committed side."""
        an = self.analyzer
        src = self.sources
        rows = np.asarray(rows)
        pre = int(rows[0]) - 1
        ext = np.concatenate([[pre], rows]) if pre >= 0 else rows
        starts = (unit.centers[ext] - an.W // 2).astype(np.int64)
        d = (starts - starts[0]).astype(np.int64)
        rs_rows = np.asarray(rs["rows"])
        gg = np.asarray(rs.get("goal_gains", np.zeros(len(rs_rows))), dtype=np.float64)
        k = np.searchsorted(rs_rows, ext)
        goal = (gg[np.clip(k, 0, len(gg) - 1)] if len(gg) == len(rs_rows) else np.zeros(len(ext)))
        lv_now = np.asarray(rs["levels"], dtype=np.float64)
        pos_now = np.asarray(rs["positions"], dtype=np.int64)
        off = max(1, int(round(self.hold_real_offset_seconds * self.fs)))
        tgt = np.asarray(anchor, dtype=np.float64) if anchor is not None else None
        mixed = getattr(an, "fragment_candidates_mix", None)
        out: List[np.ndarray] = []
        src_counts = {"current": 0, "committed": 0, "mixture_candidate": 0, "random": 0}
        for _ in range(int(n)):
            # a random LEGAL state: the committed level magnitudes, but which materials sound is
            # redrawn within the polyphony cap (no cap: returned unchanged, no draw - the engine's
            # own helper).  Without this D could tell the two sides apart by density alone.
            lv = self._draw_levels_like_committed(rng, lv_now)
            if int(self.hold_cap) > 0 and float(rng.random()) < self.hold_real_cap_redraw:
                # part of the windows keep the committed vector's own (already legal) sounding set,
                # the rest get a freshly drawn legal one - otherwise D can answer "real or not" from
                # WHICH materials sound instead of from how the sound moves
                lv = self.analyzer.cap_random_levels(lv, rng)
            pos = np.zeros((len(ext), self.M), dtype=np.int64)
            for i in range(self.M):
                L = int(src[i].shape[0])
                if i == 0:
                    pos[:, i] = (int(pos_now[i]) + d) % L
                    continue
                kind = "random"
                if float(rng.random()) < self.hold_real_near_share:
                    opts = ["current"]
                    if self._committed_positions[i]:
                        opts.append("committed")
                    if mixed is not None and tgt is not None:
                        opts.append("mixture_candidate")
                    kind = opts[int(rng.integers(0, len(opts)))]
                if kind == "current":
                    base = int(pos_now[i]) + int(rng.integers(-off, off + 1))
                elif kind == "committed":
                    pool = self._committed_positions[i]
                    base = int(pool[int(rng.integers(0, len(pool)))]) + int(rng.integers(-off, off + 1))
                elif kind == "mixture_candidate":
                    others = an.mixture_band_energy(pos_now, lv, skip=i)
                    cands = mixed(i, tgt[: 1 + an.nb], others, float(lv[i]), 4)
                    base = int(cands[int(rng.integers(0, len(cands)))]) + int(rng.integers(-off, off + 1))
                else:
                    base = int(rng.integers(0, L))
                src_counts[kind] += 1
                pos[:, i] = (int(base) + d) % L
            g = np.repeat(lv[None, :], len(ext), axis=0)
            g[:, 0] = goal
            _f, S, _c = an.material_features_at(pos)
            G0, Gb = an.grams_at_positions(src, starts, pos)
            xi = an.composition_from_grams(g, G0, Gb, S)[0]
            out.append(xi[1:] if pre >= 0 else xi)
        self._real_position_sources = src_counts
        return out

    def _hold_psi_scales(self, history, samples: List[np.ndarray], rows: np.ndarray) -> None:
        """Standardisation of psi_h: computed ONCE, from a sample of real (unmanipulated) windows of
        the first hold of the job, then frozen so that the inherited h_phi keeps its meaning."""
        st = history.mode_state.get("gan") if hasattr(history, "mode_state") else None
        hd = st.get("hold_discriminator") if isinstance(st, dict) else None
        if isinstance(hd, dict):
            m = np.asarray(hd.get("psi_mean", []), dtype=np.float64)
            s = np.asarray(hd.get("psi_std", []), dtype=np.float64)
            f = np.asarray(hd.get("phi", []), dtype=np.float64)
            if (m.shape == (self.d_psi_hold,) and s.shape == (self.d_psi_hold,) and np.all(np.isfinite(m))
                    and not bool(hd.get("psi_scale_provisional", False))):
                self.h_psi_mean, self.h_psi_std = m, np.maximum(s, 1e-6)
                self.h_psi_scale_source = "frozen_from_the_first_hold"
                self.h_psi_scale_n = int(hd.get("psi_scale_n", 0))
                if f.shape == (self.d_psi_hold,) and np.all(np.isfinite(f)):
                    self.h_phi = f
                return
        P = np.stack([self._hold_psi_raw(x, rows) for x in samples])
        m = P.mean(axis=0)
        raw = P.std(axis=0)
        s = np.maximum(raw, 1e-3)
        m[-1] = 0.0
        s[-1] = 1.0
        self.h_psi_mean, self.h_psi_std = m, s
        self.h_psi_scale_n = int(len(samples))
        frac = float(np.mean(raw[:-1] > 1e-3))
        # A hold that starts in silence produces 24 identical windows: every feature would then be
        # standardised by the floor 1e-3 and D would see gradients ~10^3 that its backtracking line
        # search rejects.  The scales therefore stay PROVISIONAL until an informative hold appears
        # (with or without a polyphony cap: the piece starts from silence either way).
        if frac < self.hold_psi_scale_min_var:
            self.h_psi_scale_source = f"provisional (only {frac:.2f} of the features vary in this hold)"
            self.h_psi_provisional = True
            self.h_phi = np.zeros(self.d_psi_hold)
        else:
            self.h_psi_scale_source = "real_unmanipulated_windows_of_the_first_hold"
            self.h_psi_provisional = False
        self.h_psi_scale_varying_fraction = frac

    def _cap_statistics(self, hr: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Polyphony-cap figures per unit: how often the sounding SET changed (swaps per minute),
        the share of holds at the cap, and D split by holds whose published plan swaps - with a
        silent entry the audible-splice cue disappears, so the two groups answer different
        questions about what the discriminator is reading."""
        caps = [h["cap"] for h in hr if h.get("cap")]
        if not caps:
            return {"max_active_materials": int(self.hold_cap)}
        hold_seconds = float(getattr(self, 'hold_seconds', 0.0) or 2.0)
        n_sw = int(sum(1 for c in caps if c.get("published_swap")))
        act = [int(c["active_materials_at_hold_start"]) for c in caps]
        d_sw = [float(c["D_actual_after_D_step"]) for c in self._commit_records
                if isinstance(c.get("hold"), dict) and c["hold"].get("swap")]
        d_no = [float(c["D_actual_after_D_step"]) for c in self._commit_records
                if isinstance(c.get("hold"), dict) and not c["hold"].get("swap")
                and "D_actual_after_D_step" in c]
        dr_sw = [float(h["real_positives"]["D_real_windows_mean"]) for h in hr
                 if h.get("real_positives") and (h.get("cap") or {}).get("published_swap")]
        dr_no = [float(h["real_positives"]["D_real_windows_mean"]) for h in hr
                 if h.get("real_positives") and not (h.get("cap") or {}).get("published_swap")]
        dp_sw = [float(h["real_positives"]["D_published_plan"]) for h in hr
                 if h.get("real_positives") and (h.get("cap") or {}).get("published_swap")
                 and h["real_positives"].get("D_published_plan") is not None]
        dp_no = [float(h["real_positives"]["D_published_plan"]) for h in hr
                 if h.get("real_positives") and not (h.get("cap") or {}).get("published_swap")
                 and h["real_positives"].get("D_published_plan") is not None]
        return {
            "max_active_materials": int(self.hold_cap),
            "holds": len(caps), "holds_with_swap": n_sw,
            "swaps_per_minute": float(n_sw / max(1e-9, len(caps) * hold_seconds / 60.0)),
            "share_at_cap": float(np.mean([1.0 if c["at_cap"] else 0.0 for c in caps])),
            "active_materials_mean": float(np.mean(act)) if act else None,
            "active_materials_share": {str(k): float(np.mean([1.0 if a_ == k else 0.0 for a_ in act]))
                                       for k in sorted(set(act))},
            "swap_candidates_per_hold": float(np.mean([c["swap_candidates_drawn"] for c in caps])),
            "swap_candidates_kept_per_hold": float(np.mean([c["swap_candidates_kept"] for c in caps])),
            "D_committed_swap": (float(np.mean(d_sw)) if d_sw else None),
            "D_committed_no_swap": (float(np.mean(d_no)) if d_no else None),
            "D_real_windows_swap": (float(np.mean(dr_sw)) if dr_sw else None),
            "D_real_windows_no_swap": (float(np.mean(dr_no)) if dr_no else None),
            "D_published_plan_swap": (float(np.mean(dp_sw)) if dp_sw else None),
            "D_published_plan_no_swap": (float(np.mean(dp_no)) if dp_no else None),
            "note": "a swap enters a silent material on a voice that jumps while still silent, so "
                    "those holds carry no audible splice; comparing the two groups says how much of "
                    "what D reads is the splice and how much is the level motion",
        }

    def _h_phi_split(self) -> Dict[str, Any]:
        """How much of the discriminator's weight norm sits on WHERE the sound is and how much on
        HOW it moves - the honest read-out of what D actually separates on."""
        motion = ("sd_phi", "dblock_phi", "dblock_c", "hop_dist2", "block_dist2",
                  "within_block_dist2", "span_dist2", "sd_R")
        g2: Dict[str, float] = {}
        for nme, v in zip(self.h_psi_names, self.h_phi):
            g2[nme.split("[")[0]] = g2.get(nme.split("[")[0], 0.0) + float(v) * float(v)
        w = float(np.sqrt(sum(v for k, v in g2.items() if k not in motion and k != "bias")))
        m = float(np.sqrt(sum(v for k, v in g2.items() if k in motion)))
        return {"where": w, "motion": m, "motion_share": (m / max(1e-12, w + m)),
                "by_group": {k: round(float(np.sqrt(v)), 3) for k, v in
                             sorted(g2.items(), key=lambda x: -x[1])},
                "norm": float(np.linalg.norm(self.h_phi))}

    def _hold_alphas(self) -> np.ndarray:
        z = self.h_logits - self.h_logits.max()
        e = np.exp(z)
        return e / e.sum()

    def _hold_log_p(self, slot: int, w: np.ndarray) -> float:
        """log p_Theta(slot, w) = log alpha_slot + log N(w; mu_slot, diag sigma_slot²).

        The choice of one kept plan inside the drawn slot uses the eq. (40) weights r_b, which are
        data and carry no parameter, so they are not part of p_Theta (they cancel in the importance
        ratio).  Likewise the per-hold restriction to the slots that survived the narrowing is
        treated as part of the environment, not of p_Theta."""
        a = self._hold_alphas()
        sig = np.maximum(np.exp(self.h_log_sigma[int(slot)]), 1e-12)
        z = (np.asarray(w, dtype=np.float64) - self.h_mu[int(slot)]) / sig
        return float(np.log(max(float(a[int(slot)]), 1e-300)) - 0.5 * float((z * z).sum())
                     - float(np.log(sig).sum()) - 0.5 * len(sig) * np.log(2.0 * np.pi))

    def _clip_steadiness(self, track: int, positions: Sequence[int]) -> np.ndarray:
        """How much the band profile of each candidate fragment moves WITHIN its clip: the summed
        variance of the normalized band ratios over the solo-bank rows that follow the position.

        Measured on the realized sound, the fast motion the engine divides the requested move by
        (`hold_summary.realized_flutter_dist2`) sits almost entirely in the band block of xi and is
        a property of the material at the played positions - level ramps add ~0.  A reference that
        sends the sound into a fragment whose spectrum flickers therefore raises the denominator of
        its own move.  The reference distribution prefers steady fragments for that reason."""
        an = self.analyzer
        f = getattr(an, "solo_f", None)
        pos = np.asarray(list(positions), dtype=np.int64)
        if f is None or not len(pos):
            return np.zeros(len(pos))
        fi = f[track]
        P = len(fi)
        n = max(2, int(getattr(an, "frag_rows", 2)))
        k = np.clip(np.round(pos / float(an.solo_hop)).astype(np.int64), 0, P - 1)
        idx = (k[:, None] + np.arange(n)[None, :]) % P
        v = fi[idx][:, :, 1:1 + an.nb]
        return v.var(axis=1).sum(axis=1)

    def _steadiest(self, track: int, positions: Sequence[int], n: int) -> List[int]:
        pos = list(dict.fromkeys(int(p) for p in positions))
        if len(pos) <= n:
            return pos
        st = self._clip_steadiness(track, pos)
        return [int(pos[k]) for k in np.argsort(st)[:n]]

    def _hold_position_pools(self, unit: UnitContext, history, anchor: np.ndarray,
                             positions_now: np.ndarray) -> Dict[str, Any]:
        """Positions a reference plan may jump to, per origin class (user item c).

        committed  : `src_position` of the committed events and the playback positions the realizer
                     actually committed earlier in this job (`stats['positions']`) - a jump back to
                     a fragment the piece has already used;
        fragment   : `an.fragment_candidates` for the band profile of the RECENT COMMITTED
                     compositions (history.recent_xi), i.e. fragments that sound like what has just
                     been committed;
        random     : fresh uniform positions (the small exploration share)."""
        an = self.analyzer
        nb = int(an.nb)
        committed: List[List[int]] = [[] for _ in range(self.M)]
        for e in getattr(history, "events", [])[-64:]:
            i = int(e.get("track", -1))
            p = e.get("src_position")
            if 0 < i < self.M and p is not None:
                committed[i].append(int(p))
        for i in range(1, self.M):
            committed[i].extend(int(p) for p in self._committed_positions[i][-24:])
        rec = [x for x in getattr(history, "recent_xi", [])[-4:] if len(x)]
        if rec:
            ratios = np.concatenate(rec, axis=0)[:, 1:1 + nb].mean(axis=0)
        else:
            ratios = np.asarray(anchor, dtype=np.float64)[1:1 + nb]
        frag: List[List[int]] = [[] for _ in range(self.M)]
        if getattr(an, "frag_f", None) is not None:
            for i in range(1, self.M):
                wide = [int(p) for p in an.fragment_candidates(
                    i, ratios, self.hold_fragment_pool * self.hold_steady_factor,
                    exclude_near=int(positions_now[i]), exclude_frames=0)]
                frag[i] = self._steadiest(i, wide, self.hold_fragment_pool)
        for i in range(1, self.M):
            committed[i] = self._steadiest(i, committed[i], self.hold_fragment_pool)
        return {"committed": committed, "fragment": frag, "band_profile_source":
                ("recent committed rows" if rec else "anchor"),
                "n_committed": [len(committed[i]) for i in range(self.M)]}

    def _hold_draw_plan(self, unit: UnitContext, slot: int, mag: float, step_frames: List[int],
                        levels_now: np.ndarray, next_jump: List[int], goal_at, pools: Dict[str, Any],
                        jumps_on: bool, o0: float, cap: int = 0, voices: Optional[Dict[int, List[int]]] = None,
                        sounding: Optional[List[int]] = None, silent: Optional[List[int]] = None,
                        project=None, positions_now: Optional[np.ndarray] = None,
                        anchor: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """One candidate reference plan.

        Levels: a smooth gesture from the current levels toward a drawn target
        `lv + beta (u - lv)`, u ~ U(0,1)^{M-1}, beta = hold_move_scale x mag x hold_level_move x o
        clipped into [0, 1], distributed evenly over the commit steps of the hold - so each step is
        one Q5 ramp the realizer can play and the ideal moves at a constant rate through the hold.
        Jumps: all at the hold start (the plan's first step), each eligible track (minimum clip
        length) with probability `hold_jump_probability`, from the slot's position pool.  Keeping
        the cuts on the hold boundary leaves the later commit blocks of the hold free of material
        discontinuities - the sound's own fast motion (flutter) stays small while the requested
        move over the hold does not.

        With a polyphony cap only the <= K SOUNDING materials get a level gesture and a jump (a
        gradient on a silent material is meaningless), and the `swap` slot is the only way a silent
        material can enter: one sounding material leaves (all its voices to 0) while one voice of a
        silent one comes in at a drawn level, and that entering voice jumps to its fragment at the
        hold start WHILE IT IS STILL SILENT - so the material change happens without an audible
        splice.  The entering fragment is pre-ranked with `an.fragment_candidates_mix` against the
        band energy of the materials that STAY."""
        rng = self.rng
        beta = float(np.clip(self.hold_move_scale * float(mag) * self.hold_level_move * float(o0), 0.0, 1.0))
        jumps0: Dict[int, int] = {}
        n_fallback = 0
        f0 = int(step_frames[0])
        lv0 = np.asarray(levels_now, dtype=np.float64)
        movable = list(range(1, self.M))
        swap: Optional[Dict[str, Any]] = None
        entering: List[int] = []
        if cap > 0 and voices is not None:
            snd = list(sounding or [])
            sil = list(silent or [])
            # free places under the cap (the piece starts from silence, and a material may have been
            # faded out): they are filled without anyone having to leave
            free = int(cap) - len(snd)
            if free > 0 and sil:
                k_ = int(min(free, len(sil)))
                entering = [int(x) for x in rng.choice(np.asarray(sil), size=k_, replace=False)]
                sil = [m for m in sil if m not in entering]
            movable = [i for m in (snd + entering) for i in voices[m]]
            if slot == 4:                       # SWAP: who sounds is the decision
                swap = self._hold_draw_swap(rng, cap, voices, snd, sil, lv0, next_jump, f0,
                                            positions_now, anchor, pools, free > 0)
                if swap is not None:
                    jumps0.update(swap["jumps"])
                    movable = [i for m in (swap["stay"] + entering) for i in voices[m]]
        if jumps_on and slot in (1, 2, 3):
            # which sounding tracks cut at the hold start; capped so that the cost and the number of
            # simultaneous cuts stay bounded when M grows (2 voices per material, 8-12 materials)
            elig = [i for i in movable
                    if f0 >= int(next_jump[i]) and float(rng.random()) < self.hold_jump_probability]
            if len(elig) > self.hold_max_jumps:
                elig = [int(x) for x in rng.choice(np.asarray(elig), size=self.hold_max_jumps, replace=False)]
            for i in elig:
                pool = pools["committed"][i] if slot == 1 else (pools["fragment"][i] if slot == 2 else [])
                if pool:
                    p = int(pool[int(rng.integers(0, len(pool)))])
                else:
                    p = int(rng.integers(0, int(self.sources[i].shape[0])))
                    if slot != 3:
                        n_fallback += 1
                jumps0[int(i)] = int(p)
        # target levels stay above `hold_level_floor`: a mixture whose materials are all near zero
        # has a noisy normalized band profile, and that noise lands in the very fast motion the
        # requested move is measured against (it is also what E_form's energy / N_eff terms guard)
        u = self.hold_level_floor + (1.0 - self.hold_level_floor) * rng.uniform(0.0, 1.0, max(1, self.M - 1))
        target = lv0[1:] + beta * (u - lv0[1:])
        if cap > 0:
            keep = np.zeros(self.M - 1, dtype=bool)
            for i in movable:
                keep[i - 1] = True
            target = np.where(keep, target, lv0[1:])      # silent materials keep their level (0)
        steps: List[Dict[str, Any]] = []
        n = len(step_frames)
        for k, f in enumerate(step_frames):
            lv = lv0.copy()
            lv[1:] = np.clip(lv0[1:] + (float(k + 1) / float(n)) * (target - lv0[1:]), 0.0, 1.0)
            if swap is not None and k >= 1:
                # from the second commit step on: the leaving material is out, the entering voice in
                for i in swap["out_tracks"]:
                    lv[i] = 0.0
                lv[swap["in_track"]] = float(swap["in_level"])
            lv[0] = float(goal_at(f))
            if project is not None:
                lv = np.asarray(project(lv), dtype=np.float64)
            steps.append({"frame": int(f), "jumps": (dict(jumps0) if k == 0 else {}),
                          "levels": [float(x) for x in lv]})
        return {"steps": steps, "slot": int(slot), "magnitude": float(mag), "beta": beta,
                "target_levels": [round(float(x), 3) for x in target],
                "n_jumps": len(jumps0), "pool_fallbacks": int(n_fallback),
                "swap": (None if swap is None else {"out": int(swap["out"]), "in": int(swap["in"]),
                                                    "in_track": int(swap["in_track"]),
                                                    "in_level": round(float(swap["in_level"]), 3),
                                                    "silent_entry": bool(swap["silent_entry"]),
                                                    "position_source": swap["position_source"]})}

    def _hold_draw_swap(self, rng, cap: int, voices: Dict[int, List[int]], sounding: List[int],
                        silent: List[int], lv0: np.ndarray, next_jump: List[int], f0: int,
                        positions_now: Optional[np.ndarray], anchor: Optional[np.ndarray],
                        pools: Dict[str, Any], free_place: bool = False) -> Optional[Dict[str, Any]]:
        """One material out, one silent material in on one voice.  The entering voice jumps at the
        hold start while it is still silent; the fragment is pre-ranked with
        `an.fragment_candidates_mix` for the recent committed profile against the band energy of the
        materials that stay, with a share from the committed history and a random share."""
        an = self.analyzer
        if not silent:
            return None
        if free_place or not sounding:
            # a place under the cap is free: a material enters without anyone leaving
            m_out, stay = -1, list(sounding)
        else:
            m_out = int(sounding[int(rng.integers(0, len(sounding)))])
            stay = [m for m in sounding if m != m_out]
        m_in = int(silent[int(rng.integers(0, len(silent)))])
        vin = voices[m_in]
        i_in = int(vin[int(rng.integers(0, len(vin)))])
        lvl = float(self.hold_level_floor + (1.0 - self.hold_level_floor) * float(rng.random()))
        pos: Optional[int] = None
        src = "none"
        if f0 >= int(next_jump[i_in]):
            mixed = getattr(an, "fragment_candidates_mix", None)
            r = float(rng.random())
            if r < 0.2 and pools["committed"][i_in]:
                pool = pools["committed"][i_in]
                pos = int(pool[int(rng.integers(0, len(pool)))])
                src = "committed"
            elif r < 0.85 and mixed is not None and anchor is not None and positions_now is not None:
                lv_stay = np.zeros(self.M)
                for m in stay:
                    lv_stay[voices[m]] = lv0[voices[m]]
                others = an.mixture_band_energy(positions_now, lv_stay, skip=i_in)
                cands = mixed(i_in, np.asarray(anchor)[: 1 + an.nb], others, lvl, 4)
                pos = int(cands[int(rng.integers(0, len(cands)))])
                src = "mixture_candidate"
            else:
                pos = int(rng.integers(0, int(self.sources[i_in].shape[0])))
                src = "random"
        return {"out": m_out, "in": m_in, "stay": stay, "in_track": i_in, "in_level": lvl,
                "out_tracks": ([int(i) for i in voices[m_out]] if m_out >= 0 else []),
                "jumps": ({i_in: int(pos)} if pos is not None else {}),
                "silent_entry": pos is not None, "position_source": src}

    def _hold_prepare(self, unit: UnitContext, history, rows: np.ndarray, xi_anchor: np.ndarray,
                      n_proposals: int, rs: Dict[str, Any]) -> List[Target]:
        """One hold: draw the reference plans, narrow them to the committed history, weight them by
        eq. (40) and publish the EXACT rows of a plan drawn from p_Theta(b, w)."""
        an = self.analyzer
        rng = self.rng
        plan_eval = rs["plan_rows"]
        rows = np.asarray(rows)
        anchor = np.asarray(xi_anchor, dtype=np.float64)
        if not self._hold_active:
            self._hold_active = True
            self._hold_seen = True
            # polyphony cap (docs/FIDELITY_CONTRACT.md): with a cap the law must also decide WHO
            # sounds, so a fifth origin class (swap) joins the categorical.  cap = 0: nothing here
            # changes, not one extra random number is drawn.
            self.hold_cap = int(rs.get("max_active_materials", 0) or 0)
            if self.hold_cap > 0:
                self.hold_slots = tuple(self.HOLD_SLOTS_CAP)
                self.hold_slot_share = tuple(self.HOLD_SLOT_SHARE_CAP)
            self.hold_seconds = float(int(rs.get("hold_frames", 0)) / float(self.fs)) or 2.0
            self._restore_hold_generator(history)
            self.reference_source = "hold_exact_plans_narrowed_to_the_committed_history"
        hold_id = int(self._hold_counter)
        self._hold_counter += 1
        t0, commit = int(rs["frame"]), max(1, int(rs["commit_frames"]))
        hold_f = max(commit, int(rs["hold_frames"]))
        search_end = int(rs["search_end_frame"])
        levels_now = np.asarray(rs["levels"], dtype=np.float64).copy()
        next_jump = [int(x) for x in rs["next_jump_frame"]]
        positions_now = np.asarray(rs["positions"], dtype=np.int64)
        jumps_on = bool(rs["jumps_enabled"]) and self.M > 1 and getattr(self, "sources", None) is not None
        cap = int(self.hold_cap)
        groups = np.asarray(rs.get("material_of_track", np.arange(self.M)), dtype=np.int64)
        project = rs.get("project_levels")
        voices = {int(m): [i for i in range(1, self.M) if int(groups[i]) == int(m)]
                  for m in sorted({int(groups[i]) for i in range(1, self.M)})}
        sounding = [m for m, vv in voices.items() if float(np.max(levels_now[vv])) > 1e-6]
        silent = [m for m in voices if m not in sounding]
        centers = unit.centers[rows]
        free = unit.free_mask[rows]
        goal_gains = np.asarray(rs.get("goal_gains", np.zeros(len(rows))), dtype=np.float64)

        def goal_at(f: int) -> float:
            k = int(np.clip(np.searchsorted(centers, int(f)), 0, len(rows) - 1))
            return float(goal_gains[k]) if len(goal_gains) == len(rows) else 0.0

        step_frames = [t0 + k * commit for k in range(max(1, int(round(hold_f / commit))))
                       if t0 + k * commit < search_end]
        if not step_frames:
            step_frames = [int(t0)]
        # rows of the last commit block of the hold: where the engine measures the requested move
        end_m = (centers >= t0 + hold_f - commit) & (centers < t0 + hold_f)
        if not (end_m & free).any():
            end_m = np.zeros(len(rows), dtype=bool)
            end_m[-1] = True
        end_idx = np.where(end_m & free)[0] if (end_m & free).any() else np.where(end_m)[0]
        in_hold = np.where(centers < t0 + hold_f)[0]
        if len(in_hold) == 0:
            in_hold = np.arange(len(rows))
        spread = in_hold[np.linspace(0, len(in_hold) - 1, min(self.hold_search_rows, len(in_hold))).astype(int)]
        pre = np.array([end_idx[0] - 1]) if end_idx[0] > 0 else np.zeros(0, dtype=np.int64)
        sub_idx = np.unique(np.concatenate([spread, pre, end_idx]).astype(np.int64))
        sub_rows = rows[sub_idx]
        end_in_sub = np.searchsorted(sub_idx, end_idx)
        free_sub = unit.free_mask[sub_rows]
        n_eval = 0
        pre_row = int(rows[0]) - 1

        def ev(steps, rws):
            """`plan_rows` with one extra leading row, so that the spectral flux of the first row of
            interest is computed from its true predecessor instead of being 0 (the flux of a row is
            the positive change of the band ratios against the previous row of the evaluated
            sequence).  Without it every hold would start with an artificial flux step."""
            rws = np.asarray(rws, dtype=np.int64)
            if pre_row < 0:
                return plan_eval(steps, rws)
            xi_e, parts_e, info_e = plan_eval(steps, np.concatenate([[pre_row], rws]))
            parts_e = {k: (v[1:] if isinstance(v, np.ndarray) and len(v) == len(rws) + 1 else v)
                       for k, v in parts_e.items()}
            return xi_e[1:], parts_e, info_e

        # ---- the unchanged continuation: the material's own flutter over this hold sets the band
        xi_none, _p_none, _i_none = ev([], rows)
        n_eval += 1
        blk = ((centers - t0) // commit).astype(np.int64)
        fl = [float(an.dist2(xi_none[m], xi_none[m].mean(axis=0)[None, :]).mean())
              for b_ in np.unique(blk[in_hold]) for m in [free & (blk == b_)] if int(m.sum()) >= 3]
        flutter_local = float(np.mean(fl)) if fl else self.hold_flutter_floor
        flutter_local = max(flutter_local, self.hold_flutter_floor)
        self._flutter_ema = (flutter_local if self._flutter_ema is None
                             else 0.7 * self._flutter_ema + 0.3 * flutter_local)
        # The band is set on the sound's OWN fast motion, exactly the quantity the engine divides
        # by: the mean d_xi² of a committed row to its commit-block mean, as an EMA over the
        # committed blocks so far.  (Measured: that motion sits almost entirely in the band-ratio
        # block of xi and is a property of the material at the played positions - level ramps
        # contribute ~0 and a jump ~25% - so it does not run away with the size of the request.)
        # Before the first committed block the unchanged continuation of this hold stands in.
        flutter = max(float(self._realized_flutter if self._realized_flutter is not None
                            else flutter_local), self.hold_flutter_floor)
        o0 = float(unit.o[rows[0]])
        s2 = (self.hold_move_scale * o0) ** 2
        lo, hi = self.hold_band_lo * flutter * s2, self.hold_band_hi * flutter * s2
        d_none = float(an.dist2(xi_none[end_idx].mean(axis=0)[None, :], anchor[None, :])[0])

        # ---- B candidate plans, screened on a row subset
        avail = [0] + ([1, 2, 3] if jumps_on else [])
        if cap > 0 and len(self.hold_slots) > 4 and silent and sounding:
            avail.append(4)                       # swap: the only way a silent material enters
        if jumps_on and getattr(an, "frag_f", None) is None:
            avail = [0, 1, 3]
        tot = sum(self.hold_slot_share[s] for s in avail)
        counts = {s: max(1, int(round(self.hold_candidates * self.hold_slot_share[s] / tot))) for s in avail}
        plan_slots: List[int] = []
        for s in avail:
            plan_slots.extend([s] * counts[s])
        plan_slots = plan_slots[: max(len(avail), self.hold_candidates)]
        while len(plan_slots) < self.hold_candidates:
            plan_slots.append(avail[len(plan_slots) % len(avail)])
        pools = self._hold_position_pools(unit, history, anchor, positions_now)
        m0, m1 = self.hold_magnitude_range
        cands: List[Dict[str, Any]] = []
        for slot in plan_slots:
            mag = float(np.exp(rng.uniform(np.log(max(m0, 1e-6)), np.log(max(m1, m0 + 1e-6)))))
            pl = self._hold_draw_plan(unit, slot, mag, step_frames, levels_now, next_jump,
                                      goal_at, pools, jumps_on, o0,
                                      cap=cap, voices=voices, sounding=sounding, silent=silent,
                                      project=project, positions_now=positions_now, anchor=anchor)
            xi_s, _ps, info_s = ev(pl["steps"], sub_rows)
            n_eval += 1
            pl["xi_sub"] = xi_s
            pl["d_anchor"] = float(an.dist2(xi_s[end_in_sub].mean(axis=0)[None, :], anchor[None, :])[0])
            # the plan's OWN fast motion inside the last commit block of the hold.  The realizer
            # follows a realizable ideal almost exactly, so this is what the engine will measure as
            # `realized_flutter_dist2`: a plan whose move is not clearly above its own flicker asks
            # for a gesture that cannot be heard as a gesture (R4).
            xe = xi_s[end_in_sub]
            pl["flutter"] = (float(an.dist2(xe, xe.mean(axis=0)[None, :]).mean()) if len(xe) >= 3
                             else float(flutter))
            pl["move_over_flutter"] = pl["d_anchor"] / max(pl["flutter"], self.hold_flutter_floor)
            pl["dropped_jumps"] = len(info_s["dropped_jumps"])
            cands.append(pl)

        def spread_of(idx: Sequence[int]) -> float:
            if len(idx) < 2:
                return 0.0
            xs = [cands[a]["xi_sub"][free_sub] if free_sub.any() else cands[a]["xi_sub"] for a in idx]
            return float(np.mean([float(an.dist2(xs[a], xs[b_]).mean())
                                  for a in range(len(xs)) for b_ in range(a + 1, len(xs))]))

        # ---- NARROWING: keep the plans inside the band around the committed history (the anchor)
        d_all = np.array([c["d_anchor"] for c in cands])
        rat_all = np.array([c["move_over_flutter"] for c in cands])
        in_band = np.where((rat_all >= self.hold_ratio_min) & (rat_all <= self.hold_ratio_max)
                           & (d_all <= hi))[0]
        keep = list(in_band)
        widened = ""
        if len(keep) < self.hold_keep_min:
            pen = (np.maximum(self.hold_ratio_min - rat_all, 0.0) / max(self.hold_ratio_min, 1e-9)
                   + np.maximum(rat_all - self.hold_ratio_max, 0.0) / max(self.hold_ratio_max, 1e-9)
                   + np.maximum(d_all - hi, 0.0) / max(hi, 1e-9))
            keep = [int(k) for k in np.argsort(pen)[: self.hold_keep_min]]
            widened = (f"fewer than {self.hold_keep_min} plans with move/flutter in "
                       f"[{self.hold_ratio_min}, {self.hold_ratio_max}] and d2 <= {hi:.3f}: "
                       f"the closest were kept")
        if len(keep) > self.hold_keep_max:
            ctr = 0.5 * (self.hold_ratio_min + self.hold_ratio_max)
            keep = [int(k) for k in np.array(keep)[np.argsort(np.abs(rat_all[np.array(keep)] - ctr))][: self.hold_keep_max]]
        spread_before, spread_after = spread_of(range(len(cands))), spread_of(keep)

        # ---- exact rows of the kept plans + eq. (40) weights among them
        kept_xi, j_ref, e_form_k, e_hist_k = [], [], [], []
        base_full = np.repeat(anchor[None, :], unit.J, axis=0)
        for k in keep:
            xi_k, parts_k, info_k = ev(cands[k]["steps"], rows)
            n_eval += 1
            kept_xi.append(xi_k)
            ef = float(self.objective.e_form_rows(unit, xi_k, parts_k, rows))
            xf = base_full.copy()
            xf[rows] = xi_k
            eh = float(self.objective.e_hist(unit, fix_hold_rows(unit, xf), history))
            e_form_k.append(ef)
            e_hist_k.append(eh)
            j_ref.append(ef + 0.2 * eh)
            cands[k]["d_anchor_rows"] = float(an.dist2(xi_k[end_idx].mean(axis=0)[None, :], anchor[None, :])[0])
        j_ref = np.asarray(j_ref, dtype=np.float64)
        z = -(j_ref - j_ref.min()) / self.tau_ref
        r_b = np.exp(z - z.max())
        r_b = r_b / r_b.sum()
        kept_slots = [int(cands[k]["slot"]) for k in keep]
        kept_arr = np.stack(kept_xi)

        # ---- generator p_Theta(b, w): slot categorical x eq. (40) inside the slot, level-space w
        a_full = self._hold_alphas()
        w_in = np.zeros(len(keep))
        for s in set(kept_slots):
            m = np.array([x == s for x in kept_slots])
            w_in[m] = r_b[m] / max(1e-300, float(r_b[m].sum()))
        p_sel = np.array([a_full[s] for s in kept_slots]) * w_in
        p_sel = p_sel / max(1e-300, float(p_sel.sum()))
        out: List[Target] = []
        draws: List[Dict[str, Any]] = []
        pub_xi: List[np.ndarray] = []
        for a in range(max(1, int(n_proposals))):
            kb = int(rng.choice(len(keep), p=p_sel))
            slot = kept_slots[kb]
            w = self.h_mu[slot] + np.exp(self.h_log_sigma[slot]) * rng.standard_normal(max(1, self.M - 1))
            steps_gen = []
            for s_ in cands[keep[kb]]["steps"]:
                lv = np.asarray(s_["levels"], dtype=np.float64).copy()
                # the displacement acts on the SOUNDING tracks of that step only (a level change on
                # a silent material would either do nothing or evict a sounding one), then the plan
                # is projected back onto the cap so that what is published is what can be played
                mv = np.clip(lv[1:] + w, 0.0, 1.0)
                lv[1:] = np.where(lv[1:] > 1e-6, mv, lv[1:]) if cap > 0 else mv
                if project is not None:
                    lv = np.asarray(project(lv), dtype=np.float64)
                steps_gen.append({"frame": int(s_["frame"]),
                                  "jumps": {int(i): int(p) for i, p in s_["jumps"].items()},
                                  "levels": [float(x) for x in lv]})
            xi_g, _pg, info_g = ev(steps_gen, rows)
            n_eval += 1
            # publish exactly the levels the evaluator used (docs/FIDELITY_CONTRACT.md, cap item 1)
            for k_, lu in enumerate(info_g.get("levels_used") or []):
                if lu is not None and k_ < len(steps_gen):
                    steps_gen[k_]["levels"] = [float(x) for x in lu]
            pub_xi.append(xi_g)
            xi_hat = base_full.copy()
            xi_hat[rows] = xi_g
            xi_hat = fix_hold_rows(unit, xi_hat)
            d_pub = float(an.dist2(xi_g[end_idx].mean(axis=0)[None, :], anchor[None, :])[0])
            logp = self._hold_log_p(slot, w)
            out.append(Target(
                f"gan:u{unit.index}:hold{hold_id}:{a}", xi_hat,
                meta={"hold": True, "b": int(kb), "slot": int(slot), "slot_name": self.hold_slots[slot],
                      "w": [float(x) for x in w], "logp_draw": float(logp),
                      "alpha": float(a_full[slot]), "select_p": float(p_sel[kb]),
                      "parent_id": f"gan:u{unit.index}:h{hold_id}:b{int(keep[kb])}",
                      "parent_origin": self.hold_slots[slot], "basis_hash": self.basis_hash,
                      "swap": (cands[keep[kb]].get("swap") is not None),
                      "swap_detail": cands[keep[kb]].get("swap"),
                      "active_materials_start": int(len(sounding)), "cap": int(cap),
                      "hold_id": hold_id, "plan": steps_gen,
                      "continuity_seconds": 0.0, "continuity_delta_norm": 0.0,
                      "requested_dist2": d_pub, "dropped_jumps": len(info_g["dropped_jumps"]),
                      "rows": [int(rows[0]), int(rows[-1])]}))
            draws.append({"proposal": int(a), "kept_index": int(kb), "slot": self.hold_slots[slot],
                          "w": [round(float(x), 4) for x in w], "requested_dist2": d_pub,
                          "requested_over_local_flutter": d_pub / max(1e-12, flutter),
                          "plan_jumps": {str(s["frame"]): s["jumps"] for s in steps_gen if s["jumps"]},
                          "dropped_jumps": int(len(info_g["dropped_jumps"]))})

        # ---- POSITIVES of D: unmanipulated playback on this hold's psi window (FIDELITY_CONTRACT C)
        psi_pos: Optional[np.ndarray] = None
        real_stat: Dict[str, Any] = {}
        n_psi = min(len(rows), self.hold_psi_blocks * max(1, self.block_rows))
        psi_rows = rows[:n_psi]
        if self.hold_positive_source == "recordings" and getattr(self, "sources", None) is not None:
            n_draw = self.hold_real_positives
            if self.h_psi_scale_source == "uninitialised" or self.h_psi_provisional:
                n_draw = max(n_draw, self.hold_psi_scale_samples)
            real = self._hold_real_windows(unit, rs, psi_rows, n_draw, rng, anchor)
            if self.h_psi_scale_source == "uninitialised" or self.h_psi_provisional:
                self._hold_psi_scales(history, real, psi_rows)
            pos_set = real[: self.hold_real_positives]
            psi_pos = np.stack([self._hold_psi(x, psi_rows) for x in pos_set])
            d_real = _sigmoid(psi_pos @ self.h_phi)
            # D of the candidate plans BEFORE realization, on the same window and feature map
            d_plan = [float(_sigmoid(float(self._hold_psi(kept_xi[j][:n_psi], psi_rows) @ self.h_phi)))
                      for j in range(len(keep))]
            d_none_D = float(_sigmoid(float(self._hold_psi(xi_none[:n_psi], psi_rows) @ self.h_phi)))
            d_med = float(np.median(d_plan)) if d_plan else None
            real_stat = {
                "positive_source": "recordings", "real_windows": int(len(pos_set)),
                "real_windows_drawn": int(len(real)), "psi_rows": int(n_psi),
                "psi_blocks": int(max(1, n_psi // max(1, self.block_rows))),
                "D_real_windows_mean": float(d_real.mean()), "D_real_windows_min": float(d_real.min()),
                "D_real_windows_max": float(d_real.max()),
                "D_candidate_plans": [round(float(x), 4) for x in d_plan],
                "D_candidate_plans_mean": (float(np.mean(d_plan)) if d_plan else None),
                "D_unchanged_continuation": d_none_D,
                "D_candidate_plans_median": d_med,
                "D_published_plan": None, "chosen_above_median": None,
                "position_sources": dict(getattr(self, "_real_position_sources", {})),
                "near_share": float(self.hold_real_near_share),
                "levels_source": ("committed level vectors" if self._committed_levels else "current levels"),
                "psi_scale_source": self.h_psi_scale_source}
            # ---- generator reward: the CLIPPED LOGIT of D on the exact PLAN rows of the published
            # candidate.  -log D(committed) saturates (D -> 0, -log D ~ 13) and then carries no
            # preference between plans; the logit is linear in psi and bounded here on purpose.
            for a_, tg in enumerate(out):
                lg = float(np.clip(float(self._hold_psi(pub_xi[a_][:n_psi], psi_rows) @ self.h_phi),
                                   -self.hold_logit_clip, self.hold_logit_clip))
                tg.meta["logit_plan"] = lg
                tg.meta["D_plan"] = float(_sigmoid(lg))
                draws[a_]["D_plan"] = float(_sigmoid(lg))
                draws[a_]["logit_plan"] = lg
            if out:
                d_pub0 = float(out[0].meta["D_plan"])
                real_stat["D_published_plan"] = d_pub0
                real_stat["chosen_above_median"] = (bool(d_pub0 >= d_med - 1e-9) if d_med is not None else None)
                # the published plan is parent + displacement w; the parent alone says whether the
                # SELECTION (slot categorical x eq. 40) prefers the plans D finds more real
                kb0 = int(out[0].meta.get("b", 0))
                real_stat["D_chosen_parent"] = (float(d_plan[kb0]) if kb0 < len(d_plan) else None)
                real_stat["parent_above_median"] = (bool(d_plan[kb0] >= d_med - 1e-9)
                                                    if (d_med is not None and kb0 < len(d_plan)) else None)
        rec = {
            "real_positives": real_stat,
            "unit": int(unit.index), "hold": hold_id, "t0_seconds": t0 / float(self.fs),
            "openness": o0, "phase": str(unit.phase_names[rows[0]]), "steps": len(step_frames),
            "rows": [int(rows[0]), int(rows[-1])],
            "flutter_local_dist2": flutter_local,
            "flutter_note": "commit blocks of the UNCHANGED continuation of this hold",
            "flutter_realized_ema_dist2": float(self._realized_flutter or flutter_local),
            "band_base_dist2": float(flutter), "band_dist2": [float(lo), float(hi)],
            "band_rule": f"[{self.hold_band_lo}, {self.hold_band_hi}] x the realized flutter "
                         f"(EMA over the committed blocks) x (hold_move_scale "
                         f"{self.hold_move_scale} x openness {o0:.2f})^2",
            "do_nothing_dist2": d_none,
            "cap": {"max_active_materials": int(cap),
                    "active_materials_at_hold_start": int(len(sounding)),
                    "at_cap": bool(cap > 0 and len(sounding) >= cap),
                    "silent_materials": int(len(silent)),
                    "published_swap": (out[0].meta.get("swap_detail") if out else None),
                    "swap_candidates_drawn": int(sum(1 for c in cands if c.get("swap") is not None)),
                    "swap_candidates_kept": int(sum(1 for k in keep if cands[k].get("swap") is not None))}
            if cap > 0 else None,
            "candidate_plans_drawn": len(cands), "candidate_plans_in_band": int(len(in_band)),
            "candidate_plans_kept": len(keep),
            "kept_fraction": float(len(in_band)) / float(max(1, len(cands))),
            "band_widened": widened,
            "reference_spread_before_dist2": spread_before, "reference_spread_after_dist2": spread_after,
            "d_anchor_all": [round(float(x), 4) for x in d_all],
            "move_over_flutter_all": [round(float(x), 2) for x in rat_all],
            "move_over_flutter_kept": [round(float(rat_all[k]), 2) for k in keep],
            "plan_flutter_kept": [round(float(cands[k]["flutter"]), 4) for k in keep],
            "move_over_flutter_min": float(self.hold_ratio_min),
            "slot_all": [self.hold_slots[c["slot"]] for c in cands],
            "beta_all": [round(float(c["beta"]), 3) for c in cands],
            "jumps_all": [int(c["n_jumps"]) for c in cands],
            "d_anchor_kept": [round(float(cands[k].get("d_anchor_rows", cands[k]["d_anchor"])), 4) for k in keep],
            "slots_drawn": {self.hold_slots[s]: int(sum(1 for c in cands if c["slot"] == s)) for s in avail},
            "slots_kept": {self.hold_slots[s]: int(sum(1 for x in kept_slots if x == s)) for s in avail},
            "J_ref": [round(float(x), 5) for x in j_ref], "weights_r_b": [round(float(x), 4) for x in r_b],
            "e_form_kept": [round(float(x), 5) for x in e_form_k],
            "e_hist_kept": [round(float(x), 5) for x in e_hist_k],
            "slot_alphas": {n: round(float(v), 4) for n, v in zip(self.hold_slots, a_full)},
            "draws": draws, "position_pool": {"band_profile": pools["band_profile_source"],
                                              "committed_positions_per_track": pools["n_committed"]},
            "plan_rows_calls": int(n_eval),
            "internal_iterations_note": "candidate plans and plan_rows calls are internal iterations, "
                                        "not musical time",
        }
        self._hold_records.append(rec)
        self._hold = {"id": hold_id, "rows": rows, "kept_xi": kept_arr, "r": r_b,
                      "slots": kept_slots, "record": rec, "psi_pos": psi_pos,
                      "psi_rows": psi_rows}
        return out

    def _hold_g_steps(self, baseline: float, n_steps: int) -> Dict[str, Any]:
        """REINFORCE (eq. 45) on the hold generator: slot logits and the level-space (mu, sigma),
        with the same self-normalised importance correction and ESS trust region as the legacy
        generator.  D, the kept plans and the committed rows are fixed here."""
        batch = [q for q in self._g_batch() if q.get("hold")]
        if not batch:
            return {"updates_G": 0, "clipped_G_steps": 0, "batch_size": 0, "importance_ess": [],
                    "ess_stopped": 0, "stop_reason": "no_hold_samples",
                    "importance_correction": "self-normalised p_theta/p_draw; stop when ESS < fraction * n"}
        adv = np.array([q["ell"] - float(baseline) for q in batch], dtype=np.float64)
        logp_draw = np.array([q["logp_draw"] for q in batch], dtype=np.float64)
        n_s = float(max(1, len(batch)))
        ess_min = self.ess_min_fraction * n_s
        nG = clipped = ess_stopped = 0
        ess_trace: List[float] = []
        stop = ""
        for _ in range(int(n_steps)):
            logp_new = np.array([self._hold_log_p(int(q["b"]), q["w"]) for q in batch], dtype=np.float64)
            ratio = np.exp(np.clip(logp_new - logp_draw, -30.0, 30.0))
            ess = float(ratio.sum() ** 2 / max(1e-300, float((ratio ** 2).sum())))
            ess_trace.append(ess)
            if len(batch) > 1 and ess < ess_min:
                ess_stopped += 1
                stop = "ess_below_trust_region"
                break
            omega = ratio / max(1e-300, float(ratio.sum())) * n_s
            g_mu = np.zeros_like(self.h_mu)
            g_ls = np.zeros_like(self.h_log_sigma)
            g_lg = np.zeros_like(self.h_logits)
            a = self._hold_alphas()
            sig = np.exp(self.h_log_sigma)
            for q, adv_i, om in zip(batch, adv, omega):
                s = int(q["b"])
                w = np.asarray(q["w"], dtype=np.float64)
                g_mu[s] += om * adv_i * (w - self.h_mu[s]) / np.maximum(sig[s] ** 2, 1e-12)
                g_ls[s] += om * adv_i * (((w - self.h_mu[s]) ** 2) / np.maximum(sig[s] ** 2, 1e-12) - 1.0)
                oh = np.zeros(len(self.h_logits))
                oh[s] = 1.0
                g_lg += om * adv_i * (oh - a)
            g_mu, g_ls, g_lg = g_mu / n_s, g_ls / n_s, g_lg / n_s
            gn = float(np.sqrt((g_mu ** 2).sum() + (g_ls ** 2).sum() + (g_lg ** 2).sum()))
            if not np.isfinite(gn):
                stop = "non_finite_gradient"
                break
            if gn <= 1e-15:
                stop = "zero_gradient (advantage 0)"
                break
            if gn > self.grad_clip:
                f = self.grad_clip / gn
                g_mu, g_ls, g_lg = g_mu * f, g_ls * f, g_lg * f
                clipped += 1
            lr = self.lr_G * self.hold_lr_G_scale
            s_mu = np.clip(-lr * g_mu, -self.step_clip, self.step_clip)
            s_ls = np.clip(-lr * g_ls, -self.step_clip, self.step_clip)
            s_lg = np.clip(-lr * g_lg, -self.step_clip, self.step_clip)
            if not (np.all(np.isfinite(s_mu)) and np.all(np.isfinite(s_ls)) and np.all(np.isfinite(s_lg))):
                stop = "non_finite_step"
                break
            self.h_mu = self.h_mu + s_mu
            self.h_log_sigma = np.clip(self.h_log_sigma + s_ls, np.log(self.hold_w_sigma_min),
                                       np.log(self.hold_w_sigma_max))
            self.h_logits = self.h_logits + s_lg
            nG += 1
        return {"updates_G": int(nG), "clipped_G_steps": int(clipped), "batch_size": int(len(batch)),
                "importance_ess": [float(x) for x in ess_trace], "ess_stopped": int(ess_stopped),
                "stop_reason": stop,
                "importance_correction": "self-normalised p_theta/p_draw on re-used committed "
                                         "samples; stop when ESS < fraction * n"}

    # ================================================================== conditioning
    def begin_unit(self, unit: UnitContext, history) -> None:
        self.unit = unit
        fm = unit.free_mask
        if fm.any():
            self.fm_idx = np.where(fm)[0]
        else:
            self.fm_idx = np.arange(unit.J)
            self.warnings.append(f"unit {unit.index}: no free windows; psi/basis use all windows")
        self.o = unit.o
        self.unit_blocks = self._unit_blocks(unit)

        # ---------------------------------------------------------- (40) reference distribution
        refs = self._draw_references(unit, history)
        self.reference_source = ("random_fragment_composition_trajectories" if self.fragment_mode()
                                 else "bank_random_legal_gain_plans")
        self.ref_xi = np.stack([r["xi"] for r in refs])                   # (B, J, d)
        j_ref = np.array([float(r["e_form"]) + 0.2 * float(r["e_hist"]) for r in refs])
        z = -(j_ref - j_ref.min()) / self.tau_ref
        w = np.exp(z - z.max())
        self.ref_w = w / w.sum()
        self.ref_J = j_ref
        order = np.argsort(-self.ref_w)
        self.ref_order = order
        div = 0.0
        if len(refs) > 1:
            xs = self.ref_xi[:, self.fm_idx, :]
            ds = [float(self.analyzer.dist2(xs[a], xs[b]).mean())
                  for a in range(len(xs)) for b in range(a + 1, len(xs))]
            div = float(np.mean(ds))
        if div < 1e-6 and len(refs) > 1:
            self.warnings.append(f"unit {unit.index}: reference distribution has very low acoustic "
                                 f"diversity (mean pair dist2 {div:.2e}); proceeding")
        ent = float(-(self.ref_w * np.log(np.clip(self.ref_w, 1e-12, 1.0))).sum())
        summary = {
            "unit": unit.index, "n_references": len(refs), "weights": self.ref_w.tolist(),
            "J_ref": j_ref.tolist(), "J_ref_definition": "E_form + 0.2 * E_hist (no discriminator)",
            "temperature": self.tau_ref, "weight_entropy_nats": ent,
            "diversity_mean_pair_dist2": div, "source": self.reference_source,
            "compositions_evaluated": int(self.extra_candidate_evaluations),
            "e_form": [float(r["e_form"]) for r in refs], "e_hist": [float(r["e_hist"]) for r in refs]}
        summary.update(self._reference_motion(unit))
        if self.fragment_mode():
            summary["fragment_sampling"] = {
                "rule": ("one random fragment composition (positions + material levels) per "
                         f"{self.frag_segment_seconds} s block; "
                         + ("every row of a block is the exact mixture at hop-spaced offsets after "
                            "the sampled positions (no interpolation)" if self.frag_offset_rows <= 0
                            else f"the first {self.frag_offset_rows} rows of a block are the exact "
                                 "mixture at hop-spaced offsets after the sampled positions, the "
                                 "rest are linearly interpolated toward the next block's first row")),
                "helper": "Analyzer.random_fragment_composition(sources, rng, offsets, goal_level)",
                "rng": "the job rng (config.seed); the draws below are the record of this unit",
                "segment_seconds": self.frag_segment_seconds,
                "segment_rows": int(round(self.frag_segment_seconds / max(1e-9, self.hop_seconds))),
                "offsets": f"arange(k) * {self.hop_frames} frames (hop {self.hop_seconds:.3f} s)",
                "sampled_rows_per_reference": int(refs[0]["sampled_rows"]),
                "interpolated_rows_per_reference": int(refs[0]["interpolated_rows"]),
                "goal_level_policy": ("form.goal_exposure: REOPEN descent from 1 to "
                                      f"{self.goal_open_cap}, Q5 rise to 1 over the last "
                                      f"{self.goal_rise_seconds} s before the goal arrival, 1 in "
                                      "GOAL_HOLD (held constant inside a block)"),
                "goal_levels_per_block": [round(float(self._frag_goal_levels[a]), 4)
                                          for (a, _b) in self._frag_bounds],
                "hold_rows": "pinned to xi_goal (fix_hold_rows)",
                "references": [{"index": int(r["index"]), "J_ref": float(j_ref[k]),
                                "weight": float(self.ref_w[k]), "blocks": r["segments"]}
                               for k, r in enumerate(refs)]}
        self.reference_summaries.append(summary)

        # ---------------------------------------------------------- (41) joint basis B_ref
        self._build_basis(unit)
        # ---------------------------------------------------------- parents + lineage
        self._build_parents(unit, history)
        # ---------------------------------------------------------- psi scales (frozen) + Theta
        self._psi_scales(unit, history)
        self._init_theta(history)
        self._g_buffer = []
        self._pending = []
        self._commit_records = []
        self._unit_stats = []
        self._step_index = 0
        self._unit_counts = {"D": 0, "G": 0, "D_rejected": 0, "update_units": 0, "skipped_blocks": 0}
        # hold mode is only known at the first prepare_reference of the unit (HOLD_CONTRACT); these
        # assignments touch neither the rng nor any legacy quantity
        self._hold_active = False
        self._hold = None
        self._hold_records = []
        self._hist_blocks = []        # row indices are unit-local

    def _build_basis(self, unit: UnitContext) -> None:
        """Few joint directions from the reference differences (small SVD, eq. 41).
        The basis acts on the whole composition trajectory at once, never on a single track.
        A deterministic sign rule removes the arbitrary SVD sign; the basis is hashed so that the
        generator coordinates can be re-initialised exactly when the basis changes (audit B7)."""
        X = self.ref_xi[:, self.fm_idx, :]                                # (B, Jf, d)
        mean = np.einsum("b,bjd->jd", self.ref_w, X)
        Dif = (X - mean[None]).reshape(len(X), -1)
        # the rank is a fixed configuration constant (unused directions stay zero) so that the
        # generator parameter shapes - and therefore signature() - do not depend on how many
        # references happened to be legal in this unit
        self.r = max(1, int(self.basis_rank))
        B = np.zeros((self.r, unit.J, self.d_xi))
        sv_all: List[float] = []
        if len(X) >= 2 and np.any(np.abs(Dif) > 1e-12):
            _U, sv, Vt = np.linalg.svd(Dif, full_matrices=False)
            sv_all = sv.tolist()
            scale = 1.0 / np.sqrt(float(len(X)))
            n_dir = 0
            for k in range(self.r):
                if k < len(sv) and sv[k] > 1e-12:
                    B[k][self.fm_idx] = (sv[k] * scale) * Vt[k].reshape(len(self.fm_idx), self.d_xi)
                    n_dir += 1
            self.basis_source = "svd_of_reference_differences"
            if n_dir < self.r:
                self.warnings.append(f"unit {unit.index}: only {n_dir} of {self.r} basis directions "
                                     f"are supported by the reference set; the rest stay zero")
        else:
            # explicit reference differences (spec allows this instead of the SVD)
            for k in range(self.r):
                b = self.ref_order[min(k + 1, len(self.ref_order) - 1)]
                B[k][self.fm_idx] = X[b] - mean
            self.basis_source = "explicit_reference_differences"
            self.warnings.append(f"unit {unit.index}: reference differences degenerate; "
                                 f"basis built from explicit reference differences")
        # deterministic sign: the largest-magnitude component of each direction is positive.  This
        # removes the arbitrary SVD sign but is NOT a coordinate transport (audit B7 option 2).
        for k in range(self.r):
            flat = B[k].ravel()
            j = int(np.argmax(np.abs(flat)))
            if flat[j] < 0:
                B[k] = -B[k]
        self.B_ref = B
        self.basis_sv = sv_all
        self.basis_norms = [float(np.linalg.norm(B[k])) for k in range(self.r)]
        self.basis_sign_rule = "largest-|component| of each direction positive"
        self.basis_hash = hashlib.sha256(np.ascontiguousarray(B).tobytes()).hexdigest()[:16]

    def _build_parents(self, unit: UnitContext, history) -> None:
        """Parents: past non-hold composition lines from the lineage (mapped onto this unit's
        grid through the phase-relative blocks) plus a few fresh reference lines, so re-enabling
        keeps exploring instead of collapsing onto the single goal point (spec §10.6)."""
        parents: List[np.ndarray] = []
        ids: List[str] = []
        origins: List[str] = []
        hp = [p for p in getattr(history, "parents", []) if isinstance(p, dict) and "blocks" in p]
        hp = sorted(hp, key=lambda q: (-int(q.get("unit", -1)), float(q.get("score", 1e9))))
        for q in hp[: self.n_hist_parents]:
            try:
                xi_p = self.objective.blocks_to_unit(unit, q["blocks"])
            except Exception as e:                                        # noqa: BLE001
                self.warnings.append(f"unit {unit.index}: parent {q.get('id')} could not be mapped "
                                     f"onto this unit ({e}); replaced by a fresh reference")
                continue
            parents.append(fix_hold_rows(unit, xi_p))
            ids.append(str(q.get("id")))
            origins.append("history_parent")
        k = 0
        while len(parents) < self.n_parents and k < len(self.ref_order):
            b = int(self.ref_order[k])
            parents.append(fix_hold_rows(unit, self.ref_xi[b]))
            ids.append(f"gan:u{unit.index}:ref{b}")
            origins.append("fresh_reference")
            k += 1
        while len(parents) < self.n_parents:                              # only if refs < parents
            parents.append(fix_hold_rows(unit, self.ref_xi[int(self.ref_order[0])]))
            ids.append(f"gan:u{unit.index}:ref{int(self.ref_order[0])}:dup{len(parents)}")
            origins.append("fresh_reference_duplicate")
        self.parent_xi = np.stack(parents[: self.n_parents])
        self.parent_ids = ids[: self.n_parents]
        self.parent_origins = origins[: self.n_parents]
        self.n_history_parents_used = sum(1 for o in self.parent_origins if o == "history_parent")

    def _psi_scales(self, unit: UnitContext, history) -> None:
        """Discriminator input standardisation: computed once, from the reference set of the first
        unit cut into commit-length blocks (the rows D is actually applied to), then FROZEN for the
        whole generation so that the inherited coefficients phi keep their meaning (audit §8.1)."""
        st = history.mode_state.get("gan") if hasattr(history, "mode_state") else None
        if (st and isinstance(st.get("psi_mean"), list) and len(st["psi_mean"]) == self.d_psi
                and isinstance(st.get("psi_std"), list) and len(st["psi_std"]) == self.d_psi):
            self.psi_mean = np.array(st["psi_mean"], dtype=np.float64)
            self.psi_std = np.maximum(np.array(st["psi_std"], dtype=np.float64), 1e-6)
            self.psi_scale_source = "frozen_from_first_unit"
            self.psi_scale_blocks = int(st.get("psi_scale_blocks", 0))
            return
        rows_sets = self.unit_blocks or [None]
        P = [self._psi_raw(self.ref_xi[b] if blk is None else self.ref_xi[b][blk], blk)
             for b in range(len(self.ref_xi)) for blk in rows_sets]
        Psi = np.stack(P)
        m = Psi.mean(axis=0)
        s = np.maximum(Psi.std(axis=0), 1e-3)
        m[-1] = 0.0
        s[-1] = 1.0                                  # bias stays 1
        self.psi_mean, self.psi_std = m, s
        self.psi_scale_source = "reference_set_commit_blocks_of_this_unit"
        self.psi_scale_blocks = int(len(P))

    def _init_theta(self, history) -> None:
        """Generator parameters at the start of a unit (audit B7 / D4.1).

        Inherited: the discriminator phi (with the frozen psi standardisation), the mixture logits
        by parent id, the parent acoustics / parent ids and the common history.
        Re-initialised: mu = 0 and log_sigma = log(sigma_init) of the displacement coefficients,
        because B_ref was rebuilt and the same numeric w no longer means the same acoustic
        direction.  Coordinate transport (audit B7 option 2) is not implemented."""
        st = history.mode_state.get("gan") if hasattr(history, "mode_state") else None
        self.logits = np.zeros(self.n_parents)
        self.mu = np.zeros((self.n_parents, self.r))
        self.log_sigma = np.full((self.n_parents, self.r), np.log(max(self.sigma_init, 1e-9)))
        self.warm_started: List[str] = []
        prev_hash = str(st.get("basis_hash", "")) if st else ""
        basis_changed = (prev_hash != self.basis_hash)
        if not st:
            self.phi = np.zeros(self.d_psi)
            self.warm_start_source = "cold"
            self.inheritance = {"phi": "cold (zero)", "logits": "cold (zero)",
                                "mu": "cold (zero)", "log_sigma": "cold (log sigma_init)"}
            self.inheritance_detail = {
                "basis_changed": True, "basis_hash_before": None, "basis_hash_after": self.basis_hash,
                "first_unit": True, "parents_matched": [], "sigma_init": self.sigma_init,
                "generator_sample_buffer": "empty (first unit)",
                "transport": "none (audit B7 option 1: re-initialise, do not transport)"}
            return
        phi = np.array(st.get("phi", []), dtype=np.float64)
        if phi.shape == (self.d_psi,) and np.all(np.isfinite(phi)):
            self.phi = phi
            self.warm_start_source = "history_mode_state"
            phi_note = "inherited"
        else:
            self.phi = np.zeros(self.d_psi)
            self.warm_start_source = "history_mode_state_without_phi"
            phi_note = "reinitialised (no usable stored phi)"
        by_id: Dict[str, Dict[str, Any]] = {}
        for e in st.get("generator", []):
            if not isinstance(e, dict):
                continue
            by_id[str(e.get("parent_id"))] = e
            # lineage: a parent record created from component b inherits that component's Theta
            for d in e.get("descendant_ids", []) or []:
                by_id.setdefault(str(d), e)
        matched: List[Dict[str, Any]] = []
        for b, pid in enumerate(self.parent_ids):
            e = by_id.get(str(pid))
            if e is None:
                continue
            lg = e.get("logit")
            if lg is not None and np.isfinite(float(lg)):
                self.logits[b] = float(lg)
            mu_inherited = False
            if not basis_changed:
                mu = np.array(e.get("mu", []), dtype=np.float64)
                ls = np.array(e.get("log_sigma", []), dtype=np.float64)
                if mu.shape == (self.r,) and np.all(np.isfinite(mu)):
                    self.mu[b] = mu
                    mu_inherited = True
                if ls.shape == (self.r,) and np.all(np.isfinite(ls)):
                    self.log_sigma[b] = np.clip(ls, np.log(self.sigma_min), np.log(self.sigma_max))
            matched.append({"component": int(b), "parent_id": str(pid), "source_entry": str(e.get("parent_id")),
                            "logit_inherited": lg is not None, "mu_inherited": bool(mu_inherited),
                            "stored_basis_hash": str(e.get("basis_hash", ""))})
            self.warm_started.append(f"{pid}<-{e.get('parent_id')}")
        self.inheritance = {
            "phi": phi_note,
            "logits": "inherited by parent id",
            "mu": ("reinitialised (basis changed)" if basis_changed else "inherited by parent id (basis unchanged)"),
            "log_sigma": ("reinitialised" if basis_changed else "inherited by parent id (basis unchanged)"),
        }
        self.inheritance_detail = {
            "basis_changed": bool(basis_changed),
            "basis_hash_before": prev_hash or None, "basis_hash_after": self.basis_hash,
            "first_unit": False,
            "parent_acoustics": "inherited (history parents mapped through phase-relative blocks)",
            "parent_ids": "inherited", "psi_standardisation": self.psi_scale_source,
            "common_history": "inherited (history.mode_state['gan'])",
            "parents_matched": matched, "sigma_init": self.sigma_init,
            "reinitialised_value": {"mu": 0.0, "log_sigma": float(np.log(max(self.sigma_init, 1e-9)))},
            "generator_sample_buffer": ("cleared at the unit start: the stored (b, w) samples were "
                                        "drawn under the previous basis" if basis_changed else
                                        "cleared at the unit start (per-unit re-use only)"),
            "baseline_ema": "inherited (loss scale is basis-independent)",
            "transport": "none (audit B7 option 1: re-initialise, do not transport)",
            "reason": "B_ref is rebuilt per unit from that unit's reference set, so a numeric w has "
                      "no fixed acoustic meaning across units (audit B7)",
        }
        self._clip_sigma()

    def _clip_sigma(self) -> None:
        self.log_sigma = np.clip(self.log_sigma, np.log(self.sigma_min), np.log(self.sigma_max))

    def alphas(self) -> np.ndarray:
        z = self.logits - self.logits.max()
        e = np.exp(z)
        return e / e.sum()

    # ================================================================== hints
    def hints(self, unit: UnitContext, history) -> List[tuple]:
        """Joint structure read off the top parent's realized composition: materials whose
        contribution trajectories move together get 'sync', opposed ones 'counter'."""
        if not len(self.parent_ids):
            return []
        b = int(np.argmax(self.alphas()))
        c = self.analyzer.split(self.parent_xi[b][self.fm_idx])[1]        # (Jf, M)
        out: List[Tuple[int, int, str, float]] = []
        sd = c.std(axis=0)
        for i in range(1, self.M):
            for j in range(i + 1, self.M):
                if sd[i] < 1e-9 or sd[j] < 1e-9:
                    continue
                rho = float(np.corrcoef(c[:, i], c[:, j])[0, 1])
                if not np.isfinite(rho) or abs(rho) < self.hint_min_corr:
                    continue
                out.append((i, j, "sync" if rho > 0 else "counter", abs(rho)))
        out.sort(key=lambda t: -t[3])
        self._last_hints = [(a, b2, rel) for (a, b2, rel, _) in out[: self.max_hints]]
        return list(self._last_hints)

    # ================================================================== proposals (eq. 41)
    def _sample_theta(self) -> Tuple[int, np.ndarray]:
        a = self.alphas()
        b = int(self.rng.choice(self.n_parents, p=a))
        w = self.mu[b] + np.exp(self.log_sigma[b]) * self.rng.standard_normal(self.r)
        return b, w

    def ideal(self, unit: UnitContext, b: int, w: np.ndarray, rows: Optional[np.ndarray] = None,
              xi_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Xi_hat = Xi_b^parent + B_ref w, openness-scaled, optionally continued from the realized
        current composition on the window rows, contracted toward the goal, holds pinned."""
        pert = np.einsum("k,kjd->jd", np.asarray(w, dtype=np.float64), self.B_ref)
        xi_hat = self.parent_xi[b] + self.o[:, None] * pert
        self._last_delta_norm = 0.0
        if rows is not None and xi_current is not None and self.continuity_seconds > 0.0:
            # audit D4.2: the window reference starts from what has actually been realized and
            # relaxes back onto the sampled ideal with an explicit time constant (seconds), instead
            # of being rebuilt from the unit start every time.  Free rows only.
            rows = np.asarray(rows)
            fm = unit.free_mask[rows]
            if fm.any():
                j0 = int(rows[fm][0])
                delta = np.asarray(xi_current, dtype=np.float64) - xi_hat[j0]
                dt = np.maximum(unit.seconds[rows] - float(unit.seconds[j0]), 0.0)
                lam = np.exp(-dt / max(1e-6, self.continuity_seconds))
                add = lam[:, None] * delta[None, :]
                add[~fm] = 0.0
                xi_hat = xi_hat.copy()
                xi_hat[rows] = xi_hat[rows] + add
                self._last_delta_norm = float(np.linalg.norm(delta))
        contract = (unit.phase_names == "CONTRACT")
        if contract.any():
            lam = np.zeros(unit.J)
            lam[contract] = (1.0 - self.o[contract]) ** 2
            xi_hat = (1.0 - lam[:, None]) * xi_hat + lam[:, None] * unit.xi_goal
        return fix_hold_rows(unit, xi_hat)

    def _target(self, unit: UnitContext, tid: str, b: int, w: np.ndarray, rows=None, xi_current=None) -> Target:
        xi_hat = self.ideal(unit, b, w, rows=rows, xi_current=xi_current)
        return Target(tid, xi_hat, meta={
            "b": int(b), "w": [float(x) for x in w], "parent_id": self.parent_ids[b],
            "parent_origin": self.parent_origins[b], "alpha": float(self.alphas()[b]),
            "logp_draw": float(self._log_p(b, w)), "basis_hash": self.basis_hash,
            "continuity_seconds": (self.continuity_seconds if rows is not None else 0.0),
            "continuity_delta_norm": float(self._last_delta_norm),
            "rows": ([int(rows[0]), int(rows[-1])] if rows is not None and len(rows) else None)})

    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        """Full-unit ideal trajectories (warm start: the engine freezes one of them as R0)."""
        n = max(1, min(int(n_targets), self.samples_per_round))
        targets: List[Target] = []
        for a in range(n):
            b, w = self._sample_theta()
            targets.append(self._target(unit, f"gan:u{unit.index}:full:{a}", b, w))
        return targets

    def prepare_reference(self, unit: UnitContext, history, rows: np.ndarray, xi_current: np.ndarray,
                          n_proposals: int) -> List[Target]:
        """Window references: independent samples (b, w) from p_Theta, each a full (J, d) ideal
        whose window rows continue the realized current composition.  The engine freezes one of
        them; the sample that produced it stays in Target.meta and is never altered afterwards.

        Hold mode (`unit.realizer_state` published by the engine, docs/HOLD_CONTRACT.md): the ideal
        must be realizable, so the whole reference distribution is rebuilt as exact PLANS narrowed
        to the committed history and the displacement acts in level space (`_hold_prepare`).  When
        `realizer_state` is missing / None nothing below changes (legacy, bit-identical)."""
        rs = getattr(unit, "realizer_state", None)
        if rs is not None:
            return self._hold_prepare(unit, history, rows, xi_current, n_proposals, rs)
        rows = np.asarray(rows)
        out: List[Target] = []
        for a in range(max(1, int(n_proposals))):
            b, w = self._sample_theta()
            out.append(self._target(unit, f"gan:u{unit.index}:s{self._step_index}:{a}", b, w,
                                    rows=rows, xi_current=xi_current))
        self._step_index += 1
        return out

    # ================================================================== mode error
    def mode_error(self, unit: UnitContext, cand: Candidate, target: Target) -> float:
        """Fit of the *realized* composition to the ideal one (free windows).  The discriminator
        deliberately does not enter the search score; it enters the generator update (eq. 44)."""
        return unit.mean_dist2(cand.xi, target.xi_hat)

    # ================================================================== adversarial update unit
    def observe_committed(self, unit: UnitContext, history, rows: np.ndarray, xi_rows: np.ndarray,
                          parts_rows: Dict[str, np.ndarray], reference: Optional[Target],
                          stats: Dict[str, Any]) -> Dict[str, Any]:
        """One bounded adversarial update unit per committed block (audit D4.3).

        The reference set, the frozen window reference and the committed realized composition are
        all fixed here; D and G alternate inside this call only.  Nothing in this method touches
        the realization or any reference that is still being scored."""
        rows = np.asarray(rows)
        ref_meta = dict(reference.meta or {}) if reference is not None else {}
        fin = (stats or {}).get("refinement", {}).get("final", {}) or {}
        if self._hold_active:
            # the positions the realizer actually committed are part of the committed history and
            # become jump candidates of the next holds ("close to the committed history")
            for i, p in enumerate((stats or {}).get("positions", []) or []):
                if 0 < i < self.M:
                    self._committed_positions[i].append(int(p))
                    self._committed_positions[i] = self._committed_positions[i][-24:]
            lv_c = (stats or {}).get("levels")
            if lv_c is not None and len(lv_c) == self.M:
                # the levels a real window is drawn with come from here (FIDELITY_CONTRACT C)
                self._committed_levels.append(np.asarray(lv_c, dtype=np.float64))
                self._committed_levels = self._committed_levels[-64:]
            # the sound's own fast motion, exactly as the engine measures it in hold_summary:
            # mean d_xi² of a committed row to the mean of its commit block
            fmb = unit.free_mask[rows]
            if int(fmb.sum()) >= 3:
                xb = np.asarray(xi_rows, dtype=np.float64)[fmb]
                f_now = float(self.analyzer.dist2(xb, xb.mean(axis=0)[None, :]).mean())
                self._realized_flutter = (f_now if self._realized_flutter is None
                                          else 0.75 * self._realized_flutter + 0.25 * f_now)
        self._pending.append({
            "rows": rows, "xi": np.asarray(xi_rows, dtype=np.float64), "meta": ref_meta,
            "fit": float(fin.get("fit", 0.0)), "e_form": float(fin.get("e_form", 0.0)),
            "ref_id": (reference.id if reference is not None else None),
            "ref_hash": (stats or {}).get("reference_hash")})
        all_rows = np.concatenate([p["rows"] for p in self._pending])
        n_free = int(unit.free_mask[all_rows].sum())
        if n_free < self.min_block_free_rows:
            self._unit_counts["skipped_blocks"] += 1
            rec = {"unit": int(unit.index), "rows": int(len(all_rows)), "free_rows": n_free,
                   "accumulated_blocks": len(self._pending), "updates_D": 0, "updates_G": 0,
                   "skipped": f"committed block too short for psi ({n_free} free rows < "
                              f"{self.min_block_free_rows}); accumulating blocks"}
            self._commit_records.append(rec)
            return dict(rec)

        xi_all = np.concatenate([p["xi"] for p in self._pending], axis=0)
        blocks = len(self._pending)
        wsum = float(sum(len(p["rows"]) for p in self._pending))
        fit = float(sum(p["fit"] * len(p["rows"]) for p in self._pending) / max(1e-9, wsum))
        e_form = float(sum(p["e_form"] * len(p["rows"]) for p in self._pending) / max(1e-9, wsum))
        last = self._pending[-1]
        self._pending = []

        # ------------------------------------------------ (43) discriminator, Theta / rows fixed
        hold = self._hold if self._hold_active else None
        d_rows_dropped = 0
        d_which = "phi"
        if hold is not None and hold.get("psi_pos") is not None:
            # ---- real data: positives = unmanipulated playback of this hold, negative = the
            # committed sound over the last `hold_psi_blocks` commit blocks (motion needs >= 2).
            # Every committed block enters exactly one update as the newest block of that window.
            self._hist_blocks.append((rows, np.asarray(xi_rows, dtype=np.float64)))
            self._hist_blocks = self._hist_blocks[-self.hold_psi_blocks:]
            neg_rows = np.concatenate([b[0] for b in self._hist_blocks])
            neg_xi = np.concatenate([b[1] for b in self._hist_blocks], axis=0)
            Psi_ref = hold["psi_pos"]
            w_pos = np.full(len(Psi_ref), 1.0 / max(1, len(Psi_ref)))
            Psi_gen = self._hold_psi(neg_xi, neg_rows)[None, :]
            d_which = "h_phi"
        elif hold is not None:
            # positives = the EXACT rows of the kept reference plans of this hold, weights r_b;
            # negative = the committed composition on the same rows
            pos = np.searchsorted(hold["rows"], all_rows)
            ok = (pos < len(hold["rows"])) & (hold["rows"][np.clip(pos, 0, len(hold["rows"]) - 1)] == all_rows)
            d_rows_dropped = int((~ok).sum())
            if d_rows_dropped:
                self.warnings.append(f"unit {unit.index}: {d_rows_dropped} committed row(s) lie outside "
                                     f"the held reference (accumulated across holds); D used the rest")
            sel = pos[ok] if ok.any() else pos[:0]
            d_rows = all_rows[ok] if ok.any() else all_rows
            xi_d = xi_all[ok] if ok.any() else xi_all
            if ok.any():
                Psi_ref = np.stack([self._psi(hold["kept_xi"][b][sel], d_rows) for b in range(len(hold["kept_xi"]))])
                w_pos = np.asarray(hold["r"], dtype=np.float64)
            else:
                Psi_ref = np.stack([self._psi(self.ref_xi[b][all_rows], all_rows) for b in range(len(self.ref_xi))])
                w_pos = self.ref_w
            Psi_gen = self._psi(xi_d, d_rows)[None, :]
        else:
            Psi_ref = np.stack([self._psi(self.ref_xi[b][all_rows], all_rows) for b in range(len(self.ref_xi))])
            w_pos = self.ref_w
            Psi_gen = self._psi(xi_all, all_rows)[None, :]
        d_stat = self._d_steps(Psi_ref, w_pos, Psi_gen, self.inner_steps_D_commit, d_which,
                               self.hold_d_l2 if d_which == "h_phi" else None)

        # ------------------------------------------------ (44) loss of the committed composition
        d_act = float(_sigmoid(float(getattr(self, d_which) @ Psi_gen[0])))
        lg_plan = (last["meta"] or {}).get("logit_plan")
        if d_which == "h_phi" and lg_plan is not None and np.isfinite(float(lg_plan)):
            # the -log D term of eq. (44) saturates once D pushes the committed side to 0 (-log D
            # ~ 13 for every plan alike).  In hold mode with real positives it is replaced by the
            # CLIPPED NEGATIVE LOGIT of D on the exact PLAN rows of the published candidate - the
            # realizer follows the plan at ~0.99, the term is linear in psi and bounded to
            # +-hold_logit_clip, so it keeps a usable preference between plans.
            d_term = float(-np.clip(float(lg_plan), -self.hold_logit_clip, self.hold_logit_clip))
            d_term_kind = f"clipped_negative_logit_on_plan_rows(+-{self.hold_logit_clip})"
        else:
            d_term = float(-np.log(max(d_act, 1e-12)))
            d_term_kind = "neg_log_D_committed"
        ell = float(d_term + self.lambda_fit * fit + self.lambda_f * e_form)
        # ------------------------------------------------ statistics (FRAG_CONTRACT)
        stat = self._commit_statistics(unit, all_rows, d_act, reference, xi_all)
        if hold is not None and bool(last["meta"].get("hold")):
            # the sample is (slot, level-space w); the slot index is the same object in every hold
            n_w = max(1, self.M - 1)
            b_i = int(last["meta"].get("slot", 0))
            w_i = np.asarray(last["meta"].get("w", np.zeros(n_w)), dtype=np.float64)
            if w_i.shape != (n_w,):
                w_i = np.zeros(n_w)
            logp_draw = last["meta"].get("logp_draw")
            if logp_draw is None or not np.isfinite(float(logp_draw)):
                logp_draw = self._hold_log_p(b_i, w_i)
        else:
            b_i = int(last["meta"].get("b", 0)) if last["meta"] else 0
            w_i = np.asarray(last["meta"].get("w", np.zeros(self.r)), dtype=np.float64)
            if w_i.shape != (self.r,):
                w_i = np.zeros(self.r)
            logp_draw = last["meta"].get("logp_draw")
            if logp_draw is None or not np.isfinite(float(logp_draw)):
                logp_draw = self._log_p(b_i, w_i)
        if self.loss_ema is None:
            baseline, baseline_source = ell, "first_committed_block_own_loss"
        else:
            baseline, baseline_source = float(self.loss_ema), f"ema_of_past_committed_losses(rate={self.baseline_rate})"
        self.loss_ema = ell if self.loss_ema is None else (1.0 - self.baseline_rate) * self.loss_ema + self.baseline_rate * ell
        self._g_buffer.append({"b": b_i, "w": w_i, "ell": ell, "logp_draw": float(logp_draw),
                               "ref_id": last["ref_id"],
                               **({"hold": True} if (hold is not None and bool(last["meta"].get("hold"))) else {})})
        self._g_buffer = self._g_buffer[-self.g_batch_max:]

        # ------------------------------------------------ (45) generator, D / basis / rows fixed
        if hold is not None:
            g_stat = self._hold_g_steps(baseline, self.inner_steps_G_commit)
        else:
            g_stat = self._g_steps(baseline, self.inner_steps_G_commit)

        self._unit_counts["D"] += int(d_stat["updates_D"])
        self._unit_counts["G"] += int(g_stat["updates_G"])
        self._unit_counts["D_rejected"] += int(d_stat["rejected_D_steps"])
        self._unit_counts["update_units"] += 1
        out = {
            "D_loss_before": d_stat["D_loss_before"], "D_loss_after": d_stat["D_loss_after"],
            "D_mean_ref_before": d_stat["D_mean_ref_before"], "D_mean_ref": d_stat["D_mean_ref"],
            "D_mean_actual_before": d_stat["D_mean_gen_before"], "D_mean_actual": d_stat["D_mean_gen"],
            "D_actual_after_D_step": d_act,
            "G_loss": ell, "G_loss_terms": {"D_term": d_term, "D_term_kind": d_term_kind,
                                            "neg_log_D_committed": float(-np.log(max(d_act, 1e-12))),
                                            "lambda_fit*fit": self.lambda_fit * fit,
                                            "lambda_f*e_form": self.lambda_f * e_form},
            "baseline": float(baseline), "baseline_source": baseline_source,
            "advantage": float(ell - baseline),
            "updates_D": int(d_stat["updates_D"]), "updates_G": int(g_stat["updates_G"]),
            "rejected_D_steps": int(d_stat["rejected_D_steps"]),
            "backtracked_D_steps": int(d_stat["backtracked_D_steps"]),
            "clipped_G_steps": int(g_stat["clipped_G_steps"]),
            "G_batch_size": int(g_stat["batch_size"]), "importance_ess": g_stat["importance_ess"],
            "ess_stopped_G_steps": int(g_stat["ess_stopped"]),
            "rows": int(len(all_rows)), "free_rows": int(n_free), "accumulated_blocks": int(blocks),
            "n_positive": int(len(Psi_ref)), "n_negative": 1,
            "reference_id": last["ref_id"], "reference_hash": last["ref_hash"],
            "sample": {"b": b_i, "parent_id": str(last["meta"].get("parent_id", "")),
                       "parent_origin": str(last["meta"].get("parent_origin", "")),
                       "w": [float(x) for x in w_i]},
            **({"hold": {"hold_id": int(last["meta"].get("hold_id", -1)),
                         "reference_age_steps": int((stats or {}).get("reference_age_steps", -1)),
                         "slot": str(last["meta"].get("slot_name", "")),
                         "positive_source": self.hold_positive_source,
                         "swap": bool(last["meta"].get("swap")),
                         "active_materials_start": int(last["meta"].get("active_materials_start", -1)),
                         "positives": ("unmanipulated playback of this hold (exact rows), uniform weights"
                                       if d_which == "h_phi" else
                                       "kept reference plans of this hold (exact rows), weights r_b"),
                         "negative_rows": int(len(self._hist_blocks) * self.block_rows if d_which == "h_phi" else len(all_rows)),
                         "rows_outside_the_held_reference": int(d_rows_dropped),
                         "alpha": self._hold_alphas().tolist(),
                         "sigma_mean": float(np.exp(self.h_log_sigma).mean())}} if hold is not None else {}),
            "phi_norm": float(np.linalg.norm(self.phi)),
            "alpha": self.alphas().tolist(), "sigma_mean": float(np.exp(self.log_sigma).mean()),
            "order": "D step(s) then G step(s), reference and candidate fixed; never during realization",
            "statistics": stat,
        }
        self._commit_records.append(dict(out, unit=int(unit.index)))
        self.step_traces.append({"unit": int(unit.index), "rows": [int(all_rows[0]), int(all_rows[-1])],
                                 "D_loss_before": out["D_loss_before"], "D_loss_after": out["D_loss_after"],
                                 "D_mean_ref": out["D_mean_ref"], "D_mean_actual": out["D_mean_actual"],
                                 "G_loss": ell, "baseline": float(baseline),
                                 "updates_D": out["updates_D"], "updates_G": out["updates_G"],
                                 "D_committed": stat.get("D_committed"),
                                 "D_random_fragment_mean": stat.get("D_random_fragment_mean"),
                                 "ideal_displacement_per_step": stat.get("ideal_displacement_per_step")})
        return out

    # ------------------------------------------------------------------ statistics (FRAG_CONTRACT)
    def _commit_statistics(self, unit: UnitContext, all_rows: np.ndarray, d_act: float,
                           reference: Optional[Target], xi_all: np.ndarray) -> Dict[str, Any]:
        """Per committed step, with the discriminator as it stands after this step's D update:

          * D of the committed realized composition vs D of FRESH random fragment compositions
            drawn at the same rows (the contract's GAN statistic).  The fresh draws are new
            samples of the reference distribution, never seen by D, and they are not used for any
            update - a pure read-out of what D has learned to separate;
          * the mean per-step displacement of the ideal: d_xi^2 (eq. 22) between the frozen
            reference's rows one model step apart, starting at this block's first row, and the
            same quantity per analysis hop, plus the realized composition's hop displacement.
        """
        out: Dict[str, Any] = {"D_committed": float(d_act), "rows": int(len(all_rows))}
        j0 = int(all_rows[0])
        if reference is not None:
            xi_hat = np.asarray(reference.xi_hat, dtype=np.float64)
            s = int(self.model_step_rows)
            j1 = j0 + s
            if j1 <= unit.J - 1:            # a full model step only, so the mean is not biased low
                d2 = float(self.analyzer.dist2(xi_hat[j1], xi_hat[j0]))
                out["ideal_step_dist2"] = d2
                out["ideal_displacement_per_step"] = float(np.sqrt(max(d2, 0.0)))
            rmeta = reference.meta or {}
            rr = rmeta.get("rows")
            if rr and int(rr[1]) > int(rr[0]):
                w0, w1 = int(rr[0]), int(rr[1]) + 1
                fmw = unit.free_mask[w0:w1]
                if fmw[1:].any():
                    dh = self.analyzer.dist2(xi_hat[w0 + 1:w1], xi_hat[w0:w1 - 1])
                    out["ideal_hop_dist2_window"] = float(dh[fmw[1:]].mean())
        if len(xi_all) > 1:
            fmc = unit.free_mask[all_rows]
            dh = self.analyzer.dist2(xi_all[1:], xi_all[:-1])
            if fmc[1:].any():
                out["realized_hop_dist2"] = float(dh[fmc[1:]].mean())
        if self.fragment_mode() and not (self._hold_active and self.hold_positive_source == "recordings"):
            # in "recordings" mode D lives on psi_h and this read-out (random fragment compositions
            # scored by the legacy phi, which is then never trained) would be meaningless; the
            # equivalent read-outs are D(real windows) and D(candidate plans) per hold
            gl = self._frag_goal_levels
            level = float(gl[j0]) if gl is not None else 0.0
            offsets = np.arange(len(all_rows), dtype=np.int64) * self.hop_frames
            d_rand: List[float] = []
            for _ in range(self.frag_d_samples):
                xi_r, _p, _m = self.analyzer.random_fragment_composition(
                    self.sources, self._stat_rng, offsets, goal_level=level)
                d_rand.append(float(self.discriminator_score(xi_r, all_rows)))
            out["D_random_fragment"] = [float(x) for x in d_rand]
            out["D_random_fragment_mean"] = float(np.mean(d_rand))
            out["D_gap_committed_minus_random"] = float(d_act - np.mean(d_rand))
            out["n_random_fragment_draws"] = int(len(d_rand))
            out["random_fragment_goal_level"] = level
        self._unit_stats.append(dict(out, unit=int(unit.index), row0=j0))
        return out

    def _d_steps(self, Psi_ref: np.ndarray, w_ref: np.ndarray, Psi_gen: np.ndarray,
                 n_steps: int, which: str = "phi", l2: Optional[float] = None) -> Dict[str, Any]:
        """Bounded gradient steps on eq. (43) with backtracking: the D loss never increases.
        `which` selects the parameter vector: the legacy phi, or `h_phi` of the hold-mode feature
        map (real windows vs the committed sound) - the update itself is the same."""
        L0, D_ref0, D_gen0 = self._d_loss(Psi_ref, w_ref, Psi_gen, getattr(self, which), l2)
        L_cur = L0
        nD = 0
        rejected = 0
        backtracked = 0
        for _ in range(int(n_steps)):
            D_ref = _sigmoid(Psi_ref @ getattr(self, which))
            D_gen = _sigmoid(Psi_gen @ getattr(self, which))
            grad = (-(w_ref[:, None] * (1.0 - D_ref)[:, None] * Psi_ref).sum(axis=0)
                    + (D_gen[:, None] * Psi_gen).mean(axis=0)
                    + 2.0 * (self.l2 if l2 is None else float(l2)) * getattr(self, which))
            if not np.all(np.isfinite(grad)):
                rejected += 1
                break
            lr = self.lr_D
            accepted = False
            for _bt in range(3):
                trial = getattr(self, which) - lr * grad
                L_try, _, _ = self._d_loss(Psi_ref, w_ref, Psi_gen, trial, l2)
                if np.isfinite(L_try) and np.all(np.isfinite(trial)) and L_try <= L_cur:
                    setattr(self, which, trial)
                    L_cur = float(L_try)
                    nD += 1
                    accepted = True
                    break
                lr *= 0.5
                backtracked += 1
            if not accepted:
                rejected += 1
                break
        L1, D_ref1, D_gen1 = self._d_loss(Psi_ref, w_ref, Psi_gen, getattr(self, which), l2)
        return {"D_loss_before": float(L0), "D_loss_after": float(L1),
                "D_mean_ref_before": float(D_ref0.mean()), "D_mean_gen_before": float(D_gen0.mean()),
                "D_mean_ref": float(D_ref1.mean()), "D_mean_gen": float(D_gen1.mean()),
                "updates_D": int(nD), "rejected_D_steps": int(rejected),
                "backtracked_D_steps": int(backtracked)}

    def _logp_of(self, q: Dict[str, Any]) -> float:
        """log p_Theta of a stored sample: the legacy (parent, xi-coefficient) law, or the hold law
        (slot, level-space displacement) for samples drawn under a held realizable reference."""
        if q.get("hold"):
            return self._hold_log_p(int(q["b"]), q["w"])
        return self._log_p(int(q["b"]), q["w"])

    def _g_batch(self) -> List[Dict[str, Any]]:
        """Newest committed sample first, older ones added while the self-normalised importance
        weights keep the effective sample size above the trust region (audit §8)."""
        chosen: List[Dict[str, Any]] = []
        for s in reversed(self._g_buffer):
            trial = chosen + [s]
            if len(trial) > 1:
                r = np.exp(np.clip([self._logp_of(q) - q["logp_draw"] for q in trial], -30.0, 30.0))
                ess = float(r.sum() ** 2 / max(1e-300, float((r ** 2).sum())))
                if ess < self.ess_min_fraction * len(trial):
                    break
            chosen = trial
        return list(reversed(chosen))

    def _g_steps(self, baseline: float, n_steps: int) -> Dict[str, Any]:
        """REINFORCE (eq. 45) on the committed samples, D and the basis fixed."""
        batch = self._g_batch()
        adv = np.array([q["ell"] - float(baseline) for q in batch], dtype=np.float64)
        logp_draw = np.array([q["logp_draw"] for q in batch], dtype=np.float64)
        n_s = float(max(1, len(batch)))
        ess_min = self.ess_min_fraction * n_s
        nG = 0
        clipped = 0
        ess_stopped = 0
        ess_trace: List[float] = []
        stop = ""
        for _ in range(int(n_steps)):
            # the batch was drawn under an older p_Theta: every re-use is corrected with
            # self-normalised importance ratios p_Theta / p_draw and stops below the trust region
            logp_new = np.array([self._log_p(q["b"], q["w"]) for q in batch], dtype=np.float64)
            ratio = np.exp(np.clip(logp_new - logp_draw, -30.0, 30.0))
            ess = float(ratio.sum() ** 2 / max(1e-300, float((ratio ** 2).sum())))
            ess_trace.append(ess)
            if len(batch) > 1 and ess < ess_min:
                ess_stopped += 1
                stop = "ess_below_trust_region"
                break
            omega = ratio / max(1e-300, float(ratio.sum())) * n_s
            g_mu = np.zeros_like(self.mu)
            g_ls = np.zeros_like(self.log_sigma)
            g_lg = np.zeros_like(self.logits)
            a = self.alphas()
            sig = np.exp(self.log_sigma)
            for q, adv_i, om in zip(batch, adv, omega):
                b = int(q["b"])
                w = q["w"]
                dw = (w - self.mu[b]) / np.maximum(sig[b] ** 2, 1e-12)
                g_mu[b] += om * adv_i * dw
                g_ls[b] += om * adv_i * (((w - self.mu[b]) ** 2) / np.maximum(sig[b] ** 2, 1e-12) - 1.0)
                oh = np.zeros(self.n_parents)
                oh[b] = 1.0
                g_lg += om * adv_i * (oh - a)
            g_mu, g_ls, g_lg = g_mu / n_s, g_ls / n_s, g_lg / n_s
            gn = float(np.sqrt((g_mu ** 2).sum() + (g_ls ** 2).sum() + (g_lg ** 2).sum()))
            if not np.isfinite(gn):
                stop = "non_finite_gradient"
                break
            if gn <= 1e-15:
                stop = "zero_gradient (advantage 0)"
                break
            if gn > self.grad_clip:
                f = self.grad_clip / gn
                g_mu, g_ls, g_lg = g_mu * f, g_ls * f, g_lg * f
                clipped += 1
            s_mu = np.clip(-self.lr_G * g_mu, -self.step_clip, self.step_clip)
            s_ls = np.clip(-self.lr_G * g_ls, -self.step_clip, self.step_clip)
            s_lg = np.clip(-self.lr_G * g_lg, -self.step_clip, self.step_clip)
            if not (np.all(np.isfinite(s_mu)) and np.all(np.isfinite(s_ls)) and np.all(np.isfinite(s_lg))):
                stop = "non_finite_step"
                break
            self.mu = self.mu + s_mu
            self.log_sigma = self.log_sigma + s_ls
            self.logits = self.logits + s_lg
            self._clip_sigma()
            nG += 1
        return {"updates_G": int(nG), "clipped_G_steps": int(clipped), "batch_size": int(len(batch)),
                "importance_ess": [float(x) for x in ess_trace], "ess_stopped": int(ess_stopped),
                "stop_reason": stop,
                "importance_correction": "self-normalised p_theta/p_draw on re-used committed "
                                         "samples; stop when ESS < fraction * n"}

    # ------------------------------------------------------------------ deprecated hook
    def update(self, unit: UnitContext, history, realizations: Sequence[Realization],
               round_index: int) -> Dict[str, Any]:
        """Not called by the engine any more (a reference must not move while it is being scored,
        audit B5/C4).  The adversarial update lives in `observe_committed`."""
        return {"skipped": "mode.update() is not called; the GAN D/G update unit runs in "
                           "observe_committed on the committed composition"}

    def _unit_statistics(self, unit: UnitContext, chosen: Realization) -> Dict[str, Any]:
        """Per-unit summary of the committed-step statistics (contract: D of the committed
        composition vs D of fresh random fragment compositions; mean per-step displacement of the
        ideal in d_xi units)."""
        st = self._unit_stats

        def col(key: str) -> List[float]:
            return [float(s[key]) for s in st if isinstance(s.get(key), (int, float))]

        d_c, d_r = col("D_committed"), col("D_random_fragment_mean")
        disp, d2 = col("ideal_displacement_per_step"), col("ideal_step_dist2")
        hop_i, hop_r = col("ideal_hop_dist2_window"), col("realized_hop_dist2")
        paired = [(float(s["D_committed"]), float(s["D_random_fragment_mean"])) for s in st
                  if isinstance(s.get("D_random_fragment_mean"), float)]
        out: Dict[str, Any] = {
            "unit": int(unit.index), "steps": len(st),
            "mode": ("fragment" if self.fragment_mode() else "baseline_bank"),
            "reference_source": self.reference_source,
            # --- contract statistic: D(committed) vs D(fresh random fragment compositions)
            "D_committed_mean": float(np.mean(d_c)) if d_c else None,
            "D_random_fragment_mean": float(np.mean(d_r)) if d_r else None,
            "D_gap_committed_minus_random_mean": (float(np.mean([a - b for a, b in paired]))
                                                  if paired else None),
            "D_committed_above_random_fraction": (float(np.mean([1.0 if a > b else 0.0 for a, b in paired]))
                                                  if paired else None),
            "D_random_fragment_draws_per_step": (int(self.frag_d_samples) if self.fragment_mode() else 0),
            # --- contract statistic: movement of the ideal
            "mean_ideal_displacement_per_step": float(np.mean(disp)) if disp else None,
            "mean_ideal_dist2_per_step": float(np.mean(d2)) if d2 else None,
            "mean_ideal_dist2_per_hop": float(np.mean(hop_i)) if hop_i else None,
            "mean_realized_dist2_per_hop": float(np.mean(hop_r)) if hop_r else None,
            "displacement_units": "d_xi^2 (eq. 22, free rows); *_displacement = sqrt of it",
            "model_step_seconds": self.model_step_seconds,
            "D_chosen_block_mean": float(self._d_block_mean(chosen.candidate.xi)),
            "D_reference_block_mean": float(np.mean([self._d_block_mean(x) for x in self.ref_xi])),
        }
        rs = self.reference_summaries[-1] if self.reference_summaries else {}
        for k in ("reference_step_dist2_mean", "reference_hop_dist2_mean",
                  "reference_step_displacement_mean"):
            if k in rs:
                out[k] = rs[k]
        if self._hold_active and self._hold_records:
            hr = self._hold_records

            def hcol(key: str) -> List[float]:
                return [float(h[key]) for h in hr if isinstance(h.get(key), (int, float))]

            d_ref = [float(c["D_mean_ref"]) for c in self._commit_records
                     if isinstance(c.get("D_mean_ref"), float)]
            req = [float(d["requested_dist2"]) for h in hr for d in h["draws"]]
            ratio = [float(d["requested_over_local_flutter"]) for h in hr for d in h["draws"]]
            slots = [str(d["slot"]) for h in hr for d in h["draws"]]
            rp = [h["real_positives"] for h in hr if h.get("real_positives")]

            def rcol(key: str) -> List[float]:
                return [float(r[key]) for r in rp if isinstance(r.get(key), (int, float))]

            out["hold"] = {
                "holds": len(hr), "reference_source": self.reference_source,
                "positive_source": self.hold_positive_source,
                # --- FIDELITY_CONTRACT (C): D(committed) vs D(real, unmanipulated windows)
                "D_real_windows_mean": (float(np.mean(rcol("D_real_windows_mean"))) if rp else None),
                "D_gap_committed_minus_real": (
                    float(np.mean(d_c) - np.mean(rcol("D_real_windows_mean")))
                    if (d_c and rcol("D_real_windows_mean")) else None),
                "D_candidate_plans_mean": (float(np.mean(rcol("D_candidate_plans_mean"))) if rp else None),
                "D_unchanged_continuation_mean": (float(np.mean(rcol("D_unchanged_continuation"))) if rp else None),
                "real_windows_per_hold": (float(np.mean(rcol("real_windows"))) if rp else None),
                "psi_rows_per_window": (float(np.mean(rcol("psi_rows"))) if rp else None),
                "psi_scale_source": self.h_psi_scale_source, "psi_dim": int(self.d_psi_hold),
                "h_phi_norm": float(np.linalg.norm(self.h_phi)),
                "h_phi_where_motion": self._h_phi_split(),
                # does the generator actually pick the plans D finds more real?
                "above_median_note": "share with D >= the median of the kept candidates (ties count, only 2-3 candidates survive the narrowing)",
                "chosen_above_median_share": (
                    float(np.mean([1.0 if r.get("chosen_above_median") else 0.0 for r in rp
                                   if r.get("chosen_above_median") is not None]))
                    if any(r.get("chosen_above_median") is not None for r in rp) else None),
                "D_published_plan_mean": (float(np.mean(rcol("D_published_plan"))) if rp else None),
                "parent_above_median_share": (
                    float(np.mean([1.0 if r.get("parent_above_median") else 0.0 for r in rp
                                   if r.get("parent_above_median") is not None]))
                    if any(r.get("parent_above_median") is not None for r in rp) else None),
                "D_chosen_parent_mean": (float(np.mean(rcol("D_chosen_parent"))) if rp else None),
                "D_candidate_plans_median_mean": (float(np.mean(rcol("D_candidate_plans_median"))) if rp else None),
                "slot_alphas_first": (hr[0]["slot_alphas"] if hr else None),
                "slot_alphas_last": (hr[-1]["slot_alphas"] if hr else None),
                "real_position_sources": {k: int(sum(int((h.get("real_positives") or {})
                                                         .get("position_sources", {}).get(k, 0)) for h in hr))
                                          for k in ("current", "committed", "mixture_candidate", "random")},
                "cap": (self._cap_statistics(hr) if self.hold_cap > 0 else
                        {"max_active_materials": 0, "note": "no polyphony cap in this job"}),
                "generator_D_term": (f"clipped negative logit of D on the published plan's rows "
                                     f"(+-{self.hold_logit_clip})"
                                     if self.hold_positive_source == "recordings" else "-log D(committed)"),
                # --- contract statistic: D(committed) vs D(references) - do they approach?
                "D_committed_mean": out["D_committed_mean"],
                "D_hold_reference_mean": (float(np.mean(d_ref)) if d_ref else None),
                "D_gap_committed_minus_hold_reference": (
                    float(np.mean([a - b for a, b in zip(d_c, d_ref)]))
                    if (d_c and d_ref and len(d_c) == len(d_ref)) else None),
                # --- narrowing
                "candidate_plans_per_hold": float(np.mean(hcol("candidate_plans_drawn"))),
                "kept_plans_per_hold": float(np.mean(hcol("candidate_plans_kept"))),
                "kept_fraction_mean": float(np.mean(hcol("kept_fraction"))),
                "move_over_own_flutter_kept_mean": float(np.mean(
                    [float(x) for h in hr for x in h["move_over_flutter_kept"]] or [0.0])),
                "move_over_own_flutter_min": float(self.hold_ratio_min),
                "band_widened_holds": int(sum(1 for h in hr if h["band_widened"])),
                "reference_spread_before_dist2": float(np.mean(hcol("reference_spread_before_dist2"))),
                "reference_spread_after_dist2": float(np.mean(hcol("reference_spread_after_dist2"))),
                "band_dist2_mean": [float(np.mean([h["band_dist2"][0] for h in hr])),
                                    float(np.mean([h["band_dist2"][1] for h in hr]))],
                "local_flutter_dist2_mean": float(np.mean(hcol("flutter_local_dist2"))),
                "realized_flutter_ema_dist2_mean": float(np.mean(hcol("flutter_realized_ema_dist2"))),
                "do_nothing_dist2_mean": float(np.mean(hcol("do_nothing_dist2"))),
                # --- requested move (the engine measures the same quantity in hold_summary)
                "requested_dist2_mean": (float(np.mean(req)) if req else None),
                "requested_over_local_flutter_mean": (float(np.mean(ratio)) if ratio else None),
                "plan_rows_calls": int(sum(hcol("plan_rows_calls"))),
                "slot_share_chosen": {n: float(np.mean([1.0 if s == n else 0.0 for s in slots]))
                                      for n in self.hold_slots} if slots else {},
                "slot_alphas": {n: float(v) for n, v in zip(self.hold_slots, self._hold_alphas())},
                "level_displacement_sigma_mean": float(np.exp(self.h_log_sigma).mean()),
                "level_displacement_mu_absmax": float(np.abs(self.h_mu).max()),
                "generator_inheritance": self.h_inheritance,
                "internal_iterations_note": "candidate plans / plan_rows calls / D and G steps are "
                                            "internal iterations, not musical time",
            }
        return out

    # ================================================================== 10.6 lineage
    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        meta = chosen.target.meta or {}
        chosen_parent = str(meta.get("parent_id", ""))
        kept: List[Dict[str, Any]] = []

        def _add(real: Realization, origin: str) -> str:
            pid = f"gan:u{unit.index}:c{real.candidate.id}"
            rec = {"id": pid, "unit": int(unit.index),
                   "blocks": self.objective.phase_blocks(unit, real.candidate.xi),
                   "score": float(real.total), "origin": origin,
                   "mode_error": float(real.mode_error), "e_form": float(real.e_form),
                   "D": float(self._d_block_mean(real.candidate.xi)),
                   "from_parent": str((real.target.meta or {}).get("parent_id", ""))}
            history.add_parent(rec)
            kept.append({k: v for k, v in rec.items() if k != "blocks"})
            return pid
        # the chosen parent record is the FINAL composite trajectory of the unit (warm start plus
        # the accepted local corrections against the frozen references), not the warm-start plan
        chosen_id = _add(chosen, "gan_chosen_final_composite")
        alts = sorted([a for a in alternatives if a is not chosen], key=lambda z: z.total)[:2]
        for a in alts:
            _add(a, "gan_alternative")
        history.selected_parent_ids.append(chosen_parent or chosen_id)

        self.lineage.append({"unit": int(unit.index), "chosen_parent_id": chosen_parent,
                             "chosen_candidate_id": int(chosen.candidate.id),
                             "new_parent_id": chosen_id, "target_id": chosen.target.id,
                             "parent_origin": str(meta.get("parent_origin", "")),
                             "w": [float(x) for x in meta.get("w", [])],
                             "basis_hash": self.basis_hash,
                             "kept_parents": [k["id"] for k in kept],
                             "alternatives_kept": len(alts)})
        for k in self._total_counts:
            self._total_counts[k] += self._unit_counts[k]
        stats = self._unit_statistics(unit, chosen)
        self.statistics.append(dict(stats, per_step=[
            {k: v for k, v in s.items() if k != "D_random_fragment"} for s in self._unit_stats]))
        gen = [{"parent_id": str(pid), "origin": str(org), "mu": self.mu[b].tolist(),
                "log_sigma": self.log_sigma[b].tolist(),
                "sigma": np.exp(self.log_sigma[b]).tolist(),
                "logit": float(self.logits[b]), "alpha": float(self.alphas()[b]),
                # mu / log_sigma are only re-usable while the basis they were learned in is alive
                "basis_hash": self.basis_hash,
                # lineage link: parents stored this unit that descend from this component
                "descendant_ids": [str(k["id"]) for k in kept if str(k["from_parent"]) == str(pid)]}
               for b, (pid, org) in enumerate(zip(self.parent_ids, self.parent_origins))]
        rs = self.reference_summaries[-1]
        history.mode_state["gan"] = {
            "phi": [float(x) for x in self.phi],
            "psi_mean": [float(x) for x in self.psi_mean], "psi_std": [float(x) for x in self.psi_std],
            "psi_scale_source": self.psi_scale_source, "psi_scale_blocks": int(self.psi_scale_blocks),
            "generator": gen,
            "basis_hash": self.basis_hash, "basis_source": self.basis_source,
            "basis_rank": int(self.r), "basis_sign_rule": self.basis_sign_rule,
            "inheritance": dict(self.inheritance), "inheritance_detail": dict(self.inheritance_detail),
            "baseline_ema": (None if self.loss_ema is None else float(self.loss_ema)),
            "update_counts": {"D": int(self._total_counts["D"]), "G": int(self._total_counts["G"]),
                              "rejected_D": int(self._total_counts["D_rejected"]),
                              "update_units": int(self._total_counts["update_units"]),
                              "skipped_blocks": int(self._total_counts["skipped_blocks"])},
            "reference_weights_summary": {
                "unit": int(unit.index), "n": int(rs["n_references"]),
                "max_weight": float(max(rs["weights"])), "entropy_nats": float(rs["weight_entropy_nats"]),
                "mean_J_ref": float(np.mean(rs["J_ref"])),
                "source": str(rs.get("source", "")),
                "diversity_mean_pair_dist2": float(rs["diversity_mean_pair_dist2"])},
            "statistics": {k: v for k, v in stats.items() if k != "per_step"},
            **({"hold_generator": {
                "slots": list(self.hold_slots), "logits": [float(x) for x in self.h_logits],
                "mu": [[float(v) for v in row] for row in self.h_mu],
                "log_sigma": [[float(v) for v in row] for row in self.h_log_sigma],
                "level_dim": int(max(1, self.M - 1)), "inheritance": self.h_inheritance,
                "note": "the displacement w lives in LEVEL space (material gains), whose acoustic "
                        "meaning does not change with the per-unit basis B_ref, so mu / log_sigma "
                        "are inherited instead of re-initialised"},
                "hold_discriminator": {
                    "positive_source": self.hold_positive_source,
                    "phi": [float(x) for x in self.h_phi],
                    "psi_mean": [float(x) for x in self.h_psi_mean],
                    "psi_std": [float(x) for x in self.h_psi_std],
                    "psi_scale_source": self.h_psi_scale_source,
                    "psi_scale_n": (0 if self.h_psi_provisional else int(self.h_psi_scale_n)),
                    "psi_scale_provisional": bool(self.h_psi_provisional),
                    "psi_scale_varying_fraction": float(self.h_psi_scale_varying_fraction),
                    "feature_names": list(self.h_psi_names)}} if self._hold_seen else {}),
            "lineage": [{"unit": int(l["unit"]), "chosen_parent_id": str(l["chosen_parent_id"]),
                         "chosen_candidate_id": int(l["chosen_candidate_id"]),
                         "new_parent_id": str(l["new_parent_id"])} for l in self.lineage][-8:],
            "last_unit": int(unit.index),
        }
        self.generator_traces.append({
            "unit": int(unit.index), "logits": self.logits.tolist(), "alpha": self.alphas().tolist(),
            "mu": self.mu.tolist(), "sigma": np.exp(self.log_sigma).tolist(),
            "parent_ids": list(self.parent_ids), "parent_origins": list(self.parent_origins),
            "history_parents_used": int(self.n_history_parents_used),
            "warm_started_parents": list(self.warm_started), "warm_start_source": self.warm_start_source,
            "inheritance": dict(self.inheritance), "inheritance_detail": dict(self.inheritance_detail),
            "basis_rank": int(self.r), "basis_source": self.basis_source, "basis_hash": self.basis_hash,
            "basis_sign_rule": self.basis_sign_rule,
            "basis_singular_values": [float(x) for x in self.basis_sv],
            "basis_frobenius_norms": self.basis_norms,
            "parent_weight_note": "near-uniform parent weights are not a defect; no rule forces "
                                  "parent bias, history-parent adoption or mode collapse"})
        self.discriminator_traces.append({
            "unit": int(unit.index), "phi": self.phi.tolist(), "phi_norm": float(np.linalg.norm(self.phi)),
            "feature_names": list(self.psi_names), "feature_scale_mean": self.psi_mean.tolist(),
            "feature_scale_std": self.psi_std.tolist(), "feature_scale_source": self.psi_scale_source,
            "feature_rows": f"committed blocks of {self.block_rows} rows "
                            f"({self.commit_seconds} s at hop {self.hop_seconds:.3f} s), free windows only",
            "D_chosen_realized_block_mean": float(self._d_block_mean(chosen.candidate.xi)),
            "D_mean_reference_block_mean": float(np.mean([self._d_block_mean(x) for x in self.ref_xi]))})
        self.update_counts.append({"unit": int(unit.index), "per_commit": list(self._commit_records),
                                   "unit_totals": dict(self._unit_counts),
                                   "cumulative": dict(self._total_counts),
                                   "inner_steps_per_update_unit": {"D": self.inner_steps_D_commit,
                                                                   "G": self.inner_steps_G_commit},
                                   "pending_blocks_at_unit_end": len(self._pending),
                                   "pending_reason": (None if not self._pending else
                                                      f"{int(unit.free_mask[np.concatenate([p['rows'] for p in self._pending])].sum())} "
                                                      f"free rows < {self.min_block_free_rows}: no D/G update for them")})
        # re-enabling a past composition line at new absolute times: whatever feasibility has
        # changed shows up as the realization gap of the windows whose reference came from a
        # history parent
        gaps = [float(rec["G_loss_terms"]["lambda_fit*fit"]) / max(1e-12, self.lambda_fit)
                for rec in self._commit_records
                if rec.get("sample", {}).get("parent_origin") == "history_parent"]
        self.unit_traces.append({
            "unit": int(unit.index), "chosen_target": chosen.target.id,
            "chosen_parent_id": chosen_parent, "normalized_mode_error": float(chosen.normalized_mode_error),
            "D_chosen_block_mean": float(self._d_block_mean(chosen.candidate.xi)),
            "update_units": int(self._unit_counts["update_units"]),
            "updates_D": int(self._unit_counts["D"]), "updates_G": int(self._unit_counts["G"]),
            "skipped_blocks": int(self._unit_counts["skipped_blocks"]),
            "n_references": int(rs["n_references"]),
            "reference_source": self.reference_source,
            "history_parents_used": int(self.n_history_parents_used),
            "inheritance": dict(self.inheritance),
            "hints": [list(h) for h in getattr(self, "_last_hints", [])],
            "history_parent_realization_gap_mean_dist2": (float(np.mean(gaps)) if gaps else None),
            "history_parent_windows": len(gaps),
            "statistics": {k: v for k, v in stats.items() if k != "per_step"}})
        if self._hold_active:
            self.hold_traces.append({"unit": int(unit.index), "holds": list(self._hold_records),
                                     "summary": stats.get("hold", {})})
        self._pending = []
        self._hold = None

    # ================================================================== checks / trace
    def signature(self) -> np.ndarray:
        parents_mean = self.parent_xi[:, self.fm_idx, :].mean(axis=(0, 1))
        out = [self.phi, self.mu.ravel(), self.log_sigma.ravel(), self.logits, parents_mean]
        if self._hold_seen:      # hold mode: the hold generator is part of the ideal distribution
            out += [self.h_logits, self.h_mu.ravel(), self.h_log_sigma.ravel(), self.h_phi]
        return np.concatenate(out)

    def trace(self) -> Dict[str, Any]:
        return {
            "generator_parameters": self.generator_traces,
            "discriminator_parameters": self.discriminator_traces,
            "update_counts": self.update_counts,
            "lineage": self.lineage,
            "reference_summary": self.reference_summaries,
            "statistics": self.statistics,
            **({"hold_mode": {
                "active": True,
                "law": "per hold the reference distribution is a set of EXACT plans that continue "
                       "from the current positions / levels and are narrowed to a distance band "
                       "around the committed history (the anchor = the mean of the last committed "
                       "block); eq. (40) weights the kept plans; p_Theta(b, w) draws an origin "
                       "class (slot) from a categorical carried across holds and units, one kept "
                       "plan inside it with the eq. (40) weights, and a Gaussian displacement w of "
                       "the MATERIAL LEVELS of the plan's steps (clipped into [0, 1]); the "
                       "published ideal is the exact plan_rows of that displaced plan and the plan "
                       "itself is in Target.meta['plan']",
                "realizability": "xi_hat[rows] = plan_rows(plan) of the engine's own evaluator; the "
                                 "xi-space displacement B_ref w of eq. (41) is NOT added in hold "
                                 "mode (it is not realizable) - it stays in the legacy path",
                "slots": list(self.hold_slots),
                "slot_meaning": {
                    "levels_only": "level moves of the materials, no jump",
                    "committed_positions": "jump to a position of the committed history (events' "
                                           "src_position, positions the realizer committed earlier)",
                    "fragment_candidates": "jump to an.fragment_candidates for the band profile of "
                                           "the recent committed compositions",
                    "random_positions": "fresh uniform position (the exploration share)"},
                "slot_share_drawn": {n: s for n, s in zip(self.hold_slots, self.hold_slot_share)},
                "narrowing": "keep the plans whose rows at the end of the hold lie in "
                             f"[{self.hold_band_lo}, {self.hold_band_hi}] x (the hold's own flutter, "
                             "measured on the UNCHANGED continuation) x (hold_move_scale x openness)^2 "
                             "of the anchor: neither the unchanged sound nor an unrelated mix",
                "discriminator": (
                    ("REAL DATA (docs/FIDELITY_CONTRACT.md C): positives = exact composition rows of "
                     "UNMANIPULATED playback on the rows of the hold - every track plays on "
                     "continuously from a random position of its own source at a constant level, the "
                     "levels drawn from the committed level vectors and the goal track on its "
                     f"schedule ({self.hold_real_positives} windows per hold, "
                     f"{self.hold_psi_blocks} commit blocks each, computed with the analyzer's exact "
                     "Gram machinery).  Negative = the committed sound over the last "
                     f"{self.hold_psi_blocks} commit blocks.  The feature map psi_h carries explicit "
                     "MOTION terms (block-to-block change of the band features and of the "
                     "contributions, hop / block / within-block / span distances), so D judges how "
                     "the sound MOVES, not only where it is; its standardisation is frozen from the "
                     "real windows of the first hold and its parameters h_phi are inherited across "
                     "units.  D step(s) then G step(s) once per committed block."
                     if self.hold_positive_source == "recordings" else
                     "positives = the kept plans' exact rows on the committed rows with weights r_b; "
                     "negative = the committed composition on the same rows; D step(s) then G "
                     "step(s) once per committed block")),
                "positive_source": self.hold_positive_source,
                "positive_source_note": "hold_positive_source = 'recordings' (default) | 'plans' "
                                        "(the previous law: random plans moved close to the sound, "
                                        "which made the D gap partly self-made)",
                "psi_h_features": list(self.h_psi_names),
                "psi_h_standardisation": {"source": self.h_psi_scale_source, "n": int(self.h_psi_scale_n),
                                          "mean": [float(x) for x in self.h_psi_mean],
                                          "std": [float(x) for x in self.h_psi_std]},
                "h_phi": [float(x) for x in self.h_phi],
                "h_phi_where_motion": self._h_phi_split(),
                "h_phi_note": "a positive weight means the feature makes a window look like UNMANIPULATED playback; the largest components say what D actually separates on",
                "generator_update": "REINFORCE (eq. 45) on (slot, level displacement) with the same "
                                    "importance correction / ESS trust region as the legacy law",
                "config_defaults_used": {
                    "hold_move_scale": float(self.hold_move_scale),
                    "hold_reference_candidates": int(self.hold_candidates),
                    "hold_band_lo": float(self.hold_band_lo), "hold_band_hi": float(self.hold_band_hi),
                    "hold_move_over_flutter_min": float(self.hold_ratio_min),
                    "hold_move_over_flutter_max": float(self.hold_ratio_max),
                    "hold_level_move": float(self.hold_level_move),
                    "hold_level_floor": float(self.hold_level_floor),
                    "hold_magnitude_min": float(self.hold_magnitude_range[0]),
                    "hold_magnitude_max": float(self.hold_magnitude_range[1]),
                    "hold_jump_probability": float(self.hold_jump_probability),
                    "hold_search_rows": int(self.hold_search_rows),
                    "hold_keep_min": int(self.hold_keep_min), "hold_keep_max": int(self.hold_keep_max),
                    "hold_fragment_pool": int(self.hold_fragment_pool),
                    "hold_steadiness_factor": int(self.hold_steady_factor),
                    "hold_max_jumps_per_plan": int(self.hold_max_jumps),
                    "hold_w_sigma_init": float(self.hold_w_sigma_init),
                    "hold_w_sigma_min": float(self.hold_w_sigma_min),
                    "hold_w_sigma_max": float(self.hold_w_sigma_max),
                    "hold_flutter_floor": float(self.hold_flutter_floor),
                    "hold_positive_source": str(self.hold_positive_source),
                    "hold_real_positives": int(self.hold_real_positives),
                    "hold_psi_blocks": int(self.hold_psi_blocks),
                    "hold_psi_scale_samples": int(self.hold_psi_scale_samples),
                    "hold_real_near_share": float(self.hold_real_near_share),
                    "hold_real_offset_seconds": float(self.hold_real_offset_seconds),
                    "hold_logit_clip": float(self.hold_logit_clip),
                    "hold_d_l2": float(self.hold_d_l2),
                    "hold_lr_G_scale": float(self.hold_lr_G_scale),
                    "hold_real_cap_redraw": float(self.hold_real_cap_redraw),
                    "hold_psi_scale_min_var": float(self.hold_psi_scale_min_var),
                    "note": "read from mode_defaults.gan with .get; not present in config.py "
                            "DEFAULTS (shared file not edited), so a project file cannot set them "
                            "until they are added there"},
                "approximations": [
                    "E_hist of eq. (40) is evaluated on a full-unit trajectory whose rows outside "
                    "the held window are the anchor, so within one hold it varies only through the "
                    "window rows: r_b is dominated by E_form there",
                    "the per-hold restriction of p_Theta to the slots that survived the narrowing "
                    "is treated as part of the environment, not of p_Theta (it cancels in the "
                    "importance ratio only when the same slots survive)",
                    "candidate plans are screened on a row subset (flux is then computed between "
                    "non-adjacent rows); the kept plans and the published plan are evaluated on "
                    "all rows"],
                "per_unit": self.hold_traces}} if self._hold_seen else {}),
            "reference_distribution": {
                "source": self.reference_source,
                "fragment_vocabulary_available": bool(self.fragment_mode()),
                "rule": ("eq. 40 over random FRAGMENT composition trajectories: a new random "
                         "fragment composition (positions + material levels) every "
                         f"{self.frag_segment_seconds} s along the unit, exact mixture rows at "
                         "hop-spaced offsets after the sampled positions (what sounds after a "
                         "jump)"
                         + ("" if self.frag_offset_rows <= 0 else
                            f", only the first {self.frag_offset_rows} rows of a block sampled and "
                            "the rest linearly interpolated")
                         + ", goal level from the form's exposure policy per phase, GOAL_HOLD "
                           "rows = xi_goal"
                         if self.fragment_mode() else
                         "eq. 40 over the private Bank reference distribution (random legal gain "
                         "plans) - the baseline behaviour when no fragment bank exists"),
                "n_references_per_unit": (int(self.frag_n_refs) if self.fragment_mode()
                                          else int(self.n_ref_target)),
                "weights": "r_b ∝ exp(-J_ref/tau_ref), J_ref = E_form + 0.2 E_hist on the "
                           "reference trajectory (E_form from the sampled compositions' c / E)",
                "config_defaults_used": {
                    "frag_references": int(self.frag_n_refs),
                    "frag_segment_seconds": float(self.frag_segment_seconds),
                    "frag_offset_rows": int(self.frag_offset_rows),
                    "frag_discriminator_samples": int(self.frag_d_samples),
                    "note": "read from mode_defaults.gan with .get; not present in config.py "
                            "DEFAULTS (shared file not edited), so a project file cannot set them "
                            "until they are added there"},
                "statistic": "D of the committed composition vs D of fresh random fragment "
                             "compositions drawn at the same rows (per step, summarised per unit); "
                             "the fresh draws come from a separate seeded stream (config seed + "
                             "977) so that the measurement never moves the job rng, are never used "
                             "for a D or G update, and are not a goal test",
                "ideal_motion": "mean per-step displacement of the frozen ideal, d_xi^2 (eq. 22) "
                                f"between rows {self.model_step_rows} apart "
                                f"({self.model_step_seconds} s model step)"},
            "update_unit": {
                "when": "once per committed block (engine commit step), never during realization",
                "fixed_during_the_update": ["reference set r_b", "frozen window reference",
                                            "committed realized composition", "basis B_ref"],
                "order": "D step(s) with backtracking, then G step(s) by REINFORCE",
                "commit_rows": int(self.block_rows), "commit_seconds": float(self.commit_seconds),
                "inner_steps_D": int(self.inner_steps_D_commit), "inner_steps_G": int(self.inner_steps_G_commit),
                "positives": "reference compositions restricted to the committed rows, weights r_b",
                "negative": "the committed realized composition on the same rows",
                "short_block_rule": f"fewer than {self.min_block_free_rows} free rows: accumulate "
                                    f"blocks; if none are left at the unit end the reason is recorded",
                "baseline": "EMA of past committed losses (rate "
                            f"{self.baseline_rate}); the first committed block uses its own loss",
                "C_fail": "not exercised under the frozen-reference engine: every frozen reference "
                          "is realized by the realization layer, so no proposal is unrealizable",
                "unused_config": {
                    "gan_adversarial_rounds_max": "unused: the update unit is one committed block, "
                                                  "not a search round (mode.update() is not called)",
                    "C_fail": "read but not applied; see above"},
            },
            "basis_change_policy": {
                "rule": "mu / log_sigma of the displacement coefficients are re-initialised whenever "
                        "the basis hash changes; phi, logits, parent acoustics, parent ids and the "
                        "common history are inherited (audit B7 / D4.1, option 1)",
                "transport": "coordinate transport (option 2) is NOT implemented",
                "sign_rule": getattr(self, "basis_sign_rule", ""),
                "per_unit": [{"unit": t["unit"], "basis_hash": t["basis_hash"],
                              "inheritance": t["inheritance"],
                              "basis_changed": t["inheritance_detail"].get("basis_changed")}
                             for t in self.generator_traces]},
            "equations": {
                "40": "r_b ∝ exp(-J_ref/tau_ref), J_ref = E_form + 0.2 E_hist (discriminator not used)",
                "41": "Xi_hat = Xi_b^parent + B_ref w, p(b,w) = alpha_b N(w; mu_b, diag sigma_b^2)",
                "42": "D_phi(Xi) = sigmoid(phi^T psi(Xi)) on free windows of realized compositions",
                "43": "L_D = -sum_b r_b log D(Xi_b^ref) - log(1-D(Xi^actual)) + l2 ||phi||^2",
                "44": "l = -log D(Xi^actual) + lambda_fit <d_xi^2> + lambda_f E_form",
                "45": "grad E l = E[(l - b_past) grad log p_Theta(b, w)]"},
            "claims": "bounded co-adaptive updates on committed realized compositions; no "
                      "convergence, no equilibrium, no optimal GAN, no discriminator threshold used "
                      "as a goal test; near-uniform parent weights are not treated as a defect",
            "warnings": list(self.warnings), "units": self.unit_traces,
            "steps": self.step_traces[-96:]}

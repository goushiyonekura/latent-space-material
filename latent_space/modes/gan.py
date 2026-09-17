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
                phi: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Eq. (43) with the reference weights r_b on the positive term."""
        D_ref = _sigmoid(Psi_ref @ phi)
        D_gen = _sigmoid(Psi_gen @ phi) if len(Psi_gen) else np.zeros(0)
        pos = -float((w_ref * np.log(np.clip(D_ref, 1e-12, 1.0))).sum())
        neg = -float(np.log(np.clip(1.0 - D_gen, 1e-12, 1.0)).mean()) if len(D_gen) else 0.0
        return pos + neg + self.l2 * float(phi @ phi), D_ref, D_gen

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
        them; the sample that produced it stays in Target.meta and is never altered afterwards."""
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
        Psi_ref = np.stack([self._psi(self.ref_xi[b][all_rows], all_rows) for b in range(len(self.ref_xi))])
        Psi_gen = self._psi(xi_all, all_rows)[None, :]
        d_stat = self._d_steps(Psi_ref, self.ref_w, Psi_gen, self.inner_steps_D_commit)

        # ------------------------------------------------ (44) loss of the committed composition
        d_act = float(_sigmoid(float(self.phi @ Psi_gen[0])))
        ell = float(-np.log(max(d_act, 1e-12)) + self.lambda_fit * fit + self.lambda_f * e_form)
        # ------------------------------------------------ statistics (FRAG_CONTRACT)
        stat = self._commit_statistics(unit, all_rows, d_act, reference, xi_all)
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
                               "ref_id": last["ref_id"]})
        self._g_buffer = self._g_buffer[-self.g_batch_max:]

        # ------------------------------------------------ (45) generator, D / basis / rows fixed
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
            "G_loss": ell, "G_loss_terms": {"neg_log_D": float(-np.log(max(d_act, 1e-12))),
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
        if self.fragment_mode():
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
                 n_steps: int) -> Dict[str, Any]:
        """Bounded gradient steps on eq. (43) with backtracking: the D loss never increases."""
        L0, D_ref0, D_gen0 = self._d_loss(Psi_ref, w_ref, Psi_gen, self.phi)
        L_cur = L0
        nD = 0
        rejected = 0
        backtracked = 0
        for _ in range(int(n_steps)):
            D_ref = _sigmoid(Psi_ref @ self.phi)
            D_gen = _sigmoid(Psi_gen @ self.phi)
            grad = (-(w_ref[:, None] * (1.0 - D_ref)[:, None] * Psi_ref).sum(axis=0)
                    + (D_gen[:, None] * Psi_gen).mean(axis=0) + 2.0 * self.l2 * self.phi)
            if not np.all(np.isfinite(grad)):
                rejected += 1
                break
            lr = self.lr_D
            accepted = False
            for _bt in range(3):
                trial = self.phi - lr * grad
                L_try, _, _ = self._d_loss(Psi_ref, w_ref, Psi_gen, trial)
                if np.isfinite(L_try) and np.all(np.isfinite(trial)) and L_try <= L_cur:
                    self.phi = trial
                    L_cur = float(L_try)
                    nD += 1
                    accepted = True
                    break
                lr *= 0.5
                backtracked += 1
            if not accepted:
                rejected += 1
                break
        L1, D_ref1, D_gen1 = self._d_loss(Psi_ref, w_ref, Psi_gen, self.phi)
        return {"D_loss_before": float(L0), "D_loss_after": float(L1),
                "D_mean_ref_before": float(D_ref0.mean()), "D_mean_gen_before": float(D_gen0.mean()),
                "D_mean_ref": float(D_ref1.mean()), "D_mean_gen": float(D_gen1.mean()),
                "updates_D": int(nD), "rejected_D_steps": int(rejected),
                "backtracked_D_steps": int(backtracked)}

    def _g_batch(self) -> List[Dict[str, Any]]:
        """Newest committed sample first, older ones added while the self-normalised importance
        weights keep the effective sample size above the trust region (audit §8)."""
        chosen: List[Dict[str, Any]] = []
        for s in reversed(self._g_buffer):
            trial = chosen + [s]
            if len(trial) > 1:
                r = np.exp(np.clip([self._log_p(q["b"], q["w"]) - q["logp_draw"] for q in trial], -30.0, 30.0))
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
        self._pending = []

    # ================================================================== checks / trace
    def signature(self) -> np.ndarray:
        parents_mean = self.parent_xi[:, self.fm_idx, :].mean(axis=(0, 1))
        return np.concatenate([self.phi, self.mu.ravel(), self.log_sigma.ravel(),
                               self.logits, parents_mean])

    def trace(self) -> Dict[str, Any]:
        return {
            "generator_parameters": self.generator_traces,
            "discriminator_parameters": self.discriminator_traces,
            "update_counts": self.update_counts,
            "lineage": self.lineage,
            "reference_summary": self.reference_summaries,
            "statistics": self.statistics,
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

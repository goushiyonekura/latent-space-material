"""Diffusion-type controller (spec §7; audit-2 §C4 / §D1): a *coupling field* over
acoustic-composition space.

The search particles are ideal composition trajectories Xi_hat (never gains, never PCM noise).
With r_i = c_i - c_i^G (contribution minus the goal-only contribution of the same phase) the
reference field energy is eq. (28)

  E_D(Xi_hat) = < lam_phi  mean_k (phi_hat - phi_anchor)^2
                + lam_2    mean_{i<j} w_ij [ r_i - s_ij r_j - o d_ij ]^2
                + lam_3    mean_{(i,j,l) in T} [ r_i r_j - o beta_ijl r_l ]^2
                + eps_D    ||xi_hat||^2 >
              + lam_G < (1-o)^2 d_xi^2(xi_hat, xi_G) >                                 (28)

with the acoustic-feature-dependent coefficients

  phi_anchor : phi block of the probe mix (goal 0.1, each material 0.4)  -- analysis only
  w_ij       = 0.1 + S_ij(t) + 0.5 |corr(M_H)_ij|
  s_ij       = sign(corr(M_H)_ij) when |corr| > 0.2, else +1 if mean_t S_ij >= 0.5 else -1
  chi_i      = unit.chi[:, i]            (tanh-bounded log-energy change of material i)
  d_ij(t)    = 0.25 tanh( h_c,i - s_ij h_c,j + chi_i(t) - chi_j(t) )
  beta_ijl   = 0.25 tanh( M_H,il - M_H,jl )

Possibility search is the finite Langevin recursion eq. (29)

  X_{j+1} = X_j - tau O M_D grad E_D(X_j) + sqrt(2 tau T_D) O L_D eps_j,  L_D L_D^T = M_D > 0

with M_D = I + kappa v v^T (v = the realized composition-change direction embedded in the
c block), O = diag(openness) applied row-wise, AR(1)-correlated eps in *iteration* time, and
GOAL_HOLD rows pinned to xi_G and removed from the free coordinates.  This is a *finite*
search: it is not an exact Gibbs sample and not a trained DDPM.

Real gains are connected through eq. (30)

  E_D(gamma; Xi_hat_a) = < d_xi^2( xi_t(gamma), xi_hat_{a,t} ) > + kappa_E E_D(Xi(gamma))  (30)

so the *realized* composition of a legal candidate is what the field scores; candidates are
never constrained to the field's level set and are never rejected for a large acoustic error.

Audit-2 revision (§C4 / §D1), what changed in this module
---------------------------------------------------------
* D1.1  The "realized pull" of particles toward candidate compositions is GONE from every code
        path used during realization.  `update()` is a no-op that the engine no longer calls.
        The Langevin exploration is kept, and it lives in `prepare_reference` -- i.e. *before*
        the reference of a window is frozen.  Internal (Langevin) iterations are counted and
        reported separately from musical time (commit / lookahead seconds).
* D1.2  `prepare_reference(unit, history, rows, xi_current, n)` explores the field on the
        window rows only, starting from the realized current composition `xi_current`, and
        rebuilds the field coefficients from the *committed* history (h_c, M_H, corr(),
        change_direction, dc_cov).  The coefficients are then frozen for the whole window:
        nothing the mode owns moves while the gain trajectory is being scored.
* D1.3  `window_error` is eq. (30) restricted to the window; the five field terms
        (anchor / pair / triple / ridge / goal) are recorded per step so that anchor and pair
        can be read separately.  The lambda weights are fixed configuration; they are never
        re-scaled automatically from the observed value share.
* D1.4  `observe_committed` re-estimates the composition-change direction v and M_D from the
        *committed* composition only, and persists a small summary that the next
        `prepare_reference` may use.  This is the legitimate history-driven re-enabling.
* D1.5  Unreachable parts of an ideal trajectory (band contributions no material supplies,
        negative contributions, contributions the gain bounds cannot produce) stay in the
        residual and are reported; they are never "realized" by breaking a gain constraint.

Fragment-vocabulary revision (docs/FRAG_CONTRACT.md, 2026-09-16), what changed
-----------------------------------------------------------------------------
Active only in FRAGMENT MODE, i.e. when the analyzer carries the fragment bank
(`getattr(analyzer, "frag_f", None) is not None`, built by `Analyzer.build_fragment_bank` when
`hires.enabled` and `hires.fragment_vocabulary`) AND the engine has handed the mode the source
PCM (`mode.sources = job.sources`, set in `engine.setup`).  Without both, every code path below
is skipped and the module behaves exactly as before (the baseline config `dev/fixture_vae.json`
produces a bit-identical render and gain curve).

* F1  `prepare_reference` integrates the field as a TIME PROCESS in MUSICAL time instead of
      running dimensionless Langevin iterations on window particles:

        xi_{k+1} = xi_k - tau M_D grad E_D(xi_k) dt + sqrt(2 tau T dt) L_D eps_k

      one step per model step (dt = `analysis.model_step_seconds`, 0.5 s in the fragment
      fixture), starting from the realized current composition `xi_current` at the first window
      row; the window rows (hop 0.1 s) are linear interpolations of the model-step knots, so the
      reference is a piecewise-linear path on the same grid on which the realizer switches
      levels (0.25 s Q5 ramp) and commits (0.5 s).  The `n` proposals are `n` INDEPENDENT NOISE
      DRAWS of the same process from the same initial condition (they no longer differ in their
      initialisation blend).  GOAL_HOLD knots and rows stay pinned to xi_G.  The field
      coefficients (w_ij, s_ij, d_ij, beta, M_D) are still rebuilt from the committed history in
      `prepare_reference` and frozen for the window -- unchanged.
      Stability: explicit Euler on eq. (29) needs tau dt L < 2; at the tau the musical time scale
      demands, one Euler step per model step diverges.  The DRIFT of a model step is therefore
      integrated in `frag_drift_substeps` sub-steps of dt/n (same total drift in the small-step
      limit, n times the stability margin) with a trust region of
      `frag_max_displacement_per_step` / n on each sub-step; the FLUCTUATION stays one Wiener
      increment per model step.  The binding fraction of the trust region is recorded.
* F2  ANCHOR TERM: the fixed 0.4 / 0.1 probe mix on the continuous clock is replaced by the mean
      of a small sample of RANDOM FRAGMENT COMPOSITIONS (`an.random_fragment_composition`,
      `frag_anchor_samples` = 24 compositions x `frag_anchor_offsets` = 3 offsets) drawn once per
      unit, per goal-exposure group (goal level 0 wherever `form.goal_exposure` caps it, 1 where
      the goal is exposed, 0.5 in the middle of the deterministic rise).  The sample is drawn
      from a generator seeded from (config seed, unit index): reproducible, it does not perturb
      the Langevin noise stream, and the history-intervention probe sees the same anchor.  Every
      draw (positions and levels) is recorded in the trace.
      `lambda_phi` is unchanged configuration -- what changed is WHERE the anchor lives: it is no
      longer a point of the continuous-clock probe family but the centre of the FRAGMENT SPACE,
      i.e. of the same reachable set from which the realizer picks its jump candidates.  The
      pair and triple terms are untouched, and so is their weight relative to the anchor.
* F3  `frag_step` (tau) and `frag_temperature_per_openness` (T per unit of mean openness) are the
      musical-time coefficients; `step` / `temperature_per_openness` keep their meaning for the
      dimensionless iteration-time recursion that `propose()` (the warm start) still uses.  The
      defaults (tau = 40, T_per_o = 0.0016, 10 drift sub-steps, trust region 1.5) were measured
      on `dev/fixture_frag.json`: they give a mean per-step displacement of the ideal of
      0.43 - 0.58 in d_xi metric units at o = 1 (a single material's full level change is ~0.33,
      a playback jump ~0.36, a complete redraw of the fragment composition ~0.88), with a
      per-step drift/noise ratio of 1.1 - 3.7 and a net (accumulated over a window) drift/noise
      ratio well above 1 -- the field decides where the ideal goes, the temperature only spreads
      the proposals.
* F4  `window_error` and `relation_terms` are unchanged in form.  `observe_committed` keeps the
      v / M_D re-estimation from the committed composition and adds one statistic: the field
      energy of the committed block against the mean field energy of the unit's random fragment
      sample, evaluated with the SAME field parameters on the same number of rows (per commit
      step in `steps`, summarised per unit in `trace()['fragment_statistics_per_unit']`).
* F5  `propose()` is unchanged (the baseline warm start); in fragment mode its particles are
      initialised toward the fragment anchor instead of the probe anchor, because they use the
      same `xi_anchor`.

Configuration keys added by this revision are read with `.get()` only and are NOT in
`config.DEFAULTS`, so no shared file was edited: `frag_step`, `frag_temperature_per_openness`,
`frag_drift_substeps`, `frag_max_displacement_per_step`, `frag_anchor_samples`,
`frag_anchor_offsets`, `frag_anchor_seed`, `frag_energy_probe_samples` (all under
`mode_defaults.diffusion`).  They therefore cannot be set from a project JSON without adding
them to `config.DEFAULTS` first (`load_config` rejects unknown keys).
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, ar1_noise, fix_hold_rows


# ---------------------------------------------------------------------------- field parameters
class FieldParams:
    """All coefficients of eq. (28) for one planning unit, one commit window, or the self-test.

    Every per-time array (`phi_anchor`, `cG`, `w`, `d`, `o`, `xi_goal`, `rows`) is indexed by the
    same row set, so the same class describes the full unit and a restricted window.
    """

    __slots__ = ("d_phi", "M", "d_xi", "n_pairs", "phi_anchor", "cG", "w", "s", "d", "beta",
                 "ti", "tj", "tl", "o", "xi_goal", "W", "rows", "pair_mask", "n_free",
                 "lam_phi", "lam_2", "lam_3", "lam_G", "eps_D")

    def __init__(self, d_phi, M, d_xi, phi_anchor, cG, w, s, d, beta, triples, o, xi_goal, W,
                 rows, lam_phi, lam_2, lam_3, lam_G, eps_D):
        self.d_phi = int(d_phi)
        self.M = int(M)
        self.d_xi = int(d_xi)
        self.n_pairs = max(1, self.M * (self.M - 1) // 2)
        self.phi_anchor = np.asarray(phi_anchor, dtype=np.float64)
        self.cG = np.asarray(cG, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)
        self.s = np.asarray(s, dtype=np.float64)
        self.d = np.asarray(d, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        tri = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        self.ti, self.tj, self.tl = tri[:, 0], tri[:, 1], tri[:, 2]
        self.o = np.asarray(o, dtype=np.float64)
        self.xi_goal = np.asarray(xi_goal, dtype=np.float64)
        self.W = np.asarray(W, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=bool)
        self.n_free = max(1, int(self.rows.sum()))
        self.pair_mask = np.triu(np.ones((self.M, self.M), dtype=np.float64), 1)
        self.lam_phi = float(lam_phi)
        self.lam_2 = float(lam_2)
        self.lam_3 = float(lam_3)
        self.lam_G = float(lam_G)
        self.eps_D = float(eps_D)


def field_terms(P: FieldParams, xi: np.ndarray, want_grad: bool = False):
    """Per-row energy density of eq. (28), its five terms, and (optionally) its *row-local*
    gradient.

    `e_row[t]` is the bracket of eq. (28) at time t including the lam_G term; the field energy
    is `e_row[P.rows].mean()`.  `g[t]` is d e_row[t] / d xi[t] (the gradient of the *sum*
    over evaluation rows); the gradient of the mean is `g * rows / n_free`.
    """
    xi = np.asarray(xi, dtype=np.float64)
    d_phi, M = P.d_phi, P.M
    phi = xi[:, :d_phi]
    c = xi[:, d_phi:d_phi + M]
    r = c - P.cG                                                  # (J, M)
    o = P.o
    n_T = max(1, len(P.ti))

    dphi = phi - P.phi_anchor
    t_anchor = P.lam_phi * (dphi * dphi).mean(axis=1)

    # pair coupling: U_ij = r_i - s_ij r_j - o d_ij
    U = r[:, :, None] - P.s[None, :, :] * r[:, None, :] - o[:, None, None] * P.d
    WU = P.w * U * P.pair_mask[None, :, :]
    t_pair = P.lam_2 * (WU * U).sum(axis=(1, 2)) / P.n_pairs

    # triple coupling: q = r_i r_j - o beta_ijl r_l
    ri, rj, rl = r[:, P.ti], r[:, P.tj], r[:, P.tl]
    q = ri * rj - o[:, None] * P.beta[None, :] * rl
    t_triple = P.lam_3 * (q * q).sum(axis=1) / n_T

    t_ridge = P.eps_D * (xi * xi).sum(axis=1)

    dg = xi - P.xi_goal
    gw = (1.0 - o) ** 2
    t_goal = P.lam_G * gw * ((dg * dg) @ P.W)

    e = t_anchor + t_pair + t_triple + t_ridge + t_goal
    parts = {"anchor": t_anchor, "pair": t_pair, "triple": t_triple, "ridge": t_ridge, "goal": t_goal}

    if not want_grad:
        return e, None, parts

    g = np.zeros_like(xi)
    g[:, :d_phi] += (2.0 * P.lam_phi / d_phi) * dphi
    gc = (2.0 * P.lam_2 / P.n_pairs) * (WU.sum(axis=2) - (P.s[None, :, :] * WU).sum(axis=1))
    coef3 = 2.0 * P.lam_3 / n_T
    for k in range(len(P.ti)):
        qk = coef3 * q[:, k]
        gc[:, P.ti[k]] += qk * rj[:, k]
        gc[:, P.tj[k]] += qk * ri[:, k]
        gc[:, P.tl[k]] -= qk * o * P.beta[k]
    g[:, d_phi:d_phi + M] += gc
    g += (2.0 * P.eps_D) * xi
    g += (2.0 * P.lam_G) * gw[:, None] * P.W[None, :] * dg
    return e, g, parts


def _eval_mask(P: FieldParams, n: int) -> np.ndarray:
    return P.rows if (len(P.rows) == n and P.rows.any()) else np.ones(n, dtype=bool)


def field_energy(P: FieldParams, xi: np.ndarray) -> float:
    """<.> of eq. (28): the mean of the row energy over the evaluation (non-hold) rows."""
    e, _, _ = field_terms(P, xi, want_grad=False)
    return float(e[_eval_mask(P, len(e))].mean())


def field_breakdown(P: FieldParams, xi: np.ndarray) -> Dict[str, float]:
    """The five terms of eq. (28), each averaged over the evaluation rows (audit D1.3)."""
    _, _, parts = field_terms(P, xi, want_grad=False)
    m = _eval_mask(P, len(xi))
    return {k: float(v[m].mean()) for k, v in parts.items()}


def field_row_gradient(P: FieldParams, xi: np.ndarray) -> np.ndarray:
    """Row-local gradient used by the Langevin drift (zero on non-evaluation rows)."""
    _, g, _ = field_terms(P, xi, want_grad=True)
    g = g * P.rows[:, None]
    return g


# ---------------------------------------------------------------------------- the mode
class DiffusionMode(ModeController):
    cli_name = "diffusion"
    internal_id = "diffusion_field"
    state_key = "diffusion"

    def __init__(self, cfg, analyzer, fs, rng, objective):
        super().__init__(cfg, analyzer, fs, rng, objective)
        md = cfg["mode_defaults"]
        p = dict(md.get("diffusion", {}))
        self.p = p
        self.lam_phi = float(p.get("lambda_phi", 1.0))
        self.lam_2 = float(p.get("lambda_2", 1.0))
        self.lam_3 = float(p.get("lambda_3", 0.25))
        self.lam_G = float(p.get("lambda_G", 1.0))
        self.eps_D = float(p.get("epsilon_D", 1e-4))
        self.tau = float(p.get("step", 0.01))
        self.T_per_o = float(p.get("temperature_per_openness", 0.05))
        self.kappa_E = float(p.get("kappa_E", 0.25))
        self.kappa_M = float(p.get("kappa_M", 0.5))
        self.anchor_nongoal = float(p.get("anchor_nongoal_gain", 0.4))
        self.anchor_goal = float(p.get("anchor_goal_gain", 0.1))
        self.rho_t = float(p.get("time_correlation", 0.8))
        # keys read with .get() only (not in config.DEFAULTS; no shared file was edited)
        self.corr_significant = float(p.get("correlation_significance", 0.2))
        self.sign_similarity = float(p.get("similarity_sign_threshold", 0.5))
        self.init_spread = float(p.get("init_spread", 0.5))
        self.max_triples = int(p.get("max_triples", 10))
        self.n_hints = int(p.get("hint_pairs", 3))
        # audit D1.1: the realized pull is disabled; the configured value is reported, never used.
        self.pull_eta_disabled = float(p.get("realized_pull", 0.5))
        # window search (audit D1.2), all bounded
        self.cont_seconds = float(p.get("window_continuation_seconds", 2.5))
        self.window_steps_max = max(1, int(p.get("window_internal_steps_max",
                                               md.get("diffusion_internal_steps_max", 8))))
        self.window_step_budget_per_unit = max(1, int(p.get("window_step_budget_per_unit", 4096)))
        self.relation_lag_seconds = float(p.get("relation_lag_seconds", 1.0))
        # isotropic part of the positive definite search matrix: M_D = m0 (I + kappa v v^T).
        # m0 = 1 reproduces the plain spec form; raising it lengthens the relaxation the finite
        # search can cover (drift grows like m0, fluctuation like sqrt(m0)).  Default: literal.
        self.m_scale = max(1e-9, float(p.get("search_matrix_scale", 1.0)))

        # ---------------- fragment-vocabulary revision (FRAG_CONTRACT, Diffusion section) -------
        # All of these are read with .get() only: none of them is in config.DEFAULTS, so no shared
        # file was edited.  They are used ONLY when the analyzer carries a fragment bank and the
        # engine has handed the mode the source PCM (`self.sources`); the baseline configuration
        # (8 bands, no hires, no `frag_f`) never reaches any of this code.
        #
        # `frag_step` (tau) and `frag_temperature_per_openness` (T per unit openness) replace
        # `step` / `temperature_per_openness` in the *musical-time* integration of
        #     xi_{k+1} = xi_k - tau M_D grad E_D(xi_k) dt + sqrt(2 tau T dt) L_D eps
        # over the window rows (dt = analysis.model_step_seconds).  `step`/`temperature_...`
        # remain the coefficients of the dimensionless iteration-time recursion eq. (29) used by
        # `propose()` (the warm start), which is unchanged.  Defaults are measured on
        # dev/fixture_frag.json: see `_frag_time_process` and the trace key
        # `ideal_displacement_per_model_step`.
        self.frag_tau = float(p.get("frag_step", 40.0))
        self.frag_T_per_o = float(p.get("frag_temperature_per_openness", 0.0016))
        self.frag_anchor_samples = max(1, int(p.get("frag_anchor_samples", 24)))
        self.frag_anchor_offsets = max(1, int(p.get("frag_anchor_offsets", 3)))
        self.frag_anchor_seed = int(p.get("frag_anchor_seed", 104729))
        self.frag_energy_probe_samples = max(1, int(p.get("frag_energy_probe_samples", 8)))
        # explicit Euler on eq. (29) is only stable for tau * dt * L < 2 (L = largest curvature of
        # E_D, dominated by the pair term in the c block); at the tau the musical time scale needs,
        # one Euler step per model step diverges.  The DRIFT of one model step is therefore
        # integrated in `frag_drift_substeps` sub-steps of dt / n (the total drift per model step
        # is unchanged in the small-step limit, the stability limit is n times larger), while the
        # FLUCTUATION stays one Wiener increment per model step -- so the reference is still a
        # piecewise-linear path on the model-step grid, which is what the realizer can follow.
        # `frag_max_displacement_per_step` is a trust region on the drift in d_xi units (a single
        # material's full level change is ~0.33, a jump ~0.36 on the fixture): it is a guard, and
        # the fraction of steps where it binds is recorded.
        self.frag_substeps = max(1, int(p.get("frag_drift_substeps", 10)))
        self.frag_max_step = float(p.get("frag_max_displacement_per_step", 1.5))
        self.model_step_seconds = float(cfg["analysis"].get("model_step_seconds", 0.5))
        self.sources = None                 # set by engine.setup (FRAG_CONTRACT); PCM of every track
        self._frag = False                  # fragment mode active for the current unit
        self.anchor_groups: Dict[float, np.ndarray] = {}
        self.anchor_row_level = np.zeros(1)
        self.anchor_meta: Dict[str, Any] = {}
        self.frag_stats: List[Dict[str, Any]] = []

        self.A = max(1, int(md.get("diffusion_particles", 4)))
        self.steps_max = max(1, int(md.get("diffusion_internal_steps_max", 8)))
        self.max_rounds = max(1, int(cfg["search"]["max_search_rounds"]))
        self.step_budget = self.steps_max * self.max_rounds

        self.Wv = analyzer.weight_vector()
        self.pair_mask = np.triu(np.ones((self.M, self.M)), 1)
        self.triples = self._fixed_triples()
        self.n_T = len(self.triples)
        self.extra_candidate_evaluations = 0

        # internal state (re-built per unit, kept for the signature)
        self.s_ij = np.ones((self.M, self.M))
        self.s_prior = np.ones((self.M, self.M))
        self.w_ij = np.zeros((1, self.M, self.M))
        self.d_ij = np.zeros((1, self.M, self.M))
        self.d_bar = np.zeros((self.M, self.M))
        self.w_bar = np.zeros((self.M, self.M))
        self.beta = np.zeros(self.n_T)
        self.v_c = np.zeros(self.M)
        self.u = np.zeros(self.d_xi)
        self.kappa_eff = 0.0
        self.aniso = 0.0
        self.particles = np.zeros((self.A, 1, self.d_xi))
        self.w_particles: Optional[np.ndarray] = None
        self.P: Optional[FieldParams] = None
        self.free_rows = np.ones(1, dtype=bool)
        self.xi_anchor = np.zeros((1, self.d_xi))
        self.phi_anchor = np.zeros((1, self.d_phi))
        self.cG = np.zeros((1, self.M))
        self.o_bar = 0.0
        self.T_D = max(1e-12, self.T_per_o)
        self.T_frag = max(1e-15, self.frag_T_per_o)
        self.init_weights: List[Dict[str, float]] = []
        self._prev_dir: Optional[np.ndarray] = None
        self._unit_ideal_cache: Optional[np.ndarray] = None

        # internal iteration counters (never mixed with musical time, audit D1.1)
        self.steps_used = 0                 # unit-level Langevin iterations (warm-start reference)
        self.window_steps_used = 0          # window-level Langevin iterations, current unit
        self.window_steps_total = 0         # ... over the whole job
        self.window_index = 0

        # per-window scoring snapshot (frozen while the trajectory is improved, audit C4)
        self._win_rows: Optional[np.ndarray] = None
        self._win_P: Optional[FieldParams] = None
        self._win_coef: Optional[Dict[str, Any]] = None
        self._win_log: Dict[str, Any] = {}
        self._win_evals = 0
        self._win_first_bd: Optional[Dict[str, float]] = None
        self._win_first_val: Optional[float] = None
        self._win_best_bd: Optional[Dict[str, float]] = None
        self._win_best_val: Optional[float] = None

        self._e_cache: Dict[tuple, tuple] = {}
        self.field_traces: List[Dict[str, Any]] = []
        self.particle_traces: List[Dict[str, Any]] = []
        self.window_traces: List[Dict[str, Any]] = []
        self.relation_traces: List[Dict[str, Any]] = []
        self.realized_traces: List[Dict[str, Any]] = []
        self.committed_summary: Dict[str, Any] = {}

    # -------------------------------------------------------------- fixed triple set T
    def _fixed_triples(self) -> List[Tuple[int, int, int]]:
        allc = list(itertools.combinations(range(self.M), 3))
        if not allc:
            self.warnings.append(f"M={self.M} admits no 3-track combination; the triple term of "
                                 f"eq.(28) degenerates to a single repeated index")
            return [(0, 0, 0)]
        if len(allc) <= self.max_triples:
            return allc
        g = np.random.default_rng(int(self.cfg.get("seed", 0)) + 977)
        sel = g.choice(len(allc), size=self.max_triples, replace=False)
        self.warnings.append(
            f"M={self.M}: {len(allc)} 3-track combinations reduced to a fixed random subset of "
            f"{self.max_triples} (spec §7.1 allows a small fixed set)")
        return [allc[int(k)] for k in np.sort(sel)]

    # -------------------------------------------------------------- M_D = m0 (I + kappa v v^T)
    def _set_M_D(self, v_c: np.ndarray, aniso: Optional[float] = None) -> None:
        """kappa is the configured kappa_M scaled by the anisotropy of the *committed*
        composition-change covariance (1.0 when no covariance is available yet)."""
        v = np.asarray(v_c, dtype=np.float64).ravel()
        n = float(np.linalg.norm(v))
        u = np.zeros(self.d_xi)
        a = 1.0 if aniso is None else float(np.clip(aniso, 0.0, 1.0))
        if n > 1e-9:
            u[self.d_phi:self.d_phi + self.M] = v / n
            self.kappa_eff = self.kappa_M * a
        else:
            self.kappa_eff = 0.0
        self.aniso = a
        self.v_c = v
        self.u = u

    def _apply_M_D(self, g: np.ndarray) -> np.ndarray:
        if self.kappa_eff > 0.0:
            g = g + self.kappa_eff * np.outer(g @ self.u, self.u)
        return self.m_scale * g

    def _apply_L_D(self, e: np.ndarray) -> np.ndarray:
        if self.kappa_eff > 0.0:
            a = np.sqrt(1.0 + self.kappa_eff) - 1.0
            e = e + a * np.outer(e @ self.u, self.u)
        return np.sqrt(self.m_scale) * e

    @staticmethod
    def _cov_anisotropy(cov: np.ndarray) -> float:
        """(lam_max - mean lam) / lam_max of a PSD matrix; 0 for isotropic, ->1 for rank one."""
        c = np.asarray(cov, dtype=np.float64)
        if c.size == 0 or not np.all(np.isfinite(c)):
            return 0.0
        w = np.clip(np.linalg.eigvalsh(0.5 * (c + c.T)), 0.0, None)
        lmax = float(w.max()) if len(w) else 0.0
        if lmax <= 1e-18:
            return 0.0
        return float(np.clip((lmax - float(w.mean())) / lmax, 0.0, 1.0))

    # ------------------------------------------------------------------ field coefficients
    def _coefficients(self, unit: UnitContext, history, rows: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """s_ij / w_ij / d_ij / beta / v from the *committed* history at the moment of the call.

        `rows` restricts the per-time coefficients to a commit window (audit D1.2/D1.4); the
        sign matrix and beta are unit-level.  Called from `begin_unit` (rows = all) and from
        `prepare_reference` (rows = the window), never during the realization of a window.
        """
        M = unit.M
        idx = np.arange(unit.J) if rows is None else np.asarray(rows, dtype=np.int64)
        corr = np.asarray(history.corr(), dtype=np.float64)
        h_c = np.asarray(history.h_c, dtype=np.float64)
        M_H = np.asarray(history.M_H, dtype=np.float64)
        has_hist = bool(history.has_history())

        s = np.array(self.s_prior, dtype=np.float64, copy=True)
        n_sig = 0
        if has_hist:
            sig = np.abs(corr) > self.corr_significant
            np.fill_diagonal(sig, False)
            s = np.where(sig, np.where(corr >= 0.0, 1.0, -1.0), s)
            n_sig = int(np.triu(sig, 1).sum())
        s = 0.5 * (s + s.T)
        s = np.where(s >= 0.0, 1.0, -1.0)
        np.fill_diagonal(s, 1.0)

        w = 0.1 + unit.S[idx] + 0.5 * np.abs(corr)[None, :, :]
        chi = unit.chi[idx]
        arg = (h_c[None, :, None] - s[None, :, :] * h_c[None, None, :]
               + chi[:, :, None] - chi[:, None, :])
        d = 0.25 * np.tanh(arg)
        beta = np.array([0.25 * np.tanh(M_H[i, l] - M_H[j, l]) for (i, j, l) in self.triples])

        v = np.asarray(history.change_direction, dtype=np.float64)
        v_src = "history.change_direction"
        if float(np.linalg.norm(v)) <= 1e-12:
            st = history.mode_state.get(self.state_key) or {}
            cand = (st.get("committed") or {}).get("v") or st.get("v")
            if cand:
                v = np.asarray(cand, dtype=np.float64)
                v_src = "mode_state.v (persisted)"
            else:
                v = np.zeros(M)
                v_src = "none (no committed change yet)"
        aniso = self._cov_anisotropy(getattr(history, "dc_cov", np.zeros((M, M))))
        return {"s": s, "w": w, "d": d, "beta": beta, "v": v, "aniso": aniso,
                "info": {"has_history": has_hist, "significant_history_correlations": n_sig,
                         "history_n_updates": int(getattr(history, "n_updates", 0)),
                         "history_commits": int(getattr(history, "commits", 0)),
                         "h_c": [float(x) for x in h_c],
                         "M_H_trace": float(np.trace(M_H)),
                         "corr_offdiag_absmax": float(np.abs(corr - np.diag(np.diag(corr))).max()),
                         "dc_cov_trace": float(np.trace(np.asarray(getattr(history, "dc_cov", np.zeros((M, M)))))),
                         "change_direction_source": v_src, "dc_cov_anisotropy": float(aniso)}}

    def _coef_on(self, unit: UnitContext, idx: np.ndarray) -> Dict[str, Any]:
        """The coefficients frozen for the current window, restricted to `idx` when `idx` is a
        subset of the window rows (the committed block is); otherwise the unit-level ones."""
        coef, wr = self._win_coef, self._win_rows
        if coef is not None and wr is not None and len(wr):
            if len(wr) == len(idx) and np.array_equal(wr, idx):
                return coef
            pos = np.searchsorted(wr, idx)
            if pos.size and int(pos.max()) < len(wr) and np.array_equal(wr[pos], idx):
                return dict(coef, w=coef["w"][pos], d=coef["d"][pos])
        return {"s": self.s_ij, "w": self.w_ij[idx], "d": self.d_ij[idx], "beta": self.beta,
                "v": self.v_c, "aniso": self.aniso, "info": {"source": "unit-level coefficients"}}

    def _params_for(self, unit: UnitContext, coef: Dict[str, Any], rows: Optional[np.ndarray]) -> FieldParams:
        """FieldParams of eq. (28) on the unit grid (rows=None) or on a commit window."""
        if rows is None:
            idx = np.arange(unit.J)
            ev = self.free_rows
        else:
            idx = np.asarray(rows, dtype=np.int64)
            ev = unit.free_mask[idx]
            if not ev.any():
                ev = np.ones(len(idx), dtype=bool)
        return FieldParams(self.d_phi, unit.M, self.d_xi, self.phi_anchor[idx], self.cG[idx],
                           coef["w"], coef["s"], coef["d"], coef["beta"], self.triples,
                           unit.o[idx], unit.xi_goal[idx], self.Wv, ev,
                           self.lam_phi, self.lam_2, self.lam_3, self.lam_G, self.eps_D)

    # ------------------------------------------------------------------ fragment vocabulary
    def _fragment_ready(self) -> bool:
        """Fragment mode: the analyzer carries the fragment bank AND the engine handed us the PCM."""
        return (getattr(self.analyzer, "frag_f", None) is not None
                and getattr(self, "sources", None) is not None)

    def _anchor_offsets(self) -> np.ndarray:
        """Row offsets (frames) of one fragment composition: `frag_anchor_offsets` analysis hops."""
        return np.arange(self.frag_anchor_offsets, dtype=np.int64) * int(self.analyzer.hop)

    def _goal_exposure_level(self, unit: UnitContext) -> np.ndarray:
        """The goal track's intended level per row under `form.goal_exposure` (0 where it is
        capped).  This mirrors the deterministic schedule the hires realizer builds (cap until
        `goal_rise_seconds` before the goal time, Q5 rise to 1, 1 inside GOAL_HOLD, Q5 descent at
        the head of a REOPEN); it is used only to decide with which goal level the random
        fragment compositions of the anchor sample are drawn, so a coarse match is enough."""
        ge = self.cfg["form"]["goal_exposure"]
        names = np.asarray(unit.phase_names)
        gl = np.zeros(unit.J, dtype=np.float64)
        if str(ge.get("policy", "contract_only")) == "free":
            gl[:] = 1.0                                    # uncapped: the goal may play like a material
        else:
            cap = float(ge.get("open_max", 0.0))
            gl[:] = cap
            gl[names == "INTRO"] = float(ge.get("intro_max", 0.0))
            rise = max(1.0, float(self.cfg["hires"]["goal_rise_seconds"]) * float(unit.fs))
            if unit.goal_arrival is not None:
                s = np.clip((unit.centers - (float(unit.goal_arrival) - rise)) / rise, 0.0, 1.0)
                q = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
                m = unit.centers < float(unit.goal_arrival)
                gl[m] = gl[m] + (1.0 - gl[m]) * q[m]
            re = unit.phase_frames.get("REOPEN")
            if re is not None and unit.index > 0:
                s = np.clip((unit.centers - float(re[0])) / rise, 0.0, 1.0)
                q = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
                m = names == "REOPEN"
                gl[m] = 1.0 + (gl[m] - 1.0) * q[m]
        gl[unit.hold_mask] = 1.0
        return np.clip(gl, 0.0, 1.0)

    def _draw_anchor_sample(self, unit: UnitContext) -> Dict[str, Any]:
        """FRAG_CONTRACT (Diffusion, anchor term): the fixed 0.4 / 0.1 probe mix is replaced by
        the mean of a small sample of RANDOM FRAGMENT COMPOSITIONS drawn once per unit.

        The sample is drawn per goal-exposure group (the goal level is 0 wherever the exposure
        policy caps it, 1 where the goal is exposed, 0.5 in the middle of the rise) with a
        generator seeded from (config seed, unit index) -- so it is reproducible, it does not
        perturb the Langevin noise stream, and the history-intervention probe (which re-runs
        `begin_unit` on a deep copy of the mode) sees exactly the same anchor.
        """
        an = self.analyzer
        offs = self._anchor_offsets()
        gl = self._goal_exposure_level(unit)
        q = np.clip(np.round(gl * 2.0) / 2.0, 0.0, 1.0)              # groups {0, 0.5, 1}
        rng = np.random.default_rng(int(self.cfg.get("seed", 0)) + self.frag_anchor_seed
                                    + 7919 * int(unit.index))
        groups: Dict[float, np.ndarray] = {}
        meta: List[Dict[str, Any]] = []
        for lev in sorted(float(x) for x in np.unique(q)):
            rows = []
            draws = []
            for _k in range(self.frag_anchor_samples):
                xi, _parts, info = an.random_fragment_composition(self.sources, rng, offs, goal_level=lev)
                rows.append(np.asarray(xi, dtype=np.float64))
                draws.append({"positions": [int(v) for v in info["positions"]],
                              "levels": [round(float(v), 6) for v in info["levels"]]})
            X = np.asarray(rows)                                     # (n, n_offsets, d_xi)
            groups[lev] = X
            flat = X.reshape(-1, self.d_xi)
            meta.append({"goal_level": float(lev), "compositions": int(len(rows)),
                         "offsets_per_composition": int(len(offs)),
                         "rows": int(len(flat)),
                         "mean_xi_phi_block": [float(v) for v in flat.mean(axis=0)[:self.d_phi]],
                         "mean_xi_c_block": [float(v) for v in
                                             flat.mean(axis=0)[self.d_phi:self.d_phi + unit.M]],
                         "spread_mean_dist2_to_sample_mean": float(
                             self.analyzer.dist2(flat, flat.mean(axis=0)[None, :]).mean()),
                         "draws": draws})
        self.anchor_groups = groups
        self.anchor_row_level = q
        xi_anchor = np.empty((unit.J, self.d_xi), dtype=np.float64)
        for lev, X in groups.items():
            xi_anchor[q == lev] = X.reshape(-1, self.d_xi).mean(axis=0)
        self.anchor_meta = {
            "source": "mean of random fragment compositions (an.random_fragment_composition)",
            "seed": int(int(self.cfg.get("seed", 0)) + self.frag_anchor_seed + 7919 * int(unit.index)),
            "offsets_frames": [int(v) for v in offs],
            "offsets_seconds": [float(v) / float(unit.fs) for v in offs],
            "goal_level_groups": [float(x) for x in sorted(groups)],
            "rows_per_group": {str(float(lev)): int((q == lev).sum()) for lev in groups},
            "groups": meta,
            "lambda_phi_note": ("lambda_phi is unchanged configuration; what changed is WHERE the "
                                "anchor lives: phi_anchor is no longer the phi block of the fixed "
                                "0.4/0.1 probe mix on the continuous clock but the mean phi of a "
                                "sample of exact fragment compositions, i.e. a point of the same "
                                "reachable set the realizer picks from"),
        }
        return {"xi_anchor": xi_anchor, "meta": self.anchor_meta}

    def _anchor_rows_for(self, level: float, n_rows: int, n_probe: int) -> List[np.ndarray]:
        """`n_probe` random fragment compositions of the anchor sample, each tiled to `n_rows`
        rows, for the "committed vs random" field-energy statistic."""
        if not self.anchor_groups:
            return []
        lev = min(self.anchor_groups, key=lambda x: abs(x - float(level)))
        X = self.anchor_groups[lev]
        k = int(min(n_probe, len(X)))
        take = np.arange(n_rows) % X.shape[1]
        return [X[a][take] for a in range(k)]

    def _fragment_sample_energy(self, P: FieldParams, n_rows: int, level: float) -> Dict[str, Any]:
        """Mean field energy of the unit's random fragment sample evaluated with the SAME field
        parameters as the composition it is compared with."""
        probes = self._anchor_rows_for(level, n_rows, self.frag_energy_probe_samples)
        if not probes:
            return {"mean": None, "std": None, "n": 0}
        e = np.array([field_energy(P, x) for x in probes])
        return {"mean": float(e.mean()), "std": float(e.std()), "n": int(len(e)),
                "min": float(e.min()), "max": float(e.max())}

    def _rep_params(self, unit: UnitContext, coef: Dict[str, Any], row: int, k: int,
                    n_rep: int) -> FieldParams:
        """FieldParams of eq. (28) for ONE grid row `row` (position `k` inside the window's
        coefficient arrays) replicated over `n_rep` independent proposal states, so that the
        row-local gradient of all proposals is obtained with one `field_row_gradient` call."""
        def rep(a):
            return np.repeat(np.asarray(a, dtype=np.float64)[None, ...], n_rep, axis=0)
        return FieldParams(self.d_phi, unit.M, self.d_xi, rep(self.phi_anchor[row]), rep(self.cG[row]),
                           rep(coef["w"][k]), coef["s"], rep(coef["d"][k]), coef["beta"], self.triples,
                           np.full(n_rep, float(unit.o[row])), rep(unit.xi_goal[row]), self.Wv,
                           np.ones(n_rep, dtype=bool), self.lam_phi, self.lam_2, self.lam_3,
                           self.lam_G, self.eps_D)

    def _frag_time_process(self, unit: UnitContext, idx: np.ndarray, coef: Dict[str, Any],
                           xi_current: np.ndarray, n_prop: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """FRAG_CONTRACT (Diffusion): integrate the field as a TIME PROCESS in musical time.

            xi_{k+1} = xi_k - tau M_D grad E_D(xi_k) dt + sqrt(2 tau T dt) L_D eps_k

        over the window, one step per model step (dt = `analysis.model_step_seconds`), starting
        from the realized current composition `xi_current` at the first window row; the rows in
        between are linear interpolations of the model-step knots (the realizer switches levels
        on a 0.25 s ramp and commits every 0.5 s, so a reference that is piecewise linear between
        model-step knots is inside what it can follow).  The `n_prop` proposals are `n_prop`
        INDEPENDENT NOISE DRAWS of the same process from the same initial condition.  GOAL_HOLD
        knots (and rows) are pinned to xi_G.  Openness scales drift and fluctuation row-wise
        exactly as in eq. (29).
        """
        an = self.analyzer
        hop_s = float(an.hop) / float(unit.fs)
        dt_model = max(hop_s, float(self.model_step_seconds))
        stride = max(1, int(round(dt_model / hop_s)))
        n = int(len(idx))
        kpos = np.arange(0, n, stride, dtype=np.int64)
        if int(kpos[-1]) != n - 1:
            kpos = np.append(kpos, n - 1)
        K = int(len(kpos))
        rows_k = idx[kpos]
        hold_k = unit.hold_mask[rows_k]
        goal_k = unit.xi_goal[rows_k]
        o_k = unit.o[rows_k]
        tau = self.frag_tau
        T = self.T_frag
        A = max(1, int(n_prop))

        Z = np.repeat(np.asarray(xi_current, dtype=np.float64).ravel()[None, :], A, axis=0)
        if bool(hold_k[0]):
            Z[:] = goal_k[0]
        X = np.empty((A, K, self.d_xi), dtype=np.float64)
        X[:, 0] = Z
        zero = np.zeros_like(Z)
        disp: List[float] = []
        disp_o: List[float] = []
        dmag: List[float] = []
        nmag: List[float] = []
        acc_drift = np.zeros_like(Z)
        acc_noise = np.zeros_like(Z)
        steps = 0
        clipped = 0
        n_clip_checked = 0
        for k in range(K - 1):
            if self.window_steps_used >= self.window_step_budget_per_unit:
                X[:, k + 1:] = X[:, k][:, None, :]
                self.warnings.append(f"unit {unit.index}: window integration budget "
                                     f"{self.window_step_budget_per_unit} reached")
                break
            dt = float(kpos[k + 1] - kpos[k]) * hop_s
            ok = float(o_k[k])
            P_k = self._rep_params(unit, coef, int(rows_k[k]), int(kpos[k]), A)
            n_sub = self.frag_substeps
            cap = (self.frag_max_step / float(n_sub)) if self.frag_max_step > 0.0 else 0.0
            Zs = Z
            bound = np.zeros(A, dtype=bool)
            for _s in range(n_sub):
                g = field_row_gradient(P_k, Zs)
                sub = (-tau * (dt / n_sub) * ok) * self._apply_M_D(g)
                if cap > 0.0:
                    sn = np.sqrt(np.maximum(an.dist2(sub, zero), 0.0))
                    over = sn > cap
                    if over.any():
                        sub = sub * np.where(over, cap / np.maximum(sn, 1e-12), 1.0)[:, None]
                        bound |= over
                Zs = Zs + sub
            drift = Zs - Z
            clipped += int(bound.sum())
            n_clip_checked += A
            eps = self.rng.standard_normal((A, self.d_xi))
            noise = (float(np.sqrt(2.0 * tau * T * dt)) * ok) * self._apply_L_D(eps)
            Zn = Z + drift + noise
            if bool(hold_k[k + 1]):
                Zn[:] = goal_k[k + 1]
            step_d = np.sqrt(np.maximum(an.dist2(Zn, Z), 0.0))
            disp += [float(v) for v in step_d]
            if ok >= 0.25:                     # o-normalised figure only where openness is meaningful
                disp_o += [float(v) / ok for v in step_d]
            dmag.append(float(np.sqrt(np.maximum(an.dist2(drift, zero), 0.0)).mean()))
            nmag.append(float(np.sqrt(np.maximum(an.dist2(noise, zero), 0.0)).mean()))
            acc_drift += drift
            acc_noise += noise
            Z = Zn
            X[:, k + 1] = Z
            self.window_steps_used += 1
            self.window_steps_total += 1
            steps += 1

        # --- linear interpolation of the model-step knots onto the window rows ---
        t = np.arange(n, dtype=np.float64)
        lo = np.clip(np.searchsorted(kpos, np.arange(n), side="right") - 1, 0, K - 2)
        hi = lo + 1
        span = (kpos[hi] - kpos[lo]).astype(np.float64)
        wgt = np.where(span > 0, (t - kpos[lo]) / np.where(span > 0, span, 1.0), 0.0)[None, :, None]
        Xr = (1.0 - wgt) * X[:, lo, :] + wgt * X[:, hi, :]
        hold = unit.hold_mask[idx]
        if hold.any():
            Xr[:, hold, :] = unit.xi_goal[idx][hold]

        d_mean = float(np.mean(disp)) if disp else 0.0
        dr = float(np.mean(dmag)) if dmag else 0.0
        nz = float(np.mean(nmag)) if nmag else 0.0
        log = {
            "integration": "musical time, forward Euler per model step, rows linearly interpolated",
            "model_step_seconds": float(dt_model),
            "rows_per_model_step": int(stride),
            "model_steps": int(steps),
            "knots": int(K),
            "independent_noise_draws": int(A),
            "initial_condition": "xi_current at the first window row (identical for every proposal)",
            "ideal_displacement_per_model_step": d_mean,
            "ideal_displacement_per_model_step_at_o1": float(np.mean(disp_o)) if disp_o else None,
            "at_o1_knots": int(len(disp_o)),
            "ideal_displacement_per_model_step_max": float(np.max(disp)) if disp else 0.0,
            "ideal_displacement_per_second": d_mean / max(1e-9, dt_model),
            "mean_drift_per_model_step": dr,
            "mean_noise_per_model_step": nz,
            "drift_noise_ratio": float(dr / nz) if nz > 1e-15 else None,
            "net_drift_over_window": float(np.sqrt(np.maximum(an.dist2(acc_drift, zero), 0.0)).mean()),
            "net_noise_over_window": float(np.sqrt(np.maximum(an.dist2(acc_noise, zero), 0.0)).mean()),
            "net_drift_noise_ratio": (float(np.sqrt(np.maximum(an.dist2(acc_drift, zero), 0.0)).mean()
                                            / max(1e-15, np.sqrt(np.maximum(an.dist2(acc_noise, zero), 0.0)).mean()))
                                      if steps else None),
            "mean_openness_on_knots": float(np.mean(o_k)),
            "drift_substeps_per_model_step": int(self.frag_substeps),
            "drift_trust_region": float(self.frag_max_step),
            "drift_trust_region_binding_fraction": (float(clipped) / float(n_clip_checked)
                                                    if n_clip_checked else 0.0),
            "tau_frag_step": float(tau), "temperature_T_frag": float(T),
            "displacement_units": "d_xi metric distance sqrt(analyzer.dist2) between consecutive knots",
        }
        return Xr, log

    # ------------------------------------------------------------------ conditioning (§7.1)
    def begin_unit(self, unit: UnitContext, history) -> None:
        J, M = unit.J, unit.M
        self._e_cache = {}
        self.steps_used = 0
        self.window_steps_used = 0
        self.window_index = 0
        self._win_rows = None
        self._win_P = None
        self._win_coef = None
        self._unit_ideal_cache = None
        self.w_particles = None
        st = history.mode_state.get(self.state_key) or {}

        fm = unit.free_mask
        if not fm.any():
            self.warnings.append(f"unit {unit.index}: no free (non-hold) window; using all windows")
            fm = np.ones(J, dtype=bool)
        self.free_rows = fm

        # --- phi_anchor ----------------------------------------------------
        # baseline: the phi block of the fixed probe mix (goal 0.1, each material 0.4).
        # fragment mode (FRAG_CONTRACT): the mean of a small sample of random fragment
        # compositions drawn once per unit -- the anchor now lives in the fragment space, i.e.
        # inside the set the realizer can actually reach.  lambda_phi is unchanged.
        g_anchor = np.full(M, self.anchor_nongoal)
        g_anchor[0] = self.anchor_goal
        self._frag = self._fragment_ready()
        self.anchor_groups = {}
        self.anchor_meta = {}
        if self._frag:
            drawn = self._draw_anchor_sample(unit)
            xi_anchor = drawn["xi_anchor"]
        else:
            xi_anchor, _ = unit.probe(g_anchor)
        self.xi_anchor = xi_anchor
        self.phi_anchor = xi_anchor[:, :self.d_phi]

        # --- c^G: contribution of the goal-only mix at the same times ------
        self.cG = unit.analyzer.split(unit.xi_goal)[1]

        # --- sign prior: acoustic similarity, or the signs persisted by the previous unit ---
        S_bar = unit.S[fm].mean(axis=0)
        prior = np.where(S_bar >= self.sign_similarity, 1.0, -1.0)
        prev = st.get("s_ij")
        if prev is not None:
            prev = np.asarray(prev, dtype=np.float64)
            if prev.shape == prior.shape:
                prior = prev.copy()
        self.s_prior = prior

        # --- field coefficients from the committed history (unit level) ----
        coef = self._coefficients(unit, history, None)
        self.s_ij = coef["s"]
        self.w_ij = coef["w"]
        self.d_ij = coef["d"]
        self.beta = coef["beta"]
        self.d_bar = self.d_ij[fm].mean(axis=0)
        self.w_bar = self.w_ij[fm].mean(axis=0)
        self._set_M_D(coef["v"], coef["aniso"] if coef["info"]["has_history"] else None)

        # --- field parameter bundle (unit level; frozen for the whole unit) -
        self.P = self._params_for(unit, coef, None)

        # --- temperature and unit-level particles (warm-start reference) ---
        self.o_bar = float(unit.mean_openness)
        self.T_D = max(1e-12, self.T_per_o * self.o_bar)
        # fragment mode: the musical-time process has its own tau / T (the iteration-time pair
        # above still drives `propose()`, i.e. the warm-start reference, unchanged).
        self.T_frag = max(1e-15, self.frag_T_per_o * self.o_bar)
        self._prev_dir = self._previous_direction(unit, st)
        self.particles = self._init_particles(unit, self._prev_dir)
        e0 = [field_energy(self.P, self.particles[a]) for a in range(self.A)]

        self.field_traces.append({
            "unit": int(unit.index),
            "lambda_phi": self.lam_phi, "lambda_2": self.lam_2, "lambda_3": self.lam_3,
            "lambda_G": self.lam_G, "epsilon_D": self.eps_D, "step_tau": self.tau,
            "temperature_T_D": self.T_D, "mean_openness": self.o_bar, "kappa_E": self.kappa_E,
            "anchor_probe_gains": g_anchor.tolist(),
            "anchor_source": ("fragment sample (FRAG_CONTRACT)" if self._frag
                              else "fixed probe mix (baseline)"),
            "anchor_fragment_sample": (dict(self.anchor_meta) if self._frag else None),
            "fragment_mode": bool(self._frag),
            "fragment_time_process": ({"step_tau_frag": self.frag_tau,
                                       "temperature_T_frag": self.T_frag,
                                       "temperature_per_openness": self.frag_T_per_o,
                                       "model_step_seconds": float(self.model_step_seconds),
                                       "note": ("prepare_reference integrates eq.(29) in MUSICAL time "
                                                "(dt = model step) instead of dimensionless iterations; "
                                                "propose() keeps the iteration-time recursion with "
                                                f"step={self.tau}, T={self.T_D}")}
                                      if self._frag else None),
            "phi_anchor_mean": self.phi_anchor[fm].mean(axis=0).tolist(),
            "w_ij_mean": self.w_bar.tolist(),
            "s_ij": self.s_ij.tolist(),
            "significant_history_correlations": coef["info"]["significant_history_correlations"],
            "d_ij_mean": self.d_bar.tolist(), "beta": self.beta.tolist(),
            "triples": [list(map(int, t)) for t in self.triples],
            "M_D": {"kappa": self.kappa_eff, "kappa_M": self.kappa_M,
                    "dc_cov_anisotropy": self.aniso, "isotropic_scale": self.m_scale,
                    "v": self.v_c.tolist(), "u_c_block": self.u[self.d_phi:self.d_phi + M].tolist()},
            "history_used": dict(coef["info"], previous_realized_blocks=bool(st.get("last_blocks"))),
            "initial_particle_field_energy": [float(x) for x in e0],
            "unit_random_fragment_sample_field_energy": (
                self._fragment_sample_energy(self.P, int(unit.J), 0.0) if self._frag else None),
            "particle_initialisation_weights": list(self.init_weights),
            "weights_are_fixed_configuration": ("the five lambda weights are configuration; they are "
                                                "never re-scaled from the observed value share of the "
                                                "terms (audit D1.3)"),
        })

    def _previous_direction(self, unit: UnitContext, st: Dict[str, Any]) -> Optional[np.ndarray]:
        blocks = st.get("last_blocks")
        if not blocks:
            return None
        try:
            bl = {k: np.asarray(v, dtype=np.float64) for k, v in blocks.items()}
            prev = self.objective.blocks_to_unit(unit, bl)
        except Exception as e:                                          # noqa: BLE001
            self.warnings.append(f"unit {unit.index}: could not map previous realized blocks ({e})")
            return None
        return prev - unit.xi_goal

    def _init_particles(self, unit: UnitContext, prev_dir: Optional[np.ndarray]) -> np.ndarray:
        """Goal composition + anchor direction (+ previous realized composition) + small noise."""
        J = unit.J
        anchor_dir = self.xi_anchor - unit.xi_goal
        o = unit.o[:, None]
        sigma = self.init_spread * float(np.sqrt(self.T_D))
        out = np.empty((self.A, J, self.d_xi))
        self.init_weights = []
        for a in range(self.A):
            alpha = float(self.rng.uniform(0.3, 1.0))
            gamma = float(self.rng.uniform(0.0, 0.8)) if prev_dir is not None else 0.0
            x = unit.xi_goal + o * (alpha * anchor_dir)
            if prev_dir is not None:
                x = x + gamma * prev_dir
            x = x + o * sigma * ar1_noise(self.rng, J, self.d_xi, self.rho_t)
            out[a] = fix_hold_rows(unit, x)
            self.init_weights.append({"anchor": alpha, "previous": gamma})
        return out

    def _unit_ideal(self, unit: UnitContext) -> np.ndarray:
        """The unit-level ideal trajectory: the lowest-energy unit particle (audit D1.2, used to
        fill the rows outside a commit window, where only the window rows matter)."""
        if self._unit_ideal_cache is None:
            if self.P is None or len(self.particles[0]) != unit.J:
                self._unit_ideal_cache = unit.xi_goal.copy()
            else:
                e = [field_energy(self.P, self.particles[a]) for a in range(self.A)]
                self._unit_ideal_cache = fix_hold_rows(unit, self.particles[int(np.argmin(e))])
        return self._unit_ideal_cache

    # ------------------------------------------------------------------ joint-structure hints
    def hints(self, unit: UnitContext, history) -> List[tuple]:
        """Strongest material-material couplings: s_ij = +1 -> 'sync', s_ij = -1 -> 'counter'."""
        if self.M < 3:
            return []
        w_bar = self.w_bar
        pairs = [(i, j) for i in range(1, self.M) for j in range(i + 1, self.M)]
        pairs.sort(key=lambda ij: -w_bar[ij[0], ij[1]])
        out = []
        for (i, j) in pairs[: max(0, self.n_hints)]:
            out.append((int(i), int(j), "sync" if self.s_ij[i, j] > 0 else "counter"))
        return out

    # ------------------------------------------------------------------ possibility search (§7.2)
    def _langevin(self, unit: UnitContext, n_steps: int) -> int:
        """Eq. (29) on every unit-level particle; displacement scaled row-wise by openness,
        holds pinned.  Used for the full-unit warm-start reference only."""
        o = unit.o[:, None]
        fm = self.free_rows[:, None]
        rows = self.free_rows
        scale_noise = float(np.sqrt(2.0 * self.tau * self.T_D))
        done = 0
        dsum = nsum = 0.0
        for _ in range(n_steps):
            if self.steps_used >= self.step_budget:
                break
            for a in range(self.A):
                x = self.particles[a]
                g = field_row_gradient(self.P, x)
                drift = o * fm * (-self.tau * self._apply_M_D(g))
                eps = ar1_noise(self.rng, unit.J, self.d_xi, self.rho_t)
                noise = o * fm * (scale_noise * self._apply_L_D(eps))
                dsum += float(np.abs(drift[rows]).mean())
                nsum += float(np.abs(noise[rows]).mean())
                self.particles[a] = fix_hold_rows(unit, x + drift + noise)
            self.steps_used += 1
            done += 1
        if done < n_steps:
            self.warnings.append(f"unit {unit.index}: Langevin step budget {self.step_budget} reached")
        n = max(1, done * self.A)
        self._last_drift = dsum / n
        self._last_noise = nsum / n
        self._unit_ideal_cache = None
        return done

    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        """Full-unit ideal trajectories (the engine freezes one of them as the unit reference R0)."""
        n_steps = max(1, int(round(self.steps_max / (1.0 + round_index))))
        done = self._langevin(unit, n_steps)
        n = max(1, min(int(n_targets), self.A))
        targets: List[Target] = []
        energies = [field_energy(self.P, self.particles[a]) for a in range(self.A)]
        for a in range(n):
            xi_hat = fix_hold_rows(unit, self.particles[a])
            targets.append(Target(
                id=f"diffusion:u{unit.index}:r{round_index}:p{a}", xi_hat=xi_hat,
                meta={"particle": a, "field_energy": float(energies[a]),
                      "scope": "unit", "internal_iterations": int(self.steps_used)}))
        self.particle_traces.append({
            "unit": int(unit.index), "round": int(round_index), "steps_this_round": int(done),
            "steps_total": int(self.steps_used), "temperature_T_D": self.T_D,
            "particle_field_energy": [float(x) for x in energies],
            "mean": float(np.mean(energies)), "std": float(np.std(energies)),
            "mean_abs_drift_per_step": float(getattr(self, "_last_drift", 0.0)),
            "mean_abs_noise_per_step": float(getattr(self, "_last_noise", 0.0)),
            "particle0_energy_terms": field_breakdown(self.P, self.particles[0]),
            "targets_returned": len(targets)})
        return targets

    # ------------------------------------------------------------------ window reference (audit D1.2)
    def _init_window_particles(self, unit: UnitContext, rows: np.ndarray,
                               xi_current: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Window particles: the persisted ideal (unit particle summary) blended with the anchor
        direction and the previous realized direction, continued from the realized current
        composition `xi_current` (row 0 of the window starts there and relaxes away)."""
        idx = np.asarray(rows, dtype=np.int64)
        n = len(idx)
        o = unit.o[idx][:, None]
        goal = unit.xi_goal[idx]
        anchor_dir = (self.xi_anchor - unit.xi_goal)[idx]
        ideal = self._unit_ideal(unit)[idx]
        prev_dir = self._prev_dir[idx] if self._prev_dir is not None else None
        hold = unit.hold_mask[idx]
        hop = unit.analyzer.hop / float(unit.fs)
        tt = np.arange(n) * hop
        decay = np.exp(-tt / max(hop, self.cont_seconds))[:, None]      # 1.0 at row 0
        sigma = self.init_spread * float(np.sqrt(self.T_D))
        xc = np.asarray(xi_current, dtype=np.float64).ravel()
        X = np.empty((self.A, n, self.d_xi))
        weights: List[Dict[str, float]] = []
        for a in range(self.A):
            alpha = float(self.rng.uniform(0.3, 1.0))
            gamma = float(self.rng.uniform(0.0, 0.8)) if prev_dir is not None else 0.0
            mu = float(self.rng.uniform(0.25, 0.75))                    # blend with the persisted ideal
            base = goal + o * (alpha * anchor_dir)
            if prev_dir is not None:
                base = base + gamma * prev_dir
            base = (1.0 - mu) * base + mu * ideal
            x = base + decay * (xc[None, :] - base[0][None, :])         # continuation from the realized now
            x = x + o * sigma * ar1_noise(self.rng, n, self.d_xi, self.rho_t)
            x[hold] = goal[hold]
            X[a] = x
            weights.append({"anchor": alpha, "previous": gamma, "persisted_ideal": mu})
        return X, weights

    def _langevin_window(self, unit: UnitContext, rows: np.ndarray, X: np.ndarray,
                         n_steps: int) -> Tuple[int, float, float]:
        """Eq. (29) restricted to the window rows.  Bounded; iterations are counted separately
        from musical time (audit D1.1)."""
        idx = np.asarray(rows, dtype=np.int64)
        n = len(idx)
        P = self._win_P
        o = unit.o[idx][:, None]
        fm = P.rows[:, None]
        goal = unit.xi_goal[idx]
        hold = unit.hold_mask[idx]
        scale_noise = float(np.sqrt(2.0 * self.tau * self.T_D))
        done = 0
        dsum = nsum = 0.0
        for _ in range(n_steps):
            if self.window_steps_used >= self.window_step_budget_per_unit:
                break
            for a in range(self.A):
                x = X[a]
                g = field_row_gradient(P, x)
                drift = o * fm * (-self.tau * self._apply_M_D(g))
                eps = ar1_noise(self.rng, n, self.d_xi, self.rho_t)
                noise = o * fm * (scale_noise * self._apply_L_D(eps))
                dsum += float(np.abs(drift).mean())
                nsum += float(np.abs(noise).mean())
                y = x + drift + noise
                y[hold] = goal[hold]
                X[a] = y
            self.window_steps_used += 1
            self.window_steps_total += 1
            done += 1
        if done < n_steps:
            self.warnings.append(f"unit {unit.index}: window Langevin budget "
                                 f"{self.window_step_budget_per_unit} reached")
        k = max(1, done * self.A)
        return done, dsum / k, nsum / k

    def prepare_reference(self, unit: UnitContext, history, rows: np.ndarray, xi_current: np.ndarray,
                          n_proposals: int) -> List[Target]:
        """Ideal trajectories for the next commit window (audit D1.2/D1.4).

        The field coefficients are rebuilt from the committed history (h_c, M_H, corr(),
        change_direction, dc_cov) *here*, the particles explore the window by a bounded number
        of Langevin iterations *here*, and everything is then frozen: neither the coefficients
        nor the particles move while the engine improves the gain trajectory against the chosen
        reference.
        """
        idx = np.asarray(rows, dtype=np.int64)
        coef = self._coefficients(unit, history, idx)
        self._win_coef = coef
        self._win_rows = idx
        self._win_P = self._params_for(unit, coef, idx)
        self._set_M_D(coef["v"], coef["aniso"] if coef["info"]["has_history"] else None)

        n = max(1, min(int(n_proposals), self.A))
        if self._frag:
            # ---- fragment mode: the field is integrated as a TIME PROCESS over the window ----
            X, tlog = self._frag_time_process(unit, idx, coef, xi_current, n)
            weights = [{"initial_condition": "xi_current", "noise_draw": int(a)} for a in range(n)]
            e_init = [float(field_energy(self._win_P, np.repeat(
                np.asarray(xi_current, dtype=np.float64).ravel()[None, :], len(idx), axis=0)))]
            done = int(tlog["model_steps"])
            drift = float(tlog["mean_drift_per_model_step"])
            noise = float(tlog["mean_noise_per_model_step"])
        else:
            X, weights = self._init_window_particles(unit, idx, xi_current)
            e_init = [field_energy(self._win_P, X[a]) for a in range(self.A)]
            done, drift, noise = self._langevin_window(unit, idx, X, self.window_steps_max)
            tlog = None
        self.w_particles = X
        energies = np.array([field_energy(self._win_P, X[a]) for a in range(len(X))])
        order = np.arange(len(X)) if self._frag else np.argsort(energies)

        base_full = self._unit_ideal(unit)
        targets: List[Target] = []
        for k in range(min(n, len(X))):
            a = int(order[k])
            xi_hat = np.array(base_full, dtype=np.float64, copy=True)
            xi_hat[idx] = X[a]
            xi_hat = fix_hold_rows(unit, xi_hat)
            targets.append(Target(
                id=f"diffusion:u{unit.index}:w{self.window_index}:p{a}", xi_hat=xi_hat,
                meta={"particle": a, "scope": "window", "window_rows": int(len(idx)),
                      "field_energy": float(energies[a]),
                      "field_energy_terms": field_breakdown(self._win_P, X[a]),
                      "internal_iterations": int(done),
                      "time_process": (dict(tlog) if tlog is not None else None),
                      "rows_outside_window": ("integrated in musical time from xi_current; rows "
                                              "outside the window filled with the unit-level ideal "
                                              "(not scored)" if self._frag else
                                              "filled with the unit-level ideal (not scored)")}))

        t0 = float(unit.seconds[idx[0]])
        self._win_log = {
            "unit": int(unit.index), "window_index": int(self.window_index),
            "musical_time_seconds": t0,
            "window_seconds": [t0, float(unit.seconds[idx[-1]])],
            "window_rows": int(len(idx)), "free_rows": int(self._win_P.rows.sum()),
            "internal_iterations_this_window": int(done),
            "internal_iterations_unit_cumulative": int(self.window_steps_used),
            "particles": int(len(X)),
            "particle_field_energy_initial": [float(x) for x in e_init],
            "particle_field_energy_after": [float(x) for x in energies],
            "particle_field_energy_spread": float(energies.max() - energies.min()),
            "mean_abs_drift_per_iteration": float(drift),
            "mean_abs_noise_per_iteration": float(noise),
            "fragment_mode": bool(self._frag),
            "time_process": (dict(tlog) if tlog is not None else None),
            "initialisation_weights": weights,
            "xi_current_continuation_seconds": self.cont_seconds,
            "xi_current_row0_dist2": float(unit.analyzer.dist2(
                np.asarray(xi_current, dtype=np.float64).ravel()[None, :], X[int(order[0])][0][None, :])[0]),
            "coefficients": coef["info"],
            "proposals": [t.id for t in targets],
        }
        self.window_traces.append(dict(self._win_log))
        self.window_index += 1
        # reset the per-window evaluation record (filled by window_error, read by observe_committed)
        self._win_evals = 0
        self._win_first_bd = None
        self._win_first_val = None
        self._win_best_bd = None
        self._win_best_val = None
        return targets

    # ------------------------------------------------------------------ connection to real gains (§7.3)
    def window_error(self, unit: UnitContext, xi_rows: np.ndarray, rows: np.ndarray,
                     target: Target) -> float:
        """Eq. (30) restricted to the commit window: mean d_xi^2 on the free rows of the window
        against the FROZEN reference + kappa_E * field energy of the realized composition on the
        same rows.  The per-term breakdown (anchor / pair / triple / ridge / goal) is kept for
        the step traces (audit D1.3)."""
        idx = np.asarray(rows, dtype=np.int64)
        P = self._win_P
        if P is None or self._win_rows is None or not np.array_equal(self._win_rows, idx):
            P = self._params_for(unit, self._coef_on(unit, idx), idx)
        m = unit.free_mask[idx]
        d = unit.analyzer.dist2(xi_rows, target.xi_hat[idx])
        fit = float(d[m].mean()) if m.any() else 0.0
        e, _, parts = field_terms(P, xi_rows, want_grad=False)
        ev = _eval_mask(P, len(e))
        e_field = float(e[ev].mean())
        bd = {k: float(v[ev].mean()) for k, v in parts.items()}
        val = fit + self.kappa_E * e_field
        bd["field_energy"] = e_field
        bd["fixed_target_fit"] = fit
        bd["kappa_E_times_field_energy"] = self.kappa_E * e_field
        self._win_evals += 1
        if self._win_first_bd is None:
            self._win_first_bd, self._win_first_val = bd, val
        if self._win_best_val is None or val < self._win_best_val:
            self._win_best_bd, self._win_best_val = bd, val
        return float(val)

    def mode_error(self, unit: UnitContext, cand: Candidate, target: Target) -> float:
        """Eq. (30) on the whole unit: <d_xi^2(realized, ideal)> over free windows + kappa_E *
        field energy of the realized composition.  Legal candidates are never rejected, only
        scored.  The cache is keyed by candidate id *and* the identity of the scored array (a
        modified candidate never reuses a stale energy)."""
        key = (int(cand.id), int(id(cand.xi)))
        hit = self._e_cache.get(key)
        if hit is None or hit[0] is not cand.xi:
            e_field = field_energy(self.P, cand.xi)
            self._e_cache[key] = (cand.xi, e_field)
        else:
            e_field = hit[1]
        return float(unit.mean_dist2(cand.xi, target.xi_hat) + self.kappa_E * e_field)

    # ------------------------------------------------------------------ relation terms (audit C3)
    def relation_terms(self, unit: UnitContext) -> Optional[Dict[str, Any]]:
        """Weights / signs / targets of the soft relation term, taken from the *same* field
        coefficients as eq. (28) -- not a decorative common fluctuation.

        omega = time-mean w_ij over the free rows of the unit, s = s_ij, dstar = time-mean
        d_ij.  Pairs that involve the goal track are given zero weight: its exposure is capped
        at 0 before the goal time, so a relation between the goal and a material cannot be
        realized there (audit C3: no counter constraint where a track is pinned at 0).
        """
        omega = np.array(self.w_bar, dtype=np.float64, copy=True)
        np.fill_diagonal(omega, 0.0)
        capped = self.cfg["form"]["goal_exposure"]["policy"] != "free"
        if capped and unit.M > 2:
            omega[0, :] = 0.0
            omega[:, 0] = 0.0
        terms = {"omega": omega, "s": np.array(self.s_ij, dtype=np.float64, copy=True),
                 "dstar": np.array(self.d_bar, dtype=np.float64, copy=True),
                 "lag_seconds": float(self.relation_lag_seconds)}
        iu = unit.analyzer.iu
        self.relation_traces.append({
            "unit": int(unit.index), "lag_seconds": float(self.relation_lag_seconds),
            "omega_upper": [float(x) for x in omega[iu]],
            "s_upper": [float(x) for x in self.s_ij[iu]],
            "dstar_upper": [float(x) for x in self.d_bar[iu]],
            "goal_pairs_zero_weight": bool(capped and unit.M > 2),
            "source": "eq.(28) coefficients w_ij (time mean), s_ij, d_ij (time mean)",
            "note": ("d_ij is the eq.(28) coefficient for the *level* difference r_i - s_ij r_j; "
                     "the engine uses it as the target of the lagged contribution change, so the "
                     "residual is not expected to reach zero -- the unreachable part stays in the "
                     "residual and is never forced by breaking a gain constraint (audit D1.5)"),
        })
        return terms

    # ------------------------------------------------------------------ committed learning (audit D1.4)
    def observe_committed(self, unit: UnitContext, history, rows: np.ndarray, xi_rows: np.ndarray,
                          parts_rows: Dict[str, np.ndarray], reference: Optional[Target],
                          stats: Dict[str, Any]) -> Dict[str, Any]:
        """The mode's own learning from the *committed* composition: the composition-change
        direction v (hence M_D) is re-estimated from the committed rows, and a small summary is
        persisted for the next `prepare_reference`.  No target adaptation of any kind."""
        idx = np.asarray(rows, dtype=np.int64)
        Pc = self._params_for(unit, self._coef_on(unit, idx), idx)
        e, _, parts = field_terms(Pc, xi_rows, want_grad=False)
        ev = _eval_mask(Pc, len(e))
        e_committed = float(e[ev].mean())
        bd_committed = {k: float(v[ev].mean()) for k, v in parts.items()}

        # --- v / M_D from the committed composition only -------------------
        v = np.asarray(history.change_direction, dtype=np.float64)
        v_src = "history.change_direction (committed rows, engine-updated)"
        if float(np.linalg.norm(v)) <= 1e-12:
            c = np.asarray(parts_rows["c"], dtype=np.float64)
            if len(c) > 2:
                dc = np.diff(c, axis=0)
                cov = dc.T @ dc / float(len(dc))
                _, V = np.linalg.eigh(cov)
                v = V[:, -1]
                if float(v @ (c[-1] - c[0])) < 0.0:
                    v = -v
                v_src = "principal direction of the committed c rows"
            else:
                v = np.asarray(self.v_c, dtype=np.float64)
                v_src = "kept (committed block too short)"
        aniso = self._cov_anisotropy(getattr(history, "dc_cov", np.zeros((unit.M, unit.M))))
        self._set_M_D(v, aniso if history.has_history() else None)

        # --- FRAG_CONTRACT statistic: field energy of the committed block vs. the mean field
        #     energy of the unit's random fragment sample, on the SAME rows / field parameters --
        frag_stat: Dict[str, Any] = {"available": False}
        if self._frag:
            lev = float(np.median(self.anchor_row_level[idx])) if len(self.anchor_row_level) == unit.J else 0.0
            rnd = self._fragment_sample_energy(Pc, int(len(idx)), lev)
            m = rnd.get("mean")
            frag_stat = {
                "available": m is not None,
                "committed_field_energy": e_committed,
                "random_fragment_sample_field_energy_mean": m,
                "random_fragment_sample_field_energy_std": rnd.get("std"),
                "random_fragment_sample_n": rnd.get("n"),
                "goal_level_group": lev,
                "difference_committed_minus_random": (None if m is None else float(e_committed - m)),
                "ratio_committed_over_random": (None if not m else float(e_committed / m)),
            }
            self.frag_stats.append(dict(
                frag_stat, unit=int(unit.index), window_index=int(max(0, self.window_index - 1)),
                ideal_displacement_per_model_step=float(
                    (self._win_log.get("time_process") or {}).get("ideal_displacement_per_model_step", 0.0)),
                ideal_displacement_per_model_step_at_o1=(
                    (self._win_log.get("time_process") or {}).get("ideal_displacement_per_model_step_at_o1")),
                drift_noise_ratio=((self._win_log.get("time_process") or {}).get("drift_noise_ratio")),
                mean_drift_per_model_step=float(
                    (self._win_log.get("time_process") or {}).get("mean_drift_per_model_step", 0.0)),
                mean_noise_per_model_step=float(
                    (self._win_log.get("time_process") or {}).get("mean_noise_per_model_step", 0.0)),
                net_drift_noise_ratio=((self._win_log.get("time_process") or {}).get("net_drift_noise_ratio"))))

        c_rows = np.asarray(parts_rows["c"], dtype=np.float64)
        dc_block = float(np.abs(c_rows[-1] - c_rows[0]).mean()) if len(c_rows) > 1 else 0.0
        summary = {
            "unit": int(unit.index), "window_index": int(max(0, self.window_index - 1)),
            "v": [float(x) for x in np.asarray(self.v_c).ravel()],
            "v_norm": float(np.linalg.norm(self.v_c)), "v_source": v_src,
            "M_D_kappa": float(self.kappa_eff), "dc_cov_anisotropy": float(self.aniso),
            "field_energy_committed": e_committed,
            "field_energy_vs_random_fragments": frag_stat,
            "committed_rows": int(len(idx)),
            "committed_mean_abs_dc": dc_block,
            "history_commits": int(getattr(history, "commits", 0)),
        }
        self.committed_summary = summary
        st = history.mode_state.get(self.state_key)
        if not isinstance(st, dict):
            st = {}
        st["committed"] = summary
        history.mode_state[self.state_key] = st

        # --- step trace: field terms and internal iterations vs musical time
        rst = (stats or {}).get("refinement") or {}
        ini = rst.get("initial") or {}
        fin = rst.get("final") or {}
        anchor = bd_committed["anchor"]
        pair = bd_committed["pair"]
        self.step_traces.append({
            "unit": int(unit.index),
            "window_index": int(max(0, self.window_index - 1)),
            "musical_time_seconds": float(unit.seconds[idx[0]]),
            "committed_seconds": float(len(idx)) * unit.analyzer.hop / float(unit.fs),
            "reference_id": None if reference is None else reference.id,
            "reference_hash": (stats or {}).get("reference_hash"),
            "reference_moved_during_realization": False,
            "field_coefficients_frozen_during_realization": True,
            "internal_iterations": {
                "window_langevin_iterations": int(self._win_log.get("internal_iterations_this_window", 0)),
                "window_langevin_iterations_unit_cumulative": int(self.window_steps_used),
                "window_langevin_iterations_job_total": int(self.window_steps_total),
                "unit_langevin_iterations": int(self.steps_used),
                "window_error_evaluations": int(self._win_evals),
                "particles": int(self.A),
            },
            "field_term_breakdown": {
                "initial_unimproved_tail": self._win_first_bd,
                "best_evaluated_tail": self._win_best_bd,
                "committed": dict(bd_committed, field_energy=e_committed,
                                  kappa_E_times_field_energy=self.kappa_E * e_committed),
            },
            "anchor_vs_pair": {"anchor": anchor, "pair": pair, "triple": bd_committed["triple"],
                               "ridge": bd_committed["ridge"], "goal": bd_committed["goal"],
                               "anchor_share": float(anchor / e_committed) if abs(e_committed) > 1e-15 else None,
                               "pair_share": float(pair / e_committed) if abs(e_committed) > 1e-15 else None,
                               "weights_auto_inverted_from_share": False},
            "joint_objective": {"initial": ini.get("J"), "final": fin.get("J"),
                                "initial_fixed_target_error": ini.get("fit"),
                                "final_fixed_target_error": fin.get("fit"),
                                "evaluations": rst.get("evaluations"),
                                "accepted": rst.get("accepted")},
            "committed_mean_abs_dc": dc_block,
            "ideal_time_process": (dict(self._win_log.get("time_process") or {}) if self._frag else None),
            "field_energy_vs_random_fragments": frag_stat,
            "v_norm": summary["v_norm"], "M_D_kappa": float(self.kappa_eff),
        })
        return {"v_norm": summary["v_norm"], "v_source": v_src,
                "field_energy_committed": e_committed,
                "field_energy_anchor": anchor, "field_energy_pair": pair,
                "M_D_kappa": float(self.kappa_eff), "dc_cov_anisotropy": float(self.aniso),
                "window_langevin_iterations": int(self._win_log.get("internal_iterations_this_window", 0)),
                "window_error_evaluations": int(self._win_evals),
                "committed_mean_abs_dc": dc_block,
                "ideal_displacement_per_model_step": float(
                    (self._win_log.get("time_process") or {}).get("ideal_displacement_per_model_step", 0.0))
                if self._frag else None,
                "drift_noise_ratio": ((self._win_log.get("time_process") or {}).get("drift_noise_ratio")
                                      if self._frag else None),
                "field_energy_vs_random_fragments": frag_stat}

    # ------------------------------------------------------------------ deprecated hook
    def update(self, unit: UnitContext, history, realizations: Sequence[Realization],
               round_index: int) -> Dict[str, Any]:
        """Removed (audit B5/D1.1): the engine does not call this, and the mode no longer moves
        particles toward realized candidates.  Kept as an explicit no-op so that any caller sees
        that the realized pull is disabled rather than silently missing."""
        return {"disabled": "realized_pull", "configured_eta": self.pull_eta_disabled,
                "reason": "a reference must not move while a trajectory is scored against it "
                          "(audit B5/C4/D1.1); exploration happens in prepare_reference"}

    # ------------------------------------------------------------------ history residue (§7.4)
    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        e_chosen = field_energy(self.P, chosen.candidate.xi)
        e_part = [field_energy(self.P, self.particles[a]) for a in range(self.A)]
        blocks = self.objective.phase_blocks(unit, chosen.candidate.xi)
        st = {
            "unit": int(unit.index),
            "s_ij": self.s_ij.tolist(),
            "d_ij_mean": self.d_bar.tolist(),
            "w_ij_mean": self.w_bar.tolist(),
            "beta": [float(x) for x in self.beta],
            "triples": [list(map(int, t)) for t in self.triples],
            "last_blocks": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in blocks.items()},
            "particle_summary": {"field_energy_mean": float(np.mean(e_part)),
                                 "field_energy_std": float(np.std(e_part)),
                                 "n_particles": int(self.A),
                                 "unit_langevin_iterations": int(self.steps_used),
                                 "window_langevin_iterations": int(self.window_steps_used),
                                 "windows_prepared": int(self.window_index)},
            "committed": dict(self.committed_summary),
            "v": [float(x) for x in np.asarray(self.v_c).ravel()],
            "M_D_kappa": float(self.kappa_eff),
            "dc_cov_anisotropy": float(self.aniso),
            "chosen_field_energy": float(e_chosen),
        }
        history.mode_state[self.state_key] = st
        fm = unit.free_mask
        resid = chosen.candidate.xi - chosen.target.xi_hat
        c_resid = unit.analyzer.split(resid)[1]
        self.realized_traces.append({
            "unit": int(unit.index), "chosen_target": chosen.target.id,
            "mode_error": float(chosen.mode_error),
            "normalized_mode_error": float(chosen.normalized_mode_error),
            "ideal_vs_realized_mean_dist2": float(unit.mean_dist2(chosen.candidate.xi, chosen.target.xi_hat)),
            "realized_field_energy": float(e_chosen),
            "realized_field_energy_terms": field_breakdown(self.P, chosen.candidate.xi),
            "kappa_E_times_field_energy": float(self.kappa_E * e_chosen),
            "target_field_energy": float(chosen.target.meta.get("field_energy", float("nan"))),
            "unreachable_residual": {
                "mean_abs_contribution_residual": float(np.abs(c_resid[fm]).mean()) if fm.any() else 0.0,
                "max_abs_contribution_residual": float(np.abs(c_resid[fm]).max()) if fm.any() else 0.0,
                "negative_ideal_contribution_rows": int(np.sum(
                    (unit.analyzer.split(chosen.target.xi_hat)[1] < 0.0).any(axis=1) & fm)),
                "note": "kept as residual (audit D1.5); never realized by violating a gain constraint",
            },
            "alternatives_realized_field_energy_mean": float(np.mean(
                [field_energy(self.P, a.candidate.xi) for a in alternatives[:6]])) if alternatives else None,
        })
        self.unit_traces.append({
            "unit": int(unit.index), "chosen_target": chosen.target.id,
            "normalized_mode_error": float(chosen.normalized_mode_error),
            "field_energy_realized": float(e_chosen),
            "field_energy_particles_mean": float(np.mean(e_part)),
            "unit_langevin_iterations": int(self.steps_used),
            "window_langevin_iterations": int(self.window_steps_used),
            "windows_prepared": int(self.window_index),
            "temperature_T_D": self.T_D,
            "fragment_mode": bool(self._frag),
            "fragment_statistics": self._fragment_unit_summary(int(unit.index)),
            "M_D_kappa": float(self.kappa_eff)})

    # ------------------------------------------------------------------ signature / trace
    def _fragment_unit_summary(self, unit_index: int) -> Optional[Dict[str, Any]]:
        """Per-unit summary of the fragment-mode statistics (FRAG_CONTRACT: mean per-step
        displacement of the ideal, drift/noise ratio, committed field energy vs. random
        fragment compositions)."""
        rows = [s for s in self.frag_stats if int(s.get("unit", -1)) == int(unit_index)]
        if not rows:
            return None

        def mean(key):
            v = [float(s[key]) for s in rows if s.get(key) is not None]
            return float(np.mean(v)) if v else None

        rat = [float(s["ratio_committed_over_random"]) for s in rows
               if s.get("ratio_committed_over_random") is not None]
        below = [s for s in rows if s.get("difference_committed_minus_random") is not None
                 and float(s["difference_committed_minus_random"]) < 0.0]
        return {
            "unit": int(unit_index), "commit_steps": int(len(rows)),
            "ideal_mean_displacement_per_model_step": mean("ideal_displacement_per_model_step"),
            "ideal_mean_displacement_per_model_step_at_o1": mean("ideal_displacement_per_model_step_at_o1"),
            "mean_drift_per_model_step": mean("mean_drift_per_model_step"),
            "mean_noise_per_model_step": mean("mean_noise_per_model_step"),
            "drift_noise_ratio_mean": mean("drift_noise_ratio"),
            "net_drift_noise_ratio_mean": mean("net_drift_noise_ratio"),
            "committed_field_energy_mean": mean("committed_field_energy"),
            "random_fragment_field_energy_mean": mean("random_fragment_sample_field_energy_mean"),
            "committed_minus_random_mean": mean("difference_committed_minus_random"),
            "committed_over_random_mean": float(np.mean(rat)) if rat else None,
            "steps_with_committed_below_random": int(len(below)),
            "units": "displacements are d_xi metric distances (sqrt of analyzer.dist2) per model step",
        }

    def signature(self) -> np.ndarray:
        fm = self.free_rows
        parts = [self.s_ij.ravel(), self.d_bar.ravel(), np.asarray(self.beta, dtype=np.float64),
                 self.m_scale * (1.0 + self.kappa_eff * (self.u ** 2)),
                 self.kappa_eff * np.asarray(self.v_c).ravel(),
                 np.array([self.kappa_eff, float(np.linalg.norm(self.v_c))])]
        for a in range(self.A):
            x = self.particles[a]
            parts.append(x[fm].mean(axis=0) if fm.any() and len(fm) == len(x) else x.mean(axis=0))
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])

    def trace(self) -> Dict[str, Any]:
        return {
            "field_parameters": self.field_traces,
            "particle_summary": self.particle_traces,
            "window_reference_preparation": self.window_traces,
            "relation_terms": self.relation_traces,
            "realized_mode_error": self.realized_traces,
            "steps": self.step_traces,
            "fragment_statistics_per_step": list(self.frag_stats),
            "fragment_statistics_per_unit": [
                s for s in (self._fragment_unit_summary(u) for u in
                            sorted({int(x.get("unit", -1)) for x in self.frag_stats}))
                if s is not None],
            "field_term_breakdown_per_step": [
                {"unit": s["unit"], "window_index": s["window_index"],
                 "musical_time_seconds": s["musical_time_seconds"],
                 "initial_unimproved_tail": s["field_term_breakdown"]["initial_unimproved_tail"],
                 "best_evaluated_tail": s["field_term_breakdown"]["best_evaluated_tail"],
                 "committed": s["field_term_breakdown"]["committed"],
                 "anchor_vs_pair": s["anchor_vs_pair"]}
                for s in self.step_traces],
            "internal_iterations_per_step": [
                {"unit": s["unit"], "window_index": s["window_index"],
                 "musical_time_seconds": s["musical_time_seconds"],
                 "committed_seconds": s["committed_seconds"], **s["internal_iterations"]}
                for s in self.step_traces],
            "musical_time_note": (
                "internal iterations and musical time are different axes and are never mixed. "
                "`*_langevin_iterations` count dimensionless iterations of eq.(29) (step tau, AR(1) "
                "coefficient time_correlation per ITERATION); `musical_time_seconds`, "
                "`committed_seconds` (realization.commit_seconds) and the lookahead window "
                "(realization.lookahead_seconds) are seconds of the piece. One commit step = one "
                "reference preparation (window_internal_steps_max iterations on A particles) + a "
                "bounded number of acoustic evaluations of the gain trajectory; no Langevin "
                "iteration is performed while a reference is frozen."),
            "audit_D1": {
                "realized_pull_of_particles": "removed from every realization path "
                                              f"(configured realized_pull={self.pull_eta_disabled} is ignored)",
                "update_hook": "no-op; not called by the engine",
                "exploration_location": "prepare_reference (before the reference is frozen)",
                "coefficients_source": "committed history (h_c, M_H, corr(), change_direction, dc_cov) "
                                       "read at prepare_reference time, then frozen for the window",
                "term_weights": "fixed configuration lambdas; never inverted from the observed share",
                "relation_terms": "supplied from the same eq.(28) coefficients (audit C3)",
                "unreachable_targets": "recorded as residual, never forced (audit D1.5)",
            },
            "fragment_revision": {
                "active": bool(self._frag),
                "detection": ("analyzer.frag_f is not None AND mode.sources is set by the engine "
                              "(docs/FRAG_CONTRACT.md)"),
                "reference": ("time process in musical time over the window rows (see "
                              "window_reference_preparation[*].time_process); n proposals = n "
                              "independent noise draws from the same initial condition xi_current"),
                "anchor": ("mean of random fragment compositions drawn once per unit per "
                           "goal-exposure group; lambda_phi unchanged, but the anchor now lives in "
                           "the fragment space (see field_parameters[*].anchor_fragment_sample)"),
                "pair_and_triple_terms": "unchanged",
                "window_error": "unchanged (eq. 30 on the window rows)",
                "relation_terms": "unchanged",
                "propose": "unchanged (baseline warm start; iteration-time recursion)",
                "step_tau": float(self.frag_tau),
                "temperature_per_openness": float(self.frag_T_per_o),
                "drift_substeps": int(self.frag_substeps),
                "trust_region_per_model_step": float(self.frag_max_step),
                "anchor_samples_per_group": int(self.frag_anchor_samples),
                "anchor_offsets": int(self.frag_anchor_offsets),
                "inherited_vs_reinitialised": (
                    "inherited across units: s_ij prior, last realized blocks, v / M_D and the "
                    "dc_cov anisotropy (history.mode_state['diffusion']); re-initialised per unit: "
                    "the random fragment anchor sample (own seeded generator), phi_anchor, the "
                    "unit particles and both internal iteration counters. Re-initialised per "
                    "window: the field coefficients (from the committed history) and the ideal "
                    "path, which always restarts from the realized xi_current -- no internal "
                    "coordinate system (basis) is carried in this mode"),
                "internal_iterations_vs_musical_time": (
                    "the fragment reference performs NO dimensionless iterations: its steps ARE "
                    "model steps (window_langevin_iterations counts them, "
                    "time_process.model_steps x frag_drift_substeps gradient evaluations each). "
                    "propose() still counts dimensionless iterations in unit_langevin_iterations"),
            },
            "equations": {
                "field_energy": "eq.(28)", "langevin": "eq.(29)", "connection": "eq.(30)",
                "window_error": "eq.(30) restricted to the commit window rows",
                "note": ("finite-step search approximation with a variable openness coefficient; "
                         "not an exact Gibbs sample and not a trained DDPM (spec §7.2)"),
                "drift_convention": ("the drift uses the row-local gradient d e_t / d xi_t, i.e. the "
                                     "gradient of J * E_D; the factor J is absorbed into the positive "
                                     "definite search matrix so tau does not depend on the window count"),
                "drift_noise_balance": ("iteration-time path (propose / non-fragment prepare_reference): "
                                        "at the default tau=step and T_D=temperature_per_openness*o_bar "
                                        "the per-step drift is a small fraction of the per-step "
                                        "fluctuation (see particle_summary.mean_abs_drift_per_step vs "
                                        "mean_abs_noise_per_step): the finite search stays in the "
                                        "transient regime near its initialisation, as spec §7.2 permits. "
                                        "FRAGMENT MODE is deliberately the opposite: frag_step / "
                                        "frag_temperature_per_openness are tuned so that the drift is at "
                                        "least as large as the fluctuation per model step and much larger "
                                        "once accumulated over a window (see fragment_statistics_per_unit: "
                                        "drift_noise_ratio_mean and net_drift_noise_ratio_mean)"),
                "fragment_time_process": ("fragment mode integrates eq.(29) as dxi = -tau M_D grad E_D dt + "
                                          "sqrt(2 tau T dt) L_D dW in MUSICAL time, one step per "
                                          "analysis.model_step_seconds over the window rows, from xi_current; "
                                          "rows between the model-step knots are linear interpolations. The "
                                          "drift of one model step is sub-stepped (frag_drift_substeps) with a "
                                          "per-sub-step trust region for explicit-Euler stability; the "
                                          "fluctuation is one Wiener increment per model step"),
            },
            "warnings": list(self.warnings),
            "units": self.unit_traces,
        }


# ---------------------------------------------------------------------------- self-test
def _random_params(seed: int = 5):
    rng = np.random.default_rng(seed)
    J, M, d_phi = 9, 4, 3
    n_pairs = M * (M - 1) // 2
    d_xi = d_phi + M + n_pairs
    triples = list(itertools.combinations(range(M), 3))
    s = np.where(rng.random((M, M)) > 0.5, 1.0, -1.0)
    s = np.where(0.5 * (s + s.T) >= 0.0, 1.0, -1.0)
    w = 0.1 + rng.random((J, M, M))
    w = 0.5 * (w + w.transpose(0, 2, 1))
    P = FieldParams(d_phi, M, d_xi,
                    phi_anchor=rng.normal(size=(J, d_phi)),
                    cG=rng.random((J, M)),
                    w=w, s=s, d=0.25 * np.tanh(rng.normal(size=(J, M, M))),
                    beta=0.25 * np.tanh(rng.normal(size=len(triples))),
                    triples=triples, o=rng.random(J), xi_goal=rng.normal(size=(J, d_xi)),
                    W=0.05 + rng.random(d_xi), rows=(rng.random(J) > 0.25),
                    lam_phi=1.0, lam_2=1.0, lam_3=0.25, lam_G=1.0, eps_D=1e-4)
    if not P.rows.any():
        P.rows[0] = True
    return P, rng.normal(size=(J, d_xi))


def _self_test(seed: int = 5):
    """Finite-difference check of field_row_gradient against field_energy."""
    P, xi = _random_params(seed)
    J, d_xi = xi.shape
    g_analytic = field_row_gradient(P, xi) / P.n_free
    h = 1e-5
    fd = np.zeros_like(xi)
    for t in range(J):
        for k in range(d_xi):
            xp = xi.copy(); xp[t, k] += h
            xm = xi.copy(); xm[t, k] -= h
            fd[t, k] = (field_energy(P, xp) - field_energy(P, xm)) / (2.0 * h)
    scale = float(np.abs(g_analytic).max())
    abs_err = float(np.abs(fd - g_analytic).max())
    # relative error where the component is not numerically negligible
    big = np.abs(g_analytic) > 1e-4 * scale
    rel_err = float((np.abs(fd - g_analytic)[big] / np.abs(g_analytic)[big]).max()) if big.any() else 0.0
    return rel_err, abs_err / max(scale, 1e-12)


def _descent_test(seed: int = 5, n: int = 40, tau: float = 0.5):
    """Zero-temperature check: the drift direction of eq. (29) must decrease E_D monotonically."""
    P, xi = _random_params(seed)
    es = [field_energy(P, xi)]
    for _ in range(n):
        xi = xi - tau * field_row_gradient(P, xi)
        es.append(field_energy(P, xi))
    rises = sum(1 for a, b in zip(es[:-1], es[1:]) if b > a + 1e-12)
    return es[0], es[-1], rises


if __name__ == "__main__":  # pragma: no cover
    seeds = (1, 2, 3, 5, 8)
    res = [_self_test(sd) for sd in seeds]
    print("field gradient finite-difference check (central differences, h = 1e-5)")
    for sd, (r, a) in zip(seeds, res):
        print(f"  seed {sd}: max relative error = {r:.3e}   max abs error / max|grad| = {a:.3e}")
    wr = max(r for r, _ in res)
    wa = max(a for _, a in res)
    print(f"  worst over seeds: relative = {wr:.3e}, scaled absolute = {wa:.3e}")
    assert wr < 1e-6 and wa < 1e-8, "gradient does not match finite differences"
    print("zero-temperature drift check (40 steps of -tau * grad, tau = 0.5)")
    bad = 0
    for sd in seeds:
        e0, e1, rises = _descent_test(sd)
        print(f"  seed {sd}: E_D {e0:.6f} -> {e1:.6f}   non-decreasing steps = {rises}")
        bad += rises
    assert bad == 0, "the drift direction does not descend the field energy"
    print("OK")

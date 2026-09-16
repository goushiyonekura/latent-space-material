"""Diffusion-type controller (spec §7): a *coupling field* over acoustic-composition space.

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
c block), O = diag(openness) applied row-wise, AR(1)-correlated eps in time, and GOAL_HOLD
rows pinned to xi_G and removed from the free coordinates.  This is a *finite* search: it is
not an exact Gibbs sample and not a trained DDPM.

Real gains are connected through eq. (30)

  E_D(gamma; Xi_hat_a) = < d_xi^2( xi_t(gamma), xi_hat_{a,t} ) > + kappa_E E_D(Xi(gamma))  (30)

so the *realized* composition of a legal candidate is what the field scores; candidates are
never constrained to the field's level set and are never rejected for a large acoustic error.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, ar1_noise, fix_hold_rows


# ---------------------------------------------------------------------------- field parameters
class FieldParams:
    """All coefficients of eq. (28) for one planning unit (or for the self-test)."""

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


def field_energy(P: FieldParams, xi: np.ndarray) -> float:
    """<.> of eq. (28): the mean of the row energy over the evaluation (non-hold) rows."""
    e, _, _ = field_terms(P, xi, want_grad=False)
    return float(e[P.rows].mean()) if P.rows.any() else float(e.mean())


def field_breakdown(P: FieldParams, xi: np.ndarray) -> Dict[str, float]:
    """The five terms of eq. (28), each averaged over the evaluation rows."""
    _, _, parts = field_terms(P, xi, want_grad=False)
    m = P.rows if P.rows.any() else np.ones(len(xi), dtype=bool)
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
        self.pull_eta = float(p.get("realized_pull", 0.5))
        self.max_triples = int(p.get("max_triples", 10))
        self.n_hints = int(p.get("hint_pairs", 3))
        # isotropic part of the positive definite search matrix: M_D = m0 (I + kappa v v^T).
        # m0 = 1 reproduces the plain spec form; raising it lengthens the relaxation the finite
        # search can cover (drift grows like m0, fluctuation like sqrt(m0)).  Default: literal.
        self.m_scale = max(1e-9, float(p.get("search_matrix_scale", 1.0)))

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
        self.d_bar = np.zeros((self.M, self.M))
        self.beta = np.zeros(self.n_T)
        self.v_c = np.zeros(self.M)
        self.u = np.zeros(self.d_xi)
        self.kappa_eff = 0.0
        self.particles = np.zeros((self.A, 1, self.d_xi))
        self.P: Optional[FieldParams] = None
        self.free_rows = np.ones(1, dtype=bool)
        self.steps_used = 0
        self._e_cache: Dict[int, float] = {}
        self.field_traces: List[Dict[str, Any]] = []
        self.particle_traces: List[Dict[str, Any]] = []
        self.realized_traces: List[Dict[str, Any]] = []

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

    # -------------------------------------------------------------- M_D = I + kappa v v^T
    def _set_M_D(self, v_c: np.ndarray) -> None:
        v = np.asarray(v_c, dtype=np.float64).ravel()
        n = float(np.linalg.norm(v))
        u = np.zeros(self.d_xi)
        if n > 1e-9:
            u[self.d_phi:self.d_phi + self.M] = v / n
            self.kappa_eff = self.kappa_M
        else:
            self.kappa_eff = 0.0
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

    # ------------------------------------------------------------------ conditioning (§7.1)
    def begin_unit(self, unit: UnitContext, history) -> None:
        J, M = unit.J, unit.M
        self._e_cache = {}
        self.steps_used = 0
        st = history.mode_state.get(self.state_key) or {}

        fm = unit.free_mask
        if not fm.any():
            self.warnings.append(f"unit {unit.index}: no free (non-hold) window; using all windows")
            fm = np.ones(J, dtype=bool)
        self.free_rows = fm

        # --- history reads -------------------------------------------------
        corr = np.asarray(history.corr(), dtype=np.float64)
        h_c = np.asarray(history.h_c, dtype=np.float64)
        M_H = np.asarray(history.M_H, dtype=np.float64)
        has_hist = bool(history.has_history())
        chg = np.asarray(history.change_direction, dtype=np.float64)
        if float(np.linalg.norm(chg)) <= 1e-12 and st.get("v"):
            chg = np.asarray(st["v"], dtype=np.float64)

        # --- phi_anchor: probe mix (goal 0.1, each material 0.4), analysis material only ---
        g_anchor = np.full(M, self.anchor_nongoal)
        g_anchor[0] = self.anchor_goal
        xi_anchor, _ = unit.probe(g_anchor)
        self.xi_anchor = xi_anchor
        self.phi_anchor = xi_anchor[:, :self.d_phi]

        # --- c^G: contribution of the goal-only mix at the same times ------
        self.cG = unit.analyzer.split(unit.xi_goal)[1]

        # --- s_ij ----------------------------------------------------------
        S_bar = unit.S[fm].mean(axis=0)
        s = np.where(S_bar >= self.sign_similarity, 1.0, -1.0)
        if st.get("s_ij") is not None:
            prev = np.asarray(st["s_ij"], dtype=np.float64)
            if prev.shape == s.shape:
                s = prev.copy()
        n_sig = 0
        if has_hist:
            sig = np.abs(corr) > self.corr_significant
            np.fill_diagonal(sig, False)
            signs = np.where(corr >= 0.0, 1.0, -1.0)
            s = np.where(sig, signs, s)
            n_sig = int(np.triu(sig, 1).sum())
        s = 0.5 * (s + s.T)
        s = np.where(s >= 0.0, 1.0, -1.0)
        np.fill_diagonal(s, 1.0)
        self.s_ij = s

        # --- w_ij, d_ij, beta_ijl -----------------------------------------
        self.w_ij = 0.1 + unit.S + 0.5 * np.abs(corr)[None, :, :]
        chi = unit.chi
        arg = (h_c[None, :, None] - s[None, :, :] * h_c[None, None, :]
               + chi[:, :, None] - chi[:, None, :])
        self.d_ij = 0.25 * np.tanh(arg)
        self.d_bar = self.d_ij[fm].mean(axis=0)
        beta = np.array([0.25 * np.tanh(M_H[i, l] - M_H[j, l]) for (i, j, l) in self.triples])
        self.beta = beta

        # --- M_D from the past composition-change direction ---------------
        self._set_M_D(chg)

        # --- field parameter bundle ---------------------------------------
        self.P = FieldParams(self.d_phi, M, self.d_xi, self.phi_anchor, self.cG, self.w_ij, s,
                             self.d_ij, beta, self.triples, unit.o, unit.xi_goal, self.Wv, fm,
                             self.lam_phi, self.lam_2, self.lam_3, self.lam_G, self.eps_D)

        # --- temperature and particles ------------------------------------
        self.o_bar = float(unit.mean_openness)
        self.T_D = max(1e-12, self.T_per_o * self.o_bar)
        prev_dir = self._previous_direction(unit, st)
        self.particles = self._init_particles(unit, prev_dir)
        e0 = [field_energy(self.P, self.particles[a]) for a in range(self.A)]

        self.field_traces.append({
            "unit": int(unit.index),
            "lambda_phi": self.lam_phi, "lambda_2": self.lam_2, "lambda_3": self.lam_3,
            "lambda_G": self.lam_G, "epsilon_D": self.eps_D, "step_tau": self.tau,
            "temperature_T_D": self.T_D, "mean_openness": self.o_bar, "kappa_E": self.kappa_E,
            "anchor_probe_gains": g_anchor.tolist(),
            "phi_anchor_mean": self.phi_anchor[fm].mean(axis=0).tolist(),
            "w_ij_mean": self.w_ij[fm].mean(axis=0).tolist(),
            "s_ij": s.tolist(), "significant_history_correlations": n_sig,
            "d_ij_mean": self.d_bar.tolist(), "beta": beta.tolist(),
            "triples": [list(map(int, t)) for t in self.triples],
            "M_D": {"kappa": self.kappa_eff, "isotropic_scale": self.m_scale, "v": self.v_c.tolist(),
                    "u_c_block": self.u[self.d_phi:self.d_phi + M].tolist()},
            "history_used": {"has_history": has_hist, "h_c": h_c.tolist(),
                             "corr_offdiag_absmax": float(np.abs(corr - np.diag(np.diag(corr))).max()),
                             "M_H_trace": float(np.trace(M_H)),
                             "previous_realized_blocks": bool(st.get("last_blocks"))},
            "initial_particle_field_energy": [float(x) for x in e0],
            "particle_initialisation_weights": list(self.init_weights),
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

    # ------------------------------------------------------------------ joint-structure hints
    def hints(self, unit: UnitContext, history) -> List[tuple]:
        """Strongest material-material couplings: s_ij = +1 -> 'sync', s_ij = -1 -> 'counter'."""
        if self.M < 3:
            return []
        fm = self.free_rows
        w_bar = self.w_ij[fm].mean(axis=0)
        pairs = [(i, j) for i in range(1, self.M) for j in range(i + 1, self.M)]
        pairs.sort(key=lambda ij: -w_bar[ij[0], ij[1]])
        out = []
        for (i, j) in pairs[: max(0, self.n_hints)]:
            out.append((int(i), int(j), "sync" if self.s_ij[i, j] > 0 else "counter"))
        return out

    # ------------------------------------------------------------------ possibility search (§7.2)
    def _langevin(self, unit: UnitContext, n_steps: int) -> int:
        """Eq. (29) on every particle; displacement scaled row-wise by openness, holds pinned."""
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
        return done

    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        n_steps = max(1, int(round(self.steps_max / (1.0 + round_index))))
        done = self._langevin(unit, n_steps)
        n = max(1, min(int(n_targets), self.A))
        targets: List[Target] = []
        energies = []
        for a in range(self.A):
            energies.append(field_energy(self.P, self.particles[a]))
        for a in range(n):
            xi_hat = fix_hold_rows(unit, self.particles[a])
            targets.append(Target(
                id=f"diffusion:u{unit.index}:r{round_index}:p{a}", xi_hat=xi_hat,
                meta={"particle": a, "field_energy": float(energies[a]), "steps": int(self.steps_used)}))
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

    # ------------------------------------------------------------------ connection to real gains (§7.3)
    def mode_error(self, unit: UnitContext, cand: Candidate, target: Target) -> float:
        """Eq. (30): <d_xi^2(realized, ideal)> over free windows + kappa_E * field energy of the
        realized composition.  Legal candidates are never rejected, only scored."""
        key = int(cand.id)
        e_field = self._e_cache.get(key)
        if e_field is None:
            e_field = field_energy(self.P, cand.xi)
            self._e_cache[key] = e_field
        return float(unit.mean_dist2(cand.xi, target.xi_hat) + self.kappa_E * e_field)

    # ------------------------------------------------------------------ required internal update
    def update(self, unit: UnitContext, history, realizations: Sequence[Realization],
               round_index: int) -> Dict[str, Any]:
        """Pull every particle a bounded step toward the realized composition it produced, and
        re-estimate the composition-change direction v (hence M_D) from those realizations."""
        if not realizations:
            return {}
        fm = self.free_rows[:, None]
        eta = self.pull_eta
        pulled = []
        e_real = []
        for r in realizations:
            a = int(r.target.meta.get("particle", -1))
            e_real.append(field_energy(self.P, r.candidate.xi))
            if not (0 <= a < self.A):
                continue
            x = self.particles[a]
            self.particles[a] = fix_hold_rows(unit, x + eta * fm * (r.candidate.xi - x))
            pulled.append(a)
        # v from the realized compositions (principal direction of the c-block increments)
        diffs = []
        for r in realizations:
            c = r.candidate.parts["c"][self.free_rows]
            if len(c) > 2:
                diffs.append(np.diff(c, axis=0))
        v_new = self.v_c
        if diffs:
            D = np.concatenate(diffs, axis=0)
            cov = D.T @ D / float(len(D))
            _, V = np.linalg.eigh(cov)
            v = V[:, -1]
            ref = np.zeros(self.M)
            for r in realizations:
                c = r.candidate.parts["c"][self.free_rows]
                if len(c) > 1:
                    ref += c[-1] - c[0]
            if float(v @ ref) < 0.0:
                v = -v
            v_new = v
            self._set_M_D(v_new)
        e_part = [field_energy(self.P, self.particles[a]) for a in range(self.A)]
        return {"pulled_particles": pulled, "eta": eta,
                "mean_particle_field_energy": float(np.mean(e_part)),
                "mean_realized_field_energy": float(np.mean(e_real)) if e_real else None,
                "v": [float(x) for x in np.asarray(v_new).ravel()],
                "M_D_kappa": float(self.kappa_eff)}

    # ------------------------------------------------------------------ history residue (§7.4)
    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        e_chosen = field_energy(self.P, chosen.candidate.xi)
        e_part = [field_energy(self.P, self.particles[a]) for a in range(self.A)]
        blocks = self.objective.phase_blocks(unit, chosen.candidate.xi)
        st = {
            "unit": int(unit.index),
            "s_ij": self.s_ij.tolist(),
            "d_ij_mean": self.d_bar.tolist(),
            "beta": [float(x) for x in self.beta],
            "triples": [list(map(int, t)) for t in self.triples],
            "last_blocks": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in blocks.items()},
            "particle_summary": {"field_energy_mean": float(np.mean(e_part)),
                                 "field_energy_std": float(np.std(e_part)),
                                 "n_particles": int(self.A), "steps_used": int(self.steps_used)},
            "v": [float(x) for x in np.asarray(self.v_c).ravel()],
            "M_D_kappa": float(self.kappa_eff),
            "chosen_field_energy": float(e_chosen),
        }
        history.mode_state[self.state_key] = st
        self.realized_traces.append({
            "unit": int(unit.index), "chosen_target": chosen.target.id,
            "mode_error": float(chosen.mode_error),
            "normalized_mode_error": float(chosen.normalized_mode_error),
            "ideal_vs_realized_mean_dist2": float(unit.mean_dist2(chosen.candidate.xi, chosen.target.xi_hat)),
            "realized_field_energy": float(e_chosen),
            "realized_field_energy_terms": field_breakdown(self.P, chosen.candidate.xi),
            "kappa_E_times_field_energy": float(self.kappa_E * e_chosen),
            "target_field_energy": float(chosen.target.meta.get("field_energy", float("nan"))),
            "alternatives_realized_field_energy_mean": float(np.mean(
                [field_energy(self.P, a.candidate.xi) for a in alternatives[:6]])) if alternatives else None,
        })
        self.unit_traces.append({
            "unit": int(unit.index), "chosen_target": chosen.target.id,
            "normalized_mode_error": float(chosen.normalized_mode_error),
            "field_energy_realized": float(e_chosen),
            "field_energy_particles_mean": float(np.mean(e_part)),
            "langevin_steps": int(self.steps_used), "temperature_T_D": self.T_D,
            "M_D_kappa": float(self.kappa_eff)})

    # ------------------------------------------------------------------ signature / trace
    def signature(self) -> np.ndarray:
        fm = self.free_rows
        parts = [self.s_ij.ravel(), self.d_bar.ravel(), np.asarray(self.beta, dtype=np.float64),
                 self.m_scale * (1.0 + self.kappa_eff * (self.u ** 2)),
                 self.kappa_eff * np.asarray(self.v_c).ravel()]
        for a in range(self.A):
            x = self.particles[a]
            parts.append(x[fm].mean(axis=0) if fm.any() and len(fm) == len(x) else x.mean(axis=0))
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])

    def trace(self) -> Dict[str, Any]:
        return {
            "field_parameters": self.field_traces,
            "particle_summary": self.particle_traces,
            "realized_mode_error": self.realized_traces,
            "equations": {
                "field_energy": "eq.(28)", "langevin": "eq.(29)", "connection": "eq.(30)",
                "note": ("finite-step search approximation with a variable openness coefficient; "
                         "not an exact Gibbs sample and not a trained DDPM (spec §7.2)"),
                "drift_convention": ("the drift uses the row-local gradient d e_t / d xi_t, i.e. the "
                                     "gradient of J * E_D; the factor J is absorbed into the positive "
                                     "definite search matrix so tau does not depend on the window count"),
                "drift_noise_balance": ("at the default tau=step and T_D=temperature_per_openness*o_bar "
                                        "the per-step drift is a small fraction of the per-step "
                                        "fluctuation (see particle_summary.mean_abs_drift_per_step vs "
                                        "mean_abs_noise_per_step): the finite search stays in the "
                                        "transient regime near its initialisation, as spec §7.2 permits"),
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

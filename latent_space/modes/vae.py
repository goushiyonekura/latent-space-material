"""VAE-type controller (spec §8; audit-2 §D2): a few shared latent factors z (d_z < M) jointly
generate an ideal acoustic composition through an explicit joint decoder

    D(z; t) = xi_G(t) + sum_k s_k tanh(z_k) v_k                                   (audit D2.3)

whose directions v_k come from the SVD of the metric-scaled differences W^{1/2}(xi_probe - xi_G)
of a few legal probe compositions of the *actual materials* over the whole piece (audit D2.2) —
not from the file order of the materials.  Sign rule and material ids are stored; the basis is
fixed for the whole job (audit D2.5).  The latent possibility region is

    q_t(z) = N( o(t) [mu_H + K_F chi(t)],  o(t)^2 Sigma_H )                       (eq. 32)

with chi(t) the material log-energy changes projected on the factor loadings.  Window references
are latent Ornstein-Uhlenbeck paths started from the latent coordinates of the realized current
composition (audit D3-like continuity for the VAE), with an explicit correlation time in seconds;
no target adaptation toward candidates during realization (audit D2.6).  Committed compositions
are projected back to latent coordinates (eq. 34) and mu_H / Sigma_H are updated with the
history time-constant rate.  Not a trained variational auto-encoder (spec [P-V])."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, ar1_rho_for, fix_hold_rows


class VAEMode(ModeController):
    cli_name = "vae"
    internal_id = "vae_latent"

    def __init__(self, cfg, analyzer, fs, rng, objective):
        super().__init__(cfg, analyzer, fs, rng, objective)
        md = cfg["mode_defaults"]
        self.p = dict(md["vae"])
        d_z = int(md["vae_latent_dim"])
        if d_z >= self.M:
            self.warnings.append(f"vae_latent_dim {d_z} >= M={self.M}; clipped to {self.M - 1}")
            d_z = self.M - 1
        self.d_z = max(1, min(d_z, self.N))
        self.W = analyzer.weight_vector()
        self.sqrtW = np.sqrt(self.W)
        self.hop_s = analyzer.hop / float(fs)
        self.model_step = float(cfg["analysis"].get("model_step_seconds", self.hop_s))
        tc = float(self.p.get("time_correlation", 0.9))
        # correlation time in seconds (old AR coefficient 0.9 at 0.1 s  ->  tau ~ 0.949 s)
        self.tau_z = float(self.p.get("latent_correlation_seconds", -0.1 / np.log(max(1e-6, min(0.999, tc)))))
        self.K_F = float(self.p["K_F"])
        self.cov_floor = float(self.p["cov_floor"])
        self.basis_ready = False
        self.basis_traces: List[Dict[str, Any]] = []
        self.fit_traces: List[Dict[str, Any]] = []
        self.latent_traces: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ acoustic shared basis
    def _build_basis(self, history) -> None:
        """Top-k metric-scaled directions of probe-composition differences over the whole piece."""
        st = history.mode_state.get("vae") if hasattr(history, "mode_state") else None
        if st and st.get("basis") and len(st["basis"].get("V", [])) == self.d_z:
            b = st["basis"]
            self.V = np.array(b["V"], dtype=np.float64)          # (k, d) directions in xi coordinates
            self.U = np.array(b["U"], dtype=np.float64)          # (k, d) orthonormal in W^{1/2} space
            self.s = np.array(b["s"], dtype=np.float64)          # (k,)
            self.loadings = np.array(b["loadings"], dtype=np.float64)  # (M, k)
            self.basis_hash = str(b["hash"])
            self.basis_source = "restored_from_history"
            self.basis_ready = True
            return
        an = self.analyzer
        J_all = an.J
        sub = np.arange(0, J_all, max(1, J_all // 400))
        probes = []
        for i in range(1, self.M):
            g = np.full(self.M, float(self.p["probe_out_gain"]))
            g[0] = float(self.p["probe_goal_gain"])
            g[i] = float(self.p["probe_in_gain"])
            probes.append(("foreground", i, g))
        g = np.full(self.M, 0.4)
        g[0] = 0.1
        probes.append(("anchor", -1, g))
        xi_goal = an.xi_goal_all[sub]
        cols = []
        per_probe = []
        for (kind, i, g) in probes:
            xi_p, _ = an.composition(np.broadcast_to(g, (len(sub), self.M)).copy(), sub)
            x = (xi_p - xi_goal) * self.sqrtW[None, :]
            cols.append(x)
            per_probe.append(x)
        X = np.concatenate(cols, axis=0)
        _u, sv, Vt = np.linalg.svd(X, full_matrices=False)
        k = self.d_z
        U = Vt[:k].copy()
        mean_x = X.mean(axis=0)
        for r in range(k):
            proj = float(mean_x @ U[r])
            if abs(proj) > 1e-9:
                if proj < 0:
                    U[r] = -U[r]
            else:
                j = int(np.argmax(np.abs(U[r])))
                if U[r, j] < 0:
                    U[r] = -U[r]
        s = sv[:k] / np.sqrt(float(X.shape[0]))
        s = np.maximum(s, 1e-6)
        V = U / self.sqrtW[None, :]           # xi-coordinate directions: W^{1/2} V = U
        loadings = np.zeros((self.M, k))
        for (kind, i, _g), x in zip(probes, per_probe):
            if i >= 1:
                loadings[i] = (x @ U.T).mean(axis=0) / s
        h = hashlib.sha256(np.ascontiguousarray(U).tobytes()).hexdigest()[:16]
        self.U, self.V, self.s, self.loadings, self.basis_hash = U, V, s, loadings, h
        self.basis_source = "svd_of_probe_differences_whole_piece"
        self.basis_ready = True
        share = float(np.sum(sv[:k] ** 2) / max(1e-300, float(np.sum(sv ** 2))))
        self.basis_traces.append({
            "source": self.basis_source, "hash": h, "singular_values": sv[: min(len(sv), 6)].tolist(),
            "probe_variance_share_captured": share, "k": int(k),
            "note": "k shared factors capture only part of the realizable composition variance; the remainder is "
                    "recorded as decoder residual (soft), not forced through gains",
            "scales": s.tolist(), "sign_rule": "mean projection of probe differences >= 0 (fallback: largest component positive)",
            "material_ids": list(range(1, self.M)), "probes": [{"kind": kd, "material": i, "gains": g.tolist()} for (kd, i, g) in probes],
            "loadings": loadings.tolist(), "rows_used": int(X.shape[0])})

    # ------------------------------------------------------------------ conditioning
    def begin_unit(self, unit: UnitContext, history) -> None:
        st = history.mode_state.get("vae")
        if st is None or "mu_H" not in st:
            self.mu_H = np.full(self.d_z, float(self.p["mu_H_init"]))
            self.Sigma_H = np.eye(self.d_z) * float(self.p["sigma_H_init"])
        else:
            self.mu_H = np.array(st["mu_H"], dtype=np.float64)
            self.Sigma_H = np.array(st["Sigma_H"], dtype=np.float64)
        if not self.basis_ready:
            self._build_basis(history)
        self.unit = unit
        self.chi_f = unit.chi[:, 1:] @ self.loadings[1:]          # (J, k) factor-wise material change
        self.mean_z = unit.o[:, None] * (self.mu_H[None, :] + self.K_F * self.chi_f)
        cov = self.Sigma_H + np.eye(self.d_z) * self.cov_floor
        self.L = np.linalg.cholesky(cov)
        self.unit_fits: List[np.ndarray] = []
        self.unit_step_count = 0
        self.latent_traces.append({"unit": unit.index, "mu_H_used": self.mu_H.tolist(),
                                   "Sigma_H_used": self.Sigma_H.tolist(), "basis_hash": self.basis_hash})

    # ------------------------------------------------------------------ decoder / encoder
    def decode(self, unit: UnitContext, z: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        xg = unit.xi_goal if rows is None else unit.xi_goal[rows]
        xi_hat = xg + (self.s[None, :] * np.tanh(z)) @ self.V
        if rows is None:
            return fix_hold_rows(unit, xi_hat)
        hm = unit.hold_mask[rows]
        xi_hat[hm] = xg[hm]
        return xi_hat

    def fit_latent(self, unit: UnitContext, xi: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Eq. (34) with the W-orthonormal basis: u_k = <W^{1/2}(xi - xi_G), U_k> / s_k, z = atanh(u)."""
        xg = unit.xi_goal if rows is None else unit.xi_goal[rows]
        x = (xi - xg) * self.sqrtW[None, :]
        u = (x @ self.U.T) / self.s[None, :]
        u = np.clip(u, -0.999, 0.999)
        z = np.arctanh(u)
        hm = unit.hold_mask if rows is None else unit.hold_mask[rows]
        z[hm] = 0.0
        return z

    def _ou_path(self, unit: UnitContext, rows: np.ndarray, z0: np.ndarray) -> np.ndarray:
        """Latent Ornstein-Uhlenbeck path on the rows: z_{j+1} = m_{j+1} + a (z_j - m_j) + sqrt(1-a^2) o L eps."""
        a = ar1_rho_for(self.hop_s, self.tau_z)
        m = self.mean_z[rows]
        o = unit.o[rows]
        z = np.zeros((len(rows), self.d_z))
        z[0] = z0
        for j in range(1, len(rows)):
            eps = self.rng.standard_normal(self.d_z)
            z[j] = m[j] + a * (z[j - 1] - m[j - 1]) + np.sqrt(max(0.0, 1.0 - a * a)) * o[j] * (self.L @ eps)
        z[unit.hold_mask[rows]] = 0.0
        return z

    # ------------------------------------------------------------------ proposals
    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        xi0 = unit.probe(unit.start_gains)[0][0]
        z0 = self.fit_latent(unit, xi0[None], np.array([0]))[0]
        rows = np.arange(unit.J)
        targets = []
        for a_ in range(n_targets):
            z = self._ou_path(unit, rows, z0)
            targets.append(Target(f"vae:u{unit.index}:full:{a_}", self.decode(unit, z),
                                  meta={"z_mean_free": z[unit.free_mask].mean(axis=0).tolist(), "basis_hash": self.basis_hash}))
        return targets

    def prepare_reference(self, unit: UnitContext, history, rows: np.ndarray, xi_current: np.ndarray,
                          n_proposals: int) -> List[Target]:
        rows = np.asarray(rows)
        z0 = self.fit_latent(unit, xi_current[None], rows[:1])[0]
        out = []
        for a_ in range(n_proposals):
            z = self._ou_path(unit, rows, z0)
            xi_hat = unit.xi_goal.copy()
            xi_hat[rows] = self.decode(unit, z, rows)
            out.append(Target(f"vae:u{unit.index}:s{self.unit_step_count}:{a_}", xi_hat,
                              meta={"z0": z0.tolist(), "z_mean": z.mean(axis=0).tolist(), "basis_hash": self.basis_hash,
                                    "tau_z_seconds": self.tau_z}))
        self.unit_step_count += 1
        return out

    def hints(self, unit: UnitContext, history) -> List[tuple]:
        # materials loading on the same sign of factor 1 tend to move together, opposite signs counter
        out = []
        ld = self.loadings[1:, 0]
        order = np.argsort(-np.abs(ld)) + 1
        if len(order) >= 2:
            a, b = int(order[0]), int(order[1])
            out.append((a, b, "sync" if ld[a - 1] * ld[b - 1] > 0 else "counter"))
        if len(order) >= 3:
            c = int(order[2])
            out.append((a, c, "sync" if ld[a - 1] * ld[c - 1] > 0 else "counter"))
        return out

    # ------------------------------------------------------------------ committed observations
    def observe_committed(self, unit: UnitContext, history, rows: np.ndarray, xi_rows: np.ndarray,
                          parts_rows: Dict[str, np.ndarray], reference: Optional[Target],
                          stats: Dict[str, Any]) -> Dict[str, Any]:
        rows = np.asarray(rows)
        fm = unit.free_mask[rows]
        if not fm.any():
            return {"skipped": "hold rows only"}
        z = self.fit_latent(unit, xi_rows, rows)[fm]
        self.unit_fits.append(z)
        dt = len(rows) * self.hop_s
        rho = history.rho_dt(dt)
        z_bar = z.mean(axis=0)
        dz = z - z_bar
        cov = dz.T @ dz / max(1, len(z)) + np.eye(self.d_z) * self.cov_floor
        self.mu_H = (1.0 - rho) * self.mu_H + rho * z_bar
        self.Sigma_H = (1.0 - rho) * self.Sigma_H + rho * cov
        self.mean_z = unit.o[:, None] * (self.mu_H[None, :] + self.K_F * self.chi_f)
        self.L = np.linalg.cholesky(self.Sigma_H + np.eye(self.d_z) * self.cov_floor)
        resid = float(unit.analyzer.dist2(xi_rows[fm], self.decode(unit, self.fit_latent(unit, xi_rows, rows), rows)[fm]).mean())
        self.step_traces.append({"unit": unit.index, "rows": int(len(rows)), "rho": rho, "z_committed_mean": z_bar.tolist(),
                                 "decoder_residual_mean_dist2": resid,
                                 "reference_z0": (reference.meta.get("z0") if reference is not None else None)})
        return {"rho": rho, "z_committed_mean": z_bar.tolist(), "decoder_residual": resid}

    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        zs = np.concatenate(self.unit_fits, axis=0) if self.unit_fits else np.zeros((0, self.d_z))
        st = dict(history.mode_state.get("vae", {}))
        st["mu_H"] = self.mu_H.tolist()
        st["Sigma_H"] = self.Sigma_H.tolist()
        st["basis"] = {"V": self.V.tolist(), "U": self.U.tolist(), "s": self.s.tolist(),
                       "loadings": self.loadings.tolist(), "hash": self.basis_hash, "material_ids": list(range(1, self.M))}
        st.setdefault("path_summaries", []).append({"unit": unit.index, "z_mean": (zs.mean(axis=0).tolist() if len(zs) else None),
                                                    "z_first": (zs[0].tolist() if len(zs) else None),
                                                    "z_last": (zs[-1].tolist() if len(zs) else None), "n_rows": int(len(zs))})
        st["path_summaries"] = st["path_summaries"][-8:]
        history.mode_state["vae"] = st
        self.fit_traces.append({"unit": unit.index, "z_actual_mean": (zs.mean(axis=0).tolist() if len(zs) else None),
                                "z_actual_cov": (np.cov(zs.T).tolist() if len(zs) > 1 else None),
                                "target_z_mean_free": chosen.target.meta.get("z_mean_free")})
        self.unit_traces.append({"unit": unit.index, "chosen_target": chosen.target.id,
                                 "normalized_mode_error": chosen.normalized_mode_error,
                                 "mu_H_after": self.mu_H.tolist(), "Sigma_H_after": self.Sigma_H.tolist()})

    def signature(self) -> np.ndarray:
        return np.concatenate([self.mu_H, self.Sigma_H.ravel(), self.mean_z[self.unit.free_mask].mean(axis=0)
                               if self.unit.free_mask.any() else np.zeros(self.d_z)])

    def trace(self) -> Dict[str, Any]:
        return {"latent_dim": self.d_z, "latent_correlation_seconds": self.tau_z, "model_step_seconds": self.model_step,
                "latent_means_covariances": self.latent_traces, "basis_definition": self.basis_traces,
                "fitted_actual_latent_summary": self.fit_traces, "decoder": "xi_G + sum_k s_k tanh(z_k) v_k",
                "warnings": list(self.warnings), "units": self.unit_traces, "steps": self.step_traces[-64:]}

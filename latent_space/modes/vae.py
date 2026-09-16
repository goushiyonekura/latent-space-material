"""VAE-type controller (spec §8): a few shared latent factors z (d_z < M) jointly generate an
ideal acoustic composition through a decoder built from *current* probe mixes,
  D_r(z; F_t) = xi_G(t) + B_r(t) z,   B_r(t) = [xi^{probe,k}(t) - xi_G(t)]_k        (eq. 31)
  q_t(z)      = N( o(t)[mu_H + K_F chi(F_t)],  o(t)^2 Sigma_H )                     (eq. 32)
Realized compositions are projected back to latent coordinates by ridge least squares
(eq. 34) and mu_H / Sigma_H are EMA-updated from them; both live in history.mode_state["vae"].
Not a trained variational auto-encoder: a latent-generative *controller* (spec [P-V])."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, ar1_noise, fix_hold_rows


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
        if d_z > self.N:
            d_z = self.N
        self.d_z = max(1, d_z)
        self.rho = float(cfg["history"]["update_rate"])
        # probe groups over non-goal tracks 1..N (round-robin), each group non-empty
        self.groups: List[List[int]] = [[i for i in range(1, self.M) if (i - 1) % self.d_z == k] for k in range(self.d_z)]
        self.W = analyzer.weight_vector()
        self.basis_traces: List[Dict[str, Any]] = []
        self.fit_traces: List[Dict[str, Any]] = []
        self.latent_traces: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ conditioning
    def begin_unit(self, unit: UnitContext, history) -> None:
        st = history.mode_state.get("vae")
        if st is None:
            self.mu_H = np.full(self.d_z, float(self.p["mu_H_init"]))
            self.Sigma_H = np.eye(self.d_z) * float(self.p["sigma_H_init"])
        else:
            self.mu_H = np.array(st["mu_H"], dtype=np.float64)
            self.Sigma_H = np.array(st["Sigma_H"], dtype=np.float64)
        self.unit = unit
        J = unit.J
        self.B = np.zeros((J, self.d_xi, self.d_z))
        self.probe_gains = []
        for k, grp in enumerate(self.groups):
            g = np.full(self.M, float(self.p["probe_out_gain"]))
            g[0] = float(self.p["probe_goal_gain"])
            g[grp] = float(self.p["probe_in_gain"])
            xi_p, _ = unit.probe(g)
            self.B[:, :, k] = xi_p - unit.xi_goal
            self.probe_gains.append(g.tolist())
        chi_group = np.stack([unit.chi[:, grp].mean(axis=1) for grp in self.groups], axis=1)  # (J, d_z)
        self.chi_group = chi_group
        self.mean_z = unit.o[:, None] * (self.mu_H[None, :] + float(self.p["K_F"]) * chi_group)
        cov = self.Sigma_H + np.eye(self.d_z) * float(self.p["cov_floor"])
        self.L = np.linalg.cholesky(cov)
        self.z_shift = np.zeros(self.d_z)
        Bm = self.B[unit.free_mask].mean(axis=0) if unit.free_mask.any() else self.B.mean(axis=0)
        sv = np.linalg.svd(Bm, compute_uv=False)
        self.basis_sv = sv
        if sv.min() < 1e-6 * max(1.0, sv.max()):
            self.warnings.append(f"unit {unit.index}: probe basis nearly degenerate (singular values {sv.tolist()})")
        self.basis_traces.append({"unit": unit.index, "probe_gains": self.probe_gains,
                                  "mean_basis_singular_values": sv.tolist(), "groups": self.groups,
                                  "mu_H_used": self.mu_H.tolist(), "Sigma_H_used": self.Sigma_H.tolist()})

    def decode(self, unit: UnitContext, z: np.ndarray) -> np.ndarray:
        xi_hat = unit.xi_goal + np.einsum("jdk,jk->jd", self.B, z)
        return fix_hold_rows(unit, xi_hat)

    def hints(self, unit: UnitContext, history) -> List[tuple]:
        # tracks sharing a latent group tend to move together; across groups they contrast
        out = []
        for grp in self.groups:
            for a, b in zip(grp[:-1], grp[1:]):
                out.append((a, b, "sync"))
        if len(self.groups) > 1 and self.groups[0] and self.groups[1]:
            out.append((self.groups[0][0], self.groups[1][0], "counter"))
        return out

    # ------------------------------------------------------------------ proposals
    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        targets = []
        rho_t = float(self.p["time_correlation"])
        for a in range(n_targets):
            eps = ar1_noise(self.rng, unit.J, self.d_z, rho_t)
            z = self.mean_z + unit.o[:, None] * self.z_shift[None, :] + unit.o[:, None] * (eps @ self.L.T)
            z[unit.hold_mask] = 0.0
            targets.append(Target(f"vae:u{unit.index}:r{round_index}:{a}", self.decode(unit, z),
                                  meta={"z_mean_free": z[unit.free_mask].mean(axis=0).tolist()}))
        return targets

    # ------------------------------------------------------------------ latent fit (eq. 34)
    def fit_latent(self, unit: UnitContext, xi: np.ndarray) -> np.ndarray:
        eps = float(self.p["ridge_epsilon"])
        BtWB = np.einsum("jdk,d,jdl->jkl", self.B, self.W, self.B) + eps * np.eye(self.d_z)[None]
        rhs = np.einsum("jdk,d,jd->jk", self.B, self.W, xi - unit.xi_goal)
        z = np.linalg.solve(BtWB, rhs[:, :, None])[:, :, 0]
        z[unit.hold_mask] = 0.0
        return z

    def update(self, unit: UnitContext, history, realizations: Sequence[Realization], round_index: int) -> Dict[str, Any]:
        """Amortised refinement: move the proposal centre halfway toward the latent coordinates of
        the best realized compositions of this round."""
        if not realizations:
            return {}
        fm = unit.free_mask
        zs = [self.fit_latent(unit, r.candidate.xi)[fm].mean(axis=0) for r in realizations]
        z_act = np.mean(zs, axis=0)
        z_prop = (self.mean_z + unit.o[:, None] * self.z_shift[None, :])[fm].mean(axis=0)
        self.z_shift = self.z_shift + 0.5 * (z_act - z_prop)
        return {"z_actual_mean": z_act.tolist(), "z_shift": self.z_shift.tolist()}

    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        fm = unit.free_mask
        z = self.fit_latent(unit, chosen.candidate.xi)
        zf = z[fm] if fm.any() else z
        z_bar = zf.mean(axis=0)
        dz = zf - z_bar
        cov = dz.T @ dz / max(1, len(zf))
        floor = np.eye(self.d_z) * float(self.p["cov_floor"])
        self.mu_H = (1.0 - self.rho) * self.mu_H + self.rho * z_bar
        self.Sigma_H = (1.0 - self.rho) * self.Sigma_H + self.rho * (cov + floor)
        resid = float(unit.mean_dist2(chosen.candidate.xi, self.decode(unit, z)))
        st = history.mode_state.get("vae", {"path_summaries": []})
        st = dict(st)
        st["mu_H"] = self.mu_H.tolist()
        st["Sigma_H"] = self.Sigma_H.tolist()
        st.setdefault("path_summaries", []).append({"unit": unit.index, "z_mean": z_bar.tolist(),
                                                    "z_first": zf[0].tolist(), "z_last": zf[-1].tolist()})
        st["path_summaries"] = st["path_summaries"][-8:]
        history.mode_state["vae"] = st
        self.fit_traces.append({"unit": unit.index, "z_actual_mean": z_bar.tolist(), "z_actual_cov": cov.tolist(),
                                "decoder_residual_mean_dist2": resid,
                                "target_z_mean_free": chosen.target.meta.get("z_mean_free")})
        self.latent_traces.append({"unit": unit.index, "mu_H_after": self.mu_H.tolist(),
                                   "Sigma_H_after": self.Sigma_H.tolist()})
        self.unit_traces.append({"unit": unit.index, "chosen_target": chosen.target.id,
                                 "normalized_mode_error": chosen.normalized_mode_error})

    def signature(self) -> np.ndarray:
        return np.concatenate([self.mu_H, self.Sigma_H.ravel(), self.mean_z.mean(axis=0), self.z_shift])

    def trace(self) -> Dict[str, Any]:
        return {"latent_dim": self.d_z, "latent_means_covariances": self.latent_traces,
                "basis_definition": self.basis_traces, "fitted_actual_latent_summary": self.fit_traces,
                "warnings": list(self.warnings), "units": self.unit_traces}

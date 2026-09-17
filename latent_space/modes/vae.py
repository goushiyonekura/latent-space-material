"""VAE-type controller (spec §8; audit-2 §D2; fragment-vocabulary revision 2026-09-16): a few
shared latent factors z (d_z < M) jointly generate an ideal acoustic composition through an
explicit joint decoder

    D(z; t) = xi_G(t) + sum_k s_k tanh(z_k) v_k                                   (audit D2.3)

whose directions v_k come from the SVD of the metric-scaled differences W^{1/2}(xi - xi_G).

Two sources of those differences, chosen by what the analyzer offers:

* fragment configs (`analyzer.frag_f` built and `self.sources` set by the engine, FRAG_CONTRACT):
  N >= 1500 rows of *random fragment compositions* — "track i plays from a random position of its
  own source at a random level" — i.e. exactly the vocabulary the hires realizer can reach by
  jumping and switching levels.  The ideal therefore lives inside the enlarged reachable set.
* baseline configs (no fragment bank): the previous probe compositions of the actual materials
  over the whole piece (audit D2.2) — unchanged.

Sign rule and material ids are stored; the basis is fixed for the whole job (audit D2.5) and
persisted in `history.mode_state['vae']`.  The latent possibility region is

    q_t(z) = N( o(t) [mu_H + K_F chi(t)],  o(t)^2 Sigma_H )                       (eq. 32)

with chi(t) the material log-energy changes projected on the factor loadings.  Window references
are latent Ornstein-Uhlenbeck paths **at the model step** (`analysis.model_step_seconds`), started
from the latent coordinates of the realized current composition and linearly interpolated onto the
observation rows, with an explicit correlation time in seconds; no target adaptation toward
candidates during realization (audit D2.6).  Committed compositions are projected back to latent
coordinates (eq. 34) and mu_H / Sigma_H are updated with the history time-constant rate.  Not a
trained variational auto-encoder (spec [P-V])."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Candidate, Realization, Target, UnitContext
from .base import ModeController, ar1_rho_for, fix_hold_rows

SAT_HIGH = 0.9      # |tanh z| above this: the decoder direction is saturated (range exhausted)
SAT_LOW = 0.1       # |tanh z| below this: the decoder direction is essentially unused


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
        self.d_z = max(1, min(d_z, self.N))          # k < M and k <= N
        self.W = analyzer.weight_vector()
        self.sqrtW = np.sqrt(self.W)
        self.hop_s = analyzer.hop / float(fs)
        self.model_step = float(cfg["analysis"].get("model_step_seconds", self.hop_s))
        tc = float(self.p.get("time_correlation", 0.9))
        # correlation time in seconds (old AR coefficient 0.9 at 0.1 s  ->  tau ~ 0.949 s)
        self.tau_z = float(self.p.get("latent_correlation_seconds", -0.1 / np.log(max(1e-6, min(0.999, tc)))))
        self.K_F = float(self.p["K_F"])
        self.cov_floor = float(self.p["cov_floor"])
        self.ridge = float(self.p.get("ridge_epsilon", 1e-6))
        # --- fragment-mode parameters (all read with .get so the shared config needs no change)
        self.sources: Optional[Sequence[np.ndarray]] = None   # set by engine.setup (FRAG_CONTRACT)
        self.frag = False
        self.basis_compositions = int(self.p.get("basis_fragment_compositions", 300))
        self.basis_offsets = int(self.p.get("basis_fragment_offsets", 5))
        self.basis_min_rows = int(self.p.get("basis_min_rows", 1500))
        self.basis_seed_offset = int(self.p.get("basis_seed_offset", 9176))
        self.scale_quantile = float(self.p.get("basis_scale_quantile", 99.0))
        self.scale_target_tanh = float(self.p.get("basis_scale_target_tanh", 0.95))
        self.sigma_H_floor_frag = float(self.p.get("sigma_H_floor_frag", 0.09))
        self.basis_ready = False
        self.basis_z_mean = np.zeros(self.d_z)
        self.basis_traces: List[Dict[str, Any]] = []
        self.fit_traces: List[Dict[str, Any]] = []
        self.latent_traces: List[Dict[str, Any]] = []
        self.unit_stats: List[Dict[str, Any]] = []
        self.frag_basis_info: Optional[Dict[str, Any]] = None
        # job-level tanh-usage counters (committed rows only)
        self.sat_counts = np.zeros(self.d_z)
        self.low_counts = np.zeros(self.d_z)
        self.tanh_sum = np.zeros(self.d_z)
        self.tanh_rows = 0

    # ------------------------------------------------------------------ fragment detection
    def _fragment_available(self) -> bool:
        """Fragment vocabulary present (FRAG_CONTRACT: analyzer.frag_f + engine-supplied sources)."""
        return getattr(self.analyzer, "frag_f", None) is not None and getattr(self, "sources", None) is not None

    # ------------------------------------------------------------------ acoustic shared basis
    def _basis_rows_fragment(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """>= `basis_min_rows` rows of W^{1/2}(xi - xi_G) from random fragment compositions.

        One composition = random playback position + random level per material (goal level 0: the
        searched region of every unit has the goal track at 0 under the contract_only policy), read
        at `basis_offsets` consecutive hop offsets — the same 0.1 s spacing and the same 0.5 s span
        as one model step.  xi_G per row: `an.xi_goal_all` at a *matching* block of consecutive grid
        rows (same spacing), drawn uniformly over the piece, so that the differences are formed
        exactly the way the decoder forms them (xi - xi_goal at the same absolute time) and the
        goal track's own variation over the piece is averaged over, not baked into a direction."""
        an = self.analyzer
        seed = (int(self.cfg.get("seed", 0)) + self.basis_seed_offset) % (2 ** 32)
        rng_b = np.random.default_rng(seed)
        hop_frames = int(an.hop)
        clip_s = float(self.cfg.get("hires", {}).get("clip_feature_seconds", 2.0))
        n_off = max(2, self.basis_offsets)
        off_all = np.arange(0, max(hop_frames, int(round(clip_s * self.fs))), hop_frames, dtype=np.int64)
        offsets = off_all[:n_off] if len(off_all) >= n_off else np.arange(n_off, dtype=np.int64) * hop_frames
        n_off = int(len(offsets))
        n_comp = max(int(self.basis_compositions), int(np.ceil(self.basis_min_rows / float(n_off))))
        blocks, goal_blocks, levels = [], [], []
        for _ in range(n_comp):
            xi, _parts, meta = an.random_fragment_composition(self.sources, rng_b, offsets, goal_level=0.0)
            j0 = int(rng_b.integers(0, max(1, an.J)))
            jj = (j0 + np.arange(n_off)) % an.J
            blocks.append(xi)
            goal_blocks.append(an.xi_goal_all[jj])
            levels.append(np.asarray(meta["levels"], dtype=np.float64))
        XI = np.concatenate(blocks, axis=0)
        XG = np.concatenate(goal_blocks, axis=0)
        X = (XI - XG) * self.sqrtW[None, :]
        info = {"kind": "random_fragment_compositions", "compositions": int(n_comp),
                "offsets_frames": offsets.tolist(), "offsets_seconds": (offsets / float(self.fs)).tolist(),
                "rows": int(X.shape[0]), "seed": int(seed), "goal_level": 0.0,
                "levels": np.stack(levels, axis=0), "n_off": n_off,
                "xi_goal_rule": "an.xi_goal_all at a random block of consecutive grid rows per composition "
                                "(same hop spacing as the offsets)",
                "row_dist2_to_goal_mean": float(an.dist2(XI, XG).mean())}
        return X, info

    def _basis_rows_probe(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Baseline (no fragment bank): the previous probe compositions over the whole piece."""
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
        cols, per_probe = [], []
        for (kind, i, g) in probes:
            xi_p, _ = an.composition(np.broadcast_to(g, (len(sub), self.M)).copy(), sub)
            x = (xi_p - xi_goal) * self.sqrtW[None, :]
            cols.append(x)
            per_probe.append(x)
        X = np.concatenate(cols, axis=0)
        info = {"kind": "probe_compositions_whole_piece", "probes": probes, "per_probe": per_probe,
                "rows": int(X.shape[0]), "grid_rows": int(len(sub))}
        return X, info

    def _build_basis(self, history) -> None:
        """Top-k metric-scaled directions of the difference rows; fixed for the whole job."""
        st = history.mode_state.get("vae") if hasattr(history, "mode_state") else None
        if st and st.get("basis") and len(st["basis"].get("V", [])) == self.d_z:
            b = st["basis"]
            self.V = np.array(b["V"], dtype=np.float64)          # (k, d) directions in xi coordinates
            self.U = np.array(b["U"], dtype=np.float64)          # (k, d) orthonormal in W^{1/2} space
            self.s = np.array(b["s"], dtype=np.float64)          # (k,)
            self.loadings = np.array(b["loadings"], dtype=np.float64)  # (M, k)
            self.basis_hash = str(b["hash"])
            self.basis_z_mean = np.array(b.get("z_mean", np.zeros(self.d_z)), dtype=np.float64)
            self.frag = bool(b.get("fragment", False))
            self.frag_basis_info = b.get("fragment_basis")
            self.basis_source = "restored_from_history"
            self.basis_ready = True
            return
        self.frag = self._fragment_available()
        if self.frag:
            X, info = self._basis_rows_fragment()
        else:
            X, info = self._basis_rows_probe()
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
        proj = X @ U.T                                          # (rows, k) coordinates of the rows
        if self.frag:
            # tanh must be able to *represent* the fragment vocabulary and still have head-room:
            # s_k maps the `scale_quantile` percentile of |u_k| onto |tanh z| = scale_target_tanh.
            q = np.percentile(np.abs(proj), self.scale_quantile, axis=0)
            s = np.maximum(q / max(1e-6, self.scale_target_tanh), 1e-6)
        else:
            s = np.maximum(sv[:k] / np.sqrt(float(X.shape[0])), 1e-6)   # unchanged baseline rule
        V = U / self.sqrtW[None, :]           # xi-coordinate directions: W^{1/2} V = U
        loadings = np.zeros((self.M, k))
        if self.frag:
            # "how far does raising material i push factor k" — ridge regression of the per-
            # composition mean factor coordinate on the per-composition material levels.
            lv = np.asarray(info.pop("levels"))                 # (n_comp, M)
            n_off = int(info.pop("n_off"))
            ubar = proj.reshape(len(lv), n_off, k).mean(axis=1)  # (n_comp, k)
            A = np.concatenate([np.ones((len(lv), 1)), lv[:, 1:]], axis=1)
            G = A.T @ A + self.ridge * np.eye(A.shape[1])
            B = np.linalg.solve(G, A.T @ ubar)                  # (1 + N, k)
            loadings[1:] = B[1:] / s[None, :]
        else:
            for (kind, i, _g), x in zip(info["probes"], info["per_probe"]):
                if i >= 1:
                    loadings[i] = (x @ U.T).mean(axis=0) / s
        h = hashlib.sha256(np.ascontiguousarray(U).tobytes()).hexdigest()[:16]
        self.U, self.V, self.s, self.loadings, self.basis_hash = U, V, s, loadings, h
        share = float(np.sum(sv[:k] ** 2) / max(1e-300, float(np.sum(sv ** 2))))
        tanh_rows = np.clip(proj / s[None, :], -0.999, 0.999)
        self.basis_z_mean = np.arctanh(tanh_rows).mean(axis=0)
        resid = float(((X - proj @ U) ** 2).sum(axis=1).mean())   # = mean dist2 of the k-truncation
        tr = {"source": ("svd_of_random_fragment_composition_differences" if self.frag
                         else "svd_of_probe_differences_whole_piece"),
              "hash": h, "singular_values": sv[: min(len(sv), 8)].tolist(),
              "variance_share_captured_by_k": share, "k": int(k), "rows_used": int(X.shape[0]),
              "scales": s.tolist(), "reconstruction_residual_dist2": resid,
              "sign_rule": "mean projection of the difference rows >= 0 (fallback: largest component positive)",
              "material_ids": list(range(1, self.M)), "loadings": loadings.tolist(),
              "latent_mean_of_basis_rows": self.basis_z_mean.tolist(),
              "row_coordinate_rms": np.sqrt((proj ** 2).mean(axis=0)).tolist(),
              "note": "k shared factors capture only part of the realizable composition variance; the remainder is "
                      "recorded as decoder residual (soft), not forced through gains"}
        if self.frag:
            info.pop("per_probe", None)
            tanh_abs = np.abs(tanh_rows)
            info.update({"scale_rule": f"s_k = percentile_{self.scale_quantile:g}(|u_k|) / {self.scale_target_tanh:g}",
                         "basis_rows_saturated_fraction": float((tanh_abs > SAT_HIGH).mean()),
                         "basis_rows_underused_fraction": float((tanh_abs < SAT_LOW).mean()),
                         "variance_share_captured_by_k": share,
                         "singular_values": sv[: min(len(sv), 8)].tolist(),
                         "reconstruction_residual_dist2": resid, "scales": s.tolist()})
            self.frag_basis_info = info
            tr["fragment_basis"] = info
        else:
            tr["probes"] = [{"kind": kd, "material": i, "gains": g.tolist()} for (kd, i, g) in info["probes"]]
            tr["probe_variance_share_captured"] = share
        self.basis_source = tr["source"]
        self.basis_ready = True
        self.basis_traces.append(tr)

    # ------------------------------------------------------------------ conditioning
    def _sigma_effective(self) -> np.ndarray:
        """Sigma_H as used for the latent spread; in fragment mode the diagonal is floored so the
        ideal keeps exploring tanh's range instead of collapsing onto the EMA mean."""
        S = np.array(self.Sigma_H, dtype=np.float64, copy=True)
        if self.frag and self.sigma_H_floor_frag > 0.0:
            d = np.diag(S).copy()
            np.fill_diagonal(S, np.maximum(d, self.sigma_H_floor_frag))
        return S

    def _refresh_L(self) -> None:
        self.Sigma_used = self._sigma_effective()
        self.L = np.linalg.cholesky(self.Sigma_used + np.eye(self.d_z) * self.cov_floor)

    def begin_unit(self, unit: UnitContext, history) -> None:
        if not self.basis_ready:
            self._build_basis(history)
        st = history.mode_state.get("vae")
        if st is None or "mu_H" not in st:
            # fragment mode: start at the centre of the fragment cloud (re-initialised, recorded)
            self.mu_H = (self.basis_z_mean.copy() if self.frag
                         else np.full(self.d_z, float(self.p["mu_H_init"])))
            sig0 = float(self.p["sigma_H_init"])
            if self.frag:
                sig0 = max(sig0, self.sigma_H_floor_frag)
            self.Sigma_H = np.eye(self.d_z) * sig0
            self.state_origin = "re-initialised (no vae state in history)"
        else:
            self.mu_H = np.array(st["mu_H"], dtype=np.float64)
            self.Sigma_H = np.array(st["Sigma_H"], dtype=np.float64)
            self.state_origin = "inherited from history.mode_state['vae']"
        self.unit = unit
        self.chi_f = unit.chi[:, 1:] @ self.loadings[1:]          # (J, k) factor-wise material change
        self.mean_z = unit.o[:, None] * (self.mu_H[None, :] + self.K_F * self.chi_f)
        self._refresh_L()
        self.unit_fits: List[np.ndarray] = []
        self.unit_step_count = 0
        self._u_sat = np.zeros(self.d_z)
        self._u_low = np.zeros(self.d_z)
        self._u_tanh = np.zeros(self.d_z)
        self._u_rows = 0
        self._u_resid: List[float] = []
        self._u_dz_row: List[float] = []
        self._u_dz_step: List[float] = []
        self._u_ideal_disp: List[float] = []
        self._u_ideal_disp_lat: List[float] = []
        self._u_ideal_disp_commit: List[float] = []
        self._u_real_disp: List[float] = []
        self._last_commit_z: Optional[np.ndarray] = None
        self._last_commit_xi: Optional[np.ndarray] = None
        self.latent_traces.append({"unit": unit.index, "mu_H_used": self.mu_H.tolist(),
                                   "Sigma_H_used": self.Sigma_H.tolist(),
                                   "Sigma_H_effective": self.Sigma_used.tolist(),
                                   "state_origin": self.state_origin, "fragment_mode": bool(self.frag),
                                   "basis_hash": self.basis_hash})

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

    # ------------------------------------------------------------------ latent path
    def _knots(self, n_rows: int) -> np.ndarray:
        """Row indices of the model-step grid inside a window (always includes first and last)."""
        if n_rows <= 1:
            return np.zeros(max(0, n_rows), dtype=np.int64)
        stride = max(1, int(round(self.model_step / max(1e-9, self.hop_s))))
        kn = np.arange(0, n_rows, stride, dtype=np.int64)
        if kn[-1] != n_rows - 1:
            kn = np.append(kn, n_rows - 1)
        return kn

    def _ou_path(self, unit: UnitContext, rows: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Latent Ornstein-Uhlenbeck path **at the model step**:

            z_{j+1} = m_{j+1} + a (z_j - m_j) + sqrt(1 - a^2) o L eps,   a = exp(-model_step / tau_z)

        drawn on the model-step knots and linearly interpolated onto the observation rows (hop
        0.1 s), so the ideal changes by one realizable amount per commit step instead of wiggling
        at the observation hop.  With model_step == hop (the baseline configs) the knots are the
        rows and this reduces exactly to the previous per-row recursion."""
        kn = self._knots(len(rows))
        m = self.mean_z[rows]
        o = unit.o[rows]
        zk = np.zeros((len(kn), self.d_z))
        zk[0] = z0
        for j in range(1, len(kn)):
            dt = (kn[j] - kn[j - 1]) * self.hop_s
            a = ar1_rho_for(dt, self.tau_z)
            eps = self.rng.standard_normal(self.d_z)
            zk[j] = m[kn[j]] + a * (zk[j - 1] - m[kn[j - 1]]) + np.sqrt(max(0.0, 1.0 - a * a)) * o[kn[j]] * (self.L @ eps)
        if len(kn) == len(rows):
            z = zk
        else:
            xs = np.arange(len(rows), dtype=np.float64)
            z = np.stack([np.interp(xs, kn.astype(np.float64), zk[:, c]) for c in range(self.d_z)], axis=1)
        z[unit.hold_mask[rows]] = 0.0
        return z, kn

    # ------------------------------------------------------------------ proposals
    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        """Full-unit ideal trajectories (warm start of the baseline engine / unit reference R0)."""
        if self.frag and float(np.max(np.abs(unit.start_gains[1:]))) <= 1e-9:
            z0 = self.basis_z_mean.copy()          # unit starts from silence: start at the cloud centre
        else:
            xi0 = unit.probe(unit.start_gains)[0][0]
            z0 = self.fit_latent(unit, xi0[None], np.array([0]))[0]
        rows = np.arange(unit.J)
        targets = []
        for a_ in range(n_targets):
            z, kn = self._ou_path(unit, rows, z0)
            xi_hat = self.decode(unit, z)
            targets.append(Target(f"vae:u{unit.index}:full:{a_}", xi_hat,
                                  meta={"z_mean_free": z[unit.free_mask].mean(axis=0).tolist(),
                                        "step_displacement": self._step_displacement(unit, xi_hat, rows, kn, z),
                                        "basis_hash": self.basis_hash}))
        return targets

    def _step_displacement(self, unit: UnitContext, xi_hat_rows: np.ndarray, rows: np.ndarray,
                           kn: np.ndarray, z: np.ndarray) -> Dict[str, float]:
        """Mean d_xi^2 the ideal moves per model step (free knots only), split into

        * `total`  : d_xi^2(D(z_{j+1}; t_{j+1}), D(z_j; t_j)) — includes the motion of xi_G(t)
                     itself, which the decoder inherits and which no gain choice can remove;
        * `latent` : the part the controller actually commands, d_xi^2 at a frozen xi_G, which by
                     W-orthonormality of U is exactly sum_k (s_k [tanh z_{j+1,k} - tanh z_{j,k}])^2."""
        if len(kn) < 2:
            return {"total": 0.0, "latent": 0.0}
        fm = unit.free_mask[rows][kn[1:]] & unit.free_mask[rows][kn[:-1]]
        if not fm.any():
            return {"total": 0.0, "latent": 0.0}
        tot = unit.analyzer.dist2(xi_hat_rows[kn[1:]], xi_hat_rows[kn[:-1]])
        w = self.s[None, :] * np.tanh(z[kn])
        lat = ((w[1:] - w[:-1]) ** 2).sum(axis=1)
        return {"total": float(tot[fm].mean()), "latent": float(lat[fm].mean())}

    def prepare_reference(self, unit: UnitContext, history, rows: np.ndarray, xi_current: np.ndarray,
                          n_proposals: int) -> List[Target]:
        rows = np.asarray(rows)
        # hires has just written the material features at the *played* positions into the unit
        # context; refresh the factor conditioning of those rows before drawing the ideal.
        self.chi_f[rows] = unit.chi[rows][:, 1:] @ self.loadings[1:]
        self.mean_z[rows] = unit.o[rows][:, None] * (self.mu_H[None, :] + self.K_F * self.chi_f[rows])
        z0 = self.fit_latent(unit, xi_current[None], rows[:1])[0]
        out = []
        for a_ in range(n_proposals):
            z, kn = self._ou_path(unit, rows, z0)
            xi_rows = self.decode(unit, z, rows)
            xi_hat = unit.xi_goal.copy()
            xi_hat[rows] = xi_rows
            out.append(Target(f"vae:u{unit.index}:s{self.unit_step_count}:{a_}", xi_hat,
                              meta={"z0": z0.tolist(), "z_mean": z.mean(axis=0).tolist(), "basis_hash": self.basis_hash,
                                    "tau_z_seconds": self.tau_z, "model_step_seconds": self.model_step,
                                    "ar_coefficient": ar1_rho_for(self.model_step, self.tau_z),
                                    "step_displacement": self._step_displacement(unit, xi_rows, rows, kn, z)}))
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
        z_all = self.fit_latent(unit, xi_rows, rows)
        z = z_all[fm]
        self.unit_fits.append(z)
        dt = len(rows) * self.hop_s
        rho = history.rho_dt(dt)
        z_bar = z.mean(axis=0)
        dz = z - z_bar
        cov = dz.T @ dz / max(1, len(z)) + np.eye(self.d_z) * self.cov_floor
        self.mu_H = (1.0 - rho) * self.mu_H + rho * z_bar
        self.Sigma_H = (1.0 - rho) * self.Sigma_H + rho * cov
        self.mean_z = unit.o[:, None] * (self.mu_H[None, :] + self.K_F * self.chi_f)
        self._refresh_L()
        # --- decoder residual on the committed rows
        resid = float(unit.analyzer.dist2(xi_rows[fm], self.decode(unit, z_all, rows)[fm]).mean())
        # --- tanh usage of the committed rows (saturation / under-use per factor)
        tz = np.abs(np.tanh(z))
        sat = (tz > SAT_HIGH).mean(axis=0)
        low = (tz < SAT_LOW).mean(axis=0)
        self._u_sat += (tz > SAT_HIGH).sum(axis=0)
        self._u_low += (tz < SAT_LOW).sum(axis=0)
        self._u_tanh += tz.sum(axis=0)
        self._u_rows += len(z)
        self.sat_counts += (tz > SAT_HIGH).sum(axis=0)
        self.low_counts += (tz < SAT_LOW).sum(axis=0)
        self.tanh_sum += tz.sum(axis=0)
        self.tanh_rows += len(z)
        # --- latent path smoothness: |dz| per observation row and per model (commit) step
        dz_row = float(np.abs(np.diff(z, axis=0)).sum(axis=1).mean()) if len(z) > 1 else 0.0
        dz_step = (float(np.abs(z_bar - self._last_commit_z).sum()) if self._last_commit_z is not None else 0.0)
        if self._last_commit_z is not None:
            self._u_dz_step.append(dz_step)
        self._last_commit_z = z_bar.copy()
        self._u_resid.append(resid)
        if len(z) > 1:
            self._u_dz_row.append(dz_row)
        disp_ref = None
        if reference is not None:
            disp_ref = reference.meta.get("step_displacement")
            if isinstance(disp_ref, dict):
                self._u_ideal_disp.append(float(disp_ref["total"]))
                self._u_ideal_disp_lat.append(float(disp_ref["latent"]))
            if len(rows) > 1:
                d_c = float(unit.analyzer.dist2(reference.xi_hat[rows[-1]][None], reference.xi_hat[rows[0]][None])[0])
                self._u_ideal_disp_commit.append(d_c)
        # --- realized yardstick: how far the *committed* composition itself moves per commit step
        xi_last = xi_rows[fm][-1]
        if self._last_commit_xi is not None:
            self._u_real_disp.append(float(unit.analyzer.dist2(xi_last[None], self._last_commit_xi[None])[0]))
        self._last_commit_xi = xi_last.copy()
        self.step_traces.append({"unit": unit.index, "rows": int(len(rows)), "rho": rho, "z_committed_mean": z_bar.tolist(),
                                 "decoder_residual_mean_dist2": resid, "tanh_saturated_fraction": float((tz > SAT_HIGH).mean()),
                                 "tanh_underused_fraction": float((tz < SAT_LOW).mean()),
                                 "mean_abs_dz_per_row": dz_row, "mean_abs_dz_per_model_step": dz_step,
                                 "ideal_step_displacement_dist2": disp_ref,
                                 "realized_step_displacement_dist2": (self._u_real_disp[-1] if self._u_real_disp else None),
                                 "reference_z0": (reference.meta.get("z0") if reference is not None else None)})
        return {"rho": rho, "z_committed_mean": z_bar.tolist(), "decoder_residual": resid,
                "tanh_saturated_fraction": float((tz > SAT_HIGH).mean()),
                "tanh_underused_fraction": float((tz < SAT_LOW).mean()),
                "tanh_saturated_per_factor": sat.tolist(), "tanh_underused_per_factor": low.tolist(),
                "mean_abs_dz_per_model_step": dz_step, "ideal_step_displacement_dist2": disp_ref}

    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        zs = np.concatenate(self.unit_fits, axis=0) if self.unit_fits else np.zeros((0, self.d_z))
        st = dict(history.mode_state.get("vae", {}))
        st["mu_H"] = self.mu_H.tolist()
        st["Sigma_H"] = self.Sigma_H.tolist()
        st["basis"] = {"V": self.V.tolist(), "U": self.U.tolist(), "s": self.s.tolist(),
                       "loadings": self.loadings.tolist(), "hash": self.basis_hash,
                       "z_mean": self.basis_z_mean.tolist(), "fragment": bool(self.frag),
                       "fragment_basis": self.frag_basis_info,
                       "material_ids": list(range(1, self.M))}
        st.setdefault("path_summaries", []).append({"unit": unit.index, "z_mean": (zs.mean(axis=0).tolist() if len(zs) else None),
                                                    "z_first": (zs[0].tolist() if len(zs) else None),
                                                    "z_last": (zs[-1].tolist() if len(zs) else None), "n_rows": int(len(zs))})
        st["path_summaries"] = st["path_summaries"][-8:]
        history.mode_state["vae"] = st
        n = max(1, self._u_rows)
        stats = {
            "unit": unit.index, "fragment_mode": bool(self.frag), "basis_hash": self.basis_hash,
            "committed_rows_fitted": int(self._u_rows), "commit_steps": int(len(self._u_resid)),
            "internal_iterations": {"reference_draws": int(self.unit_step_count), "commit_observations": int(len(self._u_resid)),
                                    "note": "internal iterations, not musical time; musical time is commits x commit_seconds"},
            "tanh_saturated_fraction": float(self._u_sat.sum() / (n * self.d_z)),
            "tanh_underused_fraction": float(self._u_low.sum() / (n * self.d_z)),
            "tanh_saturated_per_factor": (self._u_sat / n).tolist(),
            "tanh_underused_per_factor": (self._u_low / n).tolist(),
            "mean_abs_tanh_per_factor": (self._u_tanh / n).tolist(),
            "decoder_residual_mean_dist2": float(np.mean(self._u_resid)) if self._u_resid else None,
            "mean_abs_dz_per_observation_row": float(np.mean(self._u_dz_row)) if self._u_dz_row else None,
            "mean_abs_dz_per_model_step": float(np.mean(self._u_dz_step)) if self._u_dz_step else None,
            "ideal_mean_step_displacement_dist2": float(np.mean(self._u_ideal_disp)) if self._u_ideal_disp else None,
            "ideal_mean_step_displacement_latent_dist2": (float(np.mean(self._u_ideal_disp_lat))
                                                          if self._u_ideal_disp_lat else None),
            "ideal_mean_committed_step_displacement_dist2": (float(np.mean(self._u_ideal_disp_commit))
                                                             if self._u_ideal_disp_commit else None),
            "realized_mean_step_displacement_dist2": (float(np.mean(self._u_real_disp))
                                                      if self._u_real_disp else None),
            "displacement_note": "total = D(z_{j+1};t_{j+1}) vs D(z_j;t_j) and therefore also contains the motion of "
                                 "xi_G(t) itself, which the decoder inherits and no gain choice can remove (on "
                                 "dynamic goal material it dominates); latent = the part the controller commands, "
                                 "measured at a frozen xi_G; realized = the committed composition's own motion per "
                                 "commit step (the realizability yardstick)",
            "state_origin": self.state_origin,
            "mu_H_after": self.mu_H.tolist(), "Sigma_H_after": self.Sigma_H.tolist(),
            "Sigma_H_effective_after": self._sigma_effective().tolist(),
        }
        self.unit_stats.append(stats)
        self.fit_traces.append({"unit": unit.index, "z_actual_mean": (zs.mean(axis=0).tolist() if len(zs) else None),
                                "z_actual_cov": (np.cov(zs.T).tolist() if len(zs) > 1 else None),
                                "target_z_mean_free": chosen.target.meta.get("z_mean_free")})
        self.unit_traces.append({"unit": unit.index, "chosen_target": chosen.target.id,
                                 "normalized_mode_error": chosen.normalized_mode_error,
                                 "mu_H_after": self.mu_H.tolist(), "Sigma_H_after": self.Sigma_H.tolist(),
                                 "statistics": stats})

    def signature(self) -> np.ndarray:
        return np.concatenate([self.mu_H, self.Sigma_H.ravel(), self.mean_z[self.unit.free_mask].mean(axis=0)
                               if self.unit.free_mask.any() else np.zeros(self.d_z)])

    def trace(self) -> Dict[str, Any]:
        n = max(1, self.tanh_rows)
        return {"latent_dim": self.d_z, "latent_correlation_seconds": self.tau_z, "model_step_seconds": self.model_step,
                "latent_path": {"where": "Ornstein-Uhlenbeck on the model-step knots, linearly interpolated onto the "
                                         "observation rows (hop {:.3f} s)".format(self.hop_s),
                                "ar_coefficient_per_model_step": ar1_rho_for(self.model_step, self.tau_z)},
                "fragment_mode": bool(self.frag), "fragment_basis": self.frag_basis_info,
                "sigma_H_floor_fragment": (self.sigma_H_floor_frag if self.frag else None),
                "tanh_usage_committed": {"rows": int(self.tanh_rows), "thresholds": {"saturated_above": SAT_HIGH,
                                                                                     "underused_below": SAT_LOW},
                                         "saturated_fraction": float(self.sat_counts.sum() / (n * self.d_z)),
                                         "underused_fraction": float(self.low_counts.sum() / (n * self.d_z)),
                                         "saturated_per_factor": (self.sat_counts / n).tolist(),
                                         "underused_per_factor": (self.low_counts / n).tolist(),
                                         "mean_abs_tanh_per_factor": (self.tanh_sum / n).tolist()},
                "unit_statistics": self.unit_stats,
                "latent_means_covariances": self.latent_traces, "basis_definition": self.basis_traces,
                "fitted_actual_latent_summary": self.fit_traces, "decoder": "xi_G + sum_k s_k tanh(z_k) v_k",
                "warnings": list(self.warnings), "units": self.unit_traces, "steps": self.step_traces[-64:]}

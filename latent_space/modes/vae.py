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
# Hold mode (docs/HOLD_CONTRACT.md R4): the latent innovation of a held path is amplified by this
# gain before `hold_move_scale` (default 1.0) scales it further.  Calibrated on dev/fixture_hold.json
# so that the engine-measured hold_summary.requested_over_flutter_open lands in the contract's 3-6
# band at hold_move_scale = 1.0; on other materials the main session recalibrates with that key.
HOLD_INNOVATION_GAIN = 2.5


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
        # --- hold mode (docs/HOLD_CONTRACT.md): the latent path is decoded onto the REALIZABLE set.
        # All read with .get so no shared config file has to change (report lists them).
        self.move_scale = float(self.p.get("hold_move_scale", 1.0))
        self.fit_rows = max(1, min(8, int(self.p.get("hold_fit_rows", 5))))
        self.level_step0 = float(self.p.get("hold_level_step", 0.25))
        self.level_min_step = float(self.p.get("hold_level_min_step", 0.05))
        self.level_evals = max(0, int(self.p.get("hold_level_evals", 20)))
        self.n_frag_cand = max(0, int(self.p.get("hold_jump_candidates", 2)))
        self.jump_margin = float(self.p.get("hold_jump_margin", 0.03))
        self.step_margin = float(self.p.get("hold_step_margin", 0.0))
        self.target_lead = max(0, int(self.p.get("hold_target_lead_rows", 3)))
        self.plan_calls_max = max(1, int(self.p.get("hold_plan_calls_max", 160)))
        # --- control-space latent (docs/FIDELITY_CONTRACT.md "VAE (B)"): the latent space is
        # re-defined over the CONTROLS the realizer actually has, so that every decoded point is
        # realizable and the xi-space projection loss disappears by construction.
        self.latent_space = str(self.p.get("hold_latent_space", "controls"))    # "controls" | "xi"
        self.k_c = max(1, int(self.p.get("control_latent_dim", 6)))
        self.r_pca = max(1, int(self.p.get("control_fragment_pca", 2)))
        self.n_states = max(64, int(self.p.get("control_basis_states", 1500)))
        self.n_holdout = max(8, int(self.p.get("control_holdout_states", 200)))
        self.control_offsets = max(2, int(self.p.get("control_basis_offsets", 3)))
        self.control_seed_offset = int(self.p.get("control_seed_offset", 5521))
        self.jump_min_dist = float(self.p.get("control_jump_min_pca_dist", 0.6))
        self.max_jumps = max(0, int(self.p.get("control_max_jumps_per_step", 3)))
        self.sigma_C_floor = float(self.p.get("control_sigma_floor", 0.30))
        # control layout: [level of track 1..M-1 | r PCA coords of the fragment of track 1..M-1]
        self.n_ctrl_tracks = max(0, self.M - 1)
        self.d_u = self.n_ctrl_tracks * (1 + self.r_pca)
        self.k_c = min(self.k_c, max(1, self.d_u))
        self._u_pca = {i: slice(self.n_ctrl_tracks + (i - 1) * self.r_pca,
                                self.n_ctrl_tracks + i * self.r_pca) for i in range(1, self.M)}
        self._pca_cache: Dict[int, Dict[str, Any]] = {}
        self.control_ready = False
        self.control_basis_info: Optional[Dict[str, Any]] = None
        self.control_traces: List[Dict[str, Any]] = []
        self.hold_traces: List[Dict[str, Any]] = []
        self.hold_count = 0
        self.plan_calls_total = 0
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

    def _refresh_LC(self) -> None:
        """Control latent: the diagonal is floored (`control_sigma_floor`) so the ideal keeps
        exploring tanh's range, and `hold_move_scale` scales the innovation amplitude (R4)."""
        S = np.array(self.Sigma_C, dtype=np.float64, copy=True)
        np.fill_diagonal(S, np.maximum(np.diag(S), self.sigma_C_floor))
        self.Sigma_C_used = S
        self.L_C = self.move_scale * np.linalg.cholesky(S + np.eye(self.k_c) * self.cov_floor)

    def begin_unit(self, unit: UnitContext, history) -> None:
        if not self.basis_ready:
            self._build_basis(history)
        if self.latent_space == "controls" and not self.control_ready and self._fragment_available():
            self._build_control_basis(history)
        if self.control_ready:
            stc = history.mode_state.get("vae") or {}
            if "mu_C" in stc and len(stc["mu_C"]) == self.k_c:
                self.mu_C = np.array(stc["mu_C"], dtype=np.float64)
                self.Sigma_C = np.array(stc["Sigma_C"], dtype=np.float64)
                self.control_state_origin = "inherited from history.mode_state['vae']"
            else:
                self.mu_C = np.zeros(self.k_c)
                self.Sigma_C = np.eye(self.k_c) * max(float(self.p["sigma_H_init"]), self.sigma_C_floor)
                self.control_state_origin = "re-initialised (no control state in history)"
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
        if self.control_ready:
            self.chi_c = unit.chi[:, 1:] @ self.loadings_c[1:]
            self.mean_zc = unit.o[:, None] * (self.mu_C[None, :] + self.K_F * self.chi_c)
            self._refresh_LC()
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
        # --- hold mode: projection of the decoder path onto the realizable plan set
        self.unit_hold_count = 0
        self._u_proj_resid: List[float] = []
        self._u_proj_share: List[float] = []
        self._u_req_plan: List[float] = []
        self._u_req_dec: List[float] = []
        self._u_plan_calls: List[int] = []
        self._u_plan_steps: List[int] = []
        self._u_jumps_planned = 0
        self._u_jumps_dropped = 0
        self._u_level_moves: List[float] = []
        # --- control-space latent statistics
        self._u_ctrl_real: List[float] = []
        self._u_ctrl_jump_legal: List[float] = []
        self._u_enc_share: List[float] = []
        self._u_track_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self._u_ctrl_tanh: List[np.ndarray] = []
        self._hold_open: Optional[Dict[str, Any]] = None
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

    def _ou_path(self, unit: UnitContext, rows: np.ndarray, z0: np.ndarray,
                 L: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Latent Ornstein-Uhlenbeck path **at the model step**:

            z_{j+1} = m_{j+1} + a (z_j - m_j) + sqrt(1 - a^2) o L eps,   a = exp(-model_step / tau_z)

        drawn on the model-step knots and linearly interpolated onto the observation rows (hop
        0.1 s), so the ideal changes by one realizable amount per commit step instead of wiggling
        at the observation hop.  With model_step == hop (the baseline configs) the knots are the
        rows and this reduces exactly to the previous per-row recursion.  `L` overrides the
        Cholesky factor of Sigma_H (hold mode scales it by `hold_move_scale`); openness keeps
        gating the innovation, so the ideal still stands still near the goal."""
        kn = self._knots(len(rows))
        m = self.mean_z[rows]
        o = unit.o[rows]
        Lm = self.L if L is None else L
        zk = np.zeros((len(kn), self.d_z))
        zk[0] = z0
        for j in range(1, len(kn)):
            dt = (kn[j] - kn[j - 1]) * self.hop_s
            a = ar1_rho_for(dt, self.tau_z)
            eps = self.rng.standard_normal(self.d_z)
            zk[j] = m[kn[j]] + a * (zk[j - 1] - m[kn[j - 1]]) + np.sqrt(max(0.0, 1.0 - a * a)) * o[kn[j]] * (Lm @ eps)
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
        rs = getattr(unit, "realizer_state", None)
        if rs is not None:                       # hold mode (docs/HOLD_CONTRACT.md); legacy below
            if self._control_available():        # FIDELITY_CONTRACT B: latent space over the controls
                return self._prepare_reference_controls(unit, rows, xi_current, rs, n_proposals)
            return self._prepare_reference_hold(unit, rows, xi_current, z0, rs, n_proposals)
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

    def _ou_path_control(self, unit: UnitContext, rows: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The same Ornstein-Uhlenbeck recursion at the model step, in the CONTROL latent space
        (mean mu_C + K_F chi, spread Sigma_C, gated by the openness o as before)."""
        kn = self._knots(len(rows))
        m = self.mean_zc[rows]
        o = unit.o[rows]
        zk = np.zeros((len(kn), self.k_c))
        zk[0] = z0
        for j in range(1, len(kn)):
            a = ar1_rho_for((kn[j] - kn[j - 1]) * self.hop_s, self.tau_z)
            eps = self.rng.standard_normal(self.k_c)
            zk[j] = m[kn[j]] + a * (zk[j - 1] - m[kn[j - 1]]) + np.sqrt(max(0.0, 1.0 - a * a)) * o[kn[j]] * (self.L_C @ eps)
        if len(kn) == len(rows):
            z = zk
        else:
            xs = np.arange(len(rows), dtype=np.float64)
            z = np.stack([np.interp(xs, kn.astype(np.float64), zk[:, c]) for c in range(self.k_c)], axis=1)
        z[unit.hold_mask[rows]] = 0.0
        return z, kn

    def _prepare_reference_controls(self, unit: UnitContext, rows: np.ndarray, xi_anchor: np.ndarray,
                                    rs: Dict[str, Any], n_proposals: int) -> List[Target]:
        """A latent space of the REACHABLE set (docs/FIDELITY_CONTRACT.md "VAE (B)").

        z lives over the controls u = [levels | fragment PCA coords].  The OU path is drawn on the
        model-step knots from z0 = Pᵀ(u_current − ū); every knot decodes to a legal control state
        (levels clipped to [0,1], nearest fragment per track), which becomes one plan step -- a jump
        only where the minimum clip allows and only when the wanted fragment differs clearly from
        the one playing.  The published rows are the exact `plan_rows` of that plan, so the decoder
        target and the plan are the same object: the xi-space projection loss is zero by
        construction and what remains to measure is `control_realization` and `latent_tracking`."""
        an = self.analyzer
        t0, commit = int(rs["frame"]), int(rs["commit_frames"])
        hold_f, min_clip = int(rs["hold_frames"]), int(rs["min_clip_frames"])
        search_end = int(rs["search_end_frame"])
        jumps_enabled = bool(rs["jumps_enabled"])
        plan_eval = rs["plan_rows"]
        lv_start = np.asarray(rs["levels"], dtype=np.float64)
        pos_start = np.asarray(rs["positions"], dtype=np.int64)
        next_jump = [int(x) for x in rs["next_jump_frame"]]
        centers_rows = np.asarray(unit.centers)[rows]
        free = unit.free_mask[rows]
        L_src = [int(x) for x in an.source_lengths]
        blocks: List[Tuple[int, np.ndarray]] = []
        for k in range(max(1, int(round(hold_f / float(commit))))):
            f = int(t0 + k * commit)
            if f >= search_end:
                break
            blk = np.where((centers_rows >= f) & (centers_rows < f + commit))[0]
            if len(blk) == 0:
                break
            blocks.append((f, blk))
        u0 = self._controls(lv_start, pos_start)
        z0 = self._encode_controls(u0)
        # how much of u0 the k axes can represent at all (the encoder is a projection)
        us0 = (u0 - self.u_mean) / self.u_std
        enc_share = float(1.0 - ((us0 - self.P @ (self.P.T @ us0)) ** 2).sum() / max(1e-30, (us0 ** 2).sum()))
        out: List[Target] = []
        for a_ in range(max(1, n_proposals)):
            z, kn = self._ou_path_control(unit, rows, z0)
            steps: List[Dict[str, Any]] = []
            step_logs: List[Dict[str, Any]] = []
            lv_prev = lv_start.copy()
            plan_pos = {i: (int(pos_start[i]), t0) for i in range(1, self.M)}   # (position, its frame)
            plan_last_jump = {i: -(1 << 60) for i in range(1, self.M)}
            want_sum = got_sum = 0.0
            want_lvl = got_lvl = 0.0
            n_wanted = n_legal = 0
            z_target = z0
            for (f, blk) in blocks:
                z_target = z[min(int(blk[-1]) + 1, len(rows) - 1)]     # the knot this block settles on
                levels, frag, u_t = self._decode_controls(z_target)
                levels[0] = lv_prev[0]
                # fragment each track would play at f without a jump, in the same PCA coordinates
                wants: List[Tuple[float, int]] = []
                cur_coord: Dict[int, np.ndarray] = {}
                for i in range(1, self.M):
                    p = self._frag_pca(i)
                    p0, f0 = plan_pos[i]
                    k_now = an.fragment_index_at(i, int((p0 + (f - f0)) % max(1, L_src[i])))
                    cur_coord[i] = p["coords"][k_now]
                    d = float(np.sqrt(((p["coords"][frag[i]] - cur_coord[i]) ** 2).sum()))
                    legal = (jumps_enabled and f >= next_jump[i] and f - plan_last_jump[i] >= min_clip)
                    if d >= self.jump_min_dist:
                        n_wanted += 1
                        if legal:
                            n_legal += 1
                            wants.append((d, i))
                    want_sum += d * d
                    want_lvl += (float(u_t[i - 1]) - lv_prev[i]) ** 2      # before the [0,1] clip
                    got_lvl += (levels[i] - lv_prev[i]) ** 2
                wants.sort(key=lambda x: -x[0])
                jumps: Dict[int, int] = {}
                for (d, i) in wants[: self.max_jumps]:
                    p = self._frag_pca(i)
                    jumps[i] = int(p["pos"][frag[i]])
                    plan_pos[i] = (jumps[i], f)
                    plan_last_jump[i] = f
                    got_sum += d * d
                steps.append({"frame": int(f), "jumps": jumps, "levels": levels.tolist()})
                step_logs.append({"frame": int(f), "jumps": {int(k_): int(v_) for k_, v_ in jumps.items()},
                                  "tracks_wanting_a_jump": len(wants), "jumps_applied": len(jumps),
                                  "level_change_max": float(np.abs(levels[1:] - lv_prev[1:]).max()) if self.M > 1 else 0.0,
                                  "z": z_target.tolist()})
                lv_prev = levels
            # ---- exact rows of the plan (published) and of doing nothing (the fidelity yardstick)
            xi_rows, _parts, info = plan_eval(steps, rows)
            xi_stay, _ps, _is = plan_eval([], rows)
            calls = 2
            hm = unit.hold_mask[rows]
            if hm.any():
                xi_rows = np.array(xi_rows, copy=True)
                xi_rows[hm] = unit.xi_goal[rows][hm]
            xi_hat = unit.xi_goal.copy()
            xi_hat[rows] = xi_rows
            m_ = free if free.any() else np.ones(len(rows), dtype=bool)
            anchor = np.asarray(xi_anchor, dtype=np.float64)[None, :]
            end_blk = blocks[-1][1] if blocks else np.arange(len(rows))[-1:]
            req = float(an.dist2(xi_rows[end_blk].mean(axis=0)[None, :], anchor)[0])
            stay_req = float(an.dist2(xi_stay[end_blk].mean(axis=0)[None, :], anchor)[0])
            ctrl_real = float((got_sum + got_lvl) / max(1e-30, want_sum + want_lvl))
            n_jumps = int(sum(len(s["jumps"]) for s in steps))
            self.hold_count += 1
            self.unit_hold_count += 1
            self.plan_calls_total += calls
            self._u_ctrl_real.append(ctrl_real)
            self._u_ctrl_jump_legal.append(float(n_legal / max(1, n_wanted)))
            self._u_enc_share.append(enc_share)
            self._u_req_plan.append(req)
            self._u_plan_calls.append(calls)
            self._u_plan_steps.append(len(steps))
            self._u_jumps_planned += n_jumps
            self._u_jumps_dropped += len(info.get("dropped_jumps") or [])
            self._u_proj_resid.append(0.0)
            self._u_proj_share.append(0.0)
            self._u_req_dec.append(req)
            if a_ == 0:            # latent tracking is closed when the next hold opens / the unit ends
                self._close_hold_track()
                self._hold_open = {"z0": z0.copy(), "z_req": np.asarray(z_target, dtype=np.float64).copy(),
                                   "z_last": None, "unit": unit.index, "hold": int(self.unit_hold_count)}
            hold_tr = {"unit": unit.index, "hold": int(self.unit_hold_count), "proposal": int(a_),
                       "t0_seconds": t0 / float(self.fs), "rows": [int(rows[0]), int(rows[-1])],
                       "openness": float(unit.o[rows[0]]), "plan_steps": int(len(steps)),
                       "plan_rows_calls": int(calls), "candidate_plans_evaluated": int(calls),
                       "control_realization": ctrl_real,
                       "jump_legal_share": float(n_legal / max(1, n_wanted)),
                       "encoder_control_share_captured": enc_share,
                       "requested_dist2_at_hold_end": req, "stay_dist2_at_hold_end": stay_req,
                       "jumps_planned": n_jumps, "jumps_dropped_by_min_clip": len(info.get("dropped_jumps") or []),
                       "requested_dz_norm": float(np.linalg.norm(z_target - z0)),
                       "steps": step_logs,
                       "note": "candidate plans are internal iterations, not musical time; the decoded control "
                               "state IS the plan, so no candidate search is needed"}
            self.hold_traces.append(hold_tr)
            out.append(Target(f"vae:u{unit.index}:chold{self.unit_step_count}:{a_}", xi_hat,
                              meta={"plan": steps, "latent_space": "controls", "z0": z0.tolist(),
                                    "z_target": np.asarray(z_target).tolist(), "basis_hash": self.basis_hash,
                                    "tau_z_seconds": self.tau_z, "model_step_seconds": self.model_step,
                                    "ar_coefficient": ar1_rho_for(self.model_step, self.tau_z),
                                    "hold_move_scale": self.move_scale, "control_realization": ctrl_real,
                                    "projection_residual_dist2": 0.0,
                                    "law_fidelity": 1.0,
                                    "law_fidelity_note": "1 by construction: the decoded control state IS the "
                                                         "published plan, so d2(plan rows, law target) = 0; the "
                                                         "residual fidelity is control_realization",
                                    "requested_dist2_projected_plan": req,
                                    "requested_dist2_decoder_path": req,
                                    "candidate_plans_evaluated": int(calls), "plan_steps": int(len(steps)),
                                    "jumps_planned": n_jumps,
                                    "step_displacement": self._step_displacement_rows(unit, xi_rows, rows, kn)}))
        self.unit_step_count += 1
        self.hold_traces = self.hold_traces[-96:]
        return out

    def _close_hold_track(self) -> None:
        """`latent_tracking`: pair the Δz a hold requested with the Δz the committed sound made,
        both measured with the same control encoder."""
        h = getattr(self, "_hold_open", None)
        if h is not None and h.get("z_last") is not None:
            self._u_track_pairs.append((h["z_req"] - h["z0"], h["z_last"] - h["z0"]))
        self._hold_open = None

    def _step_displacement_rows(self, unit: UnitContext, xi_rows: np.ndarray, rows: np.ndarray,
                                kn: np.ndarray) -> Dict[str, float]:
        """Per-model-step displacement of a published ideal when the latent coordinates are not in
        xi-space (control mode): only the total is defined there."""
        if len(kn) < 2:
            return {"total": 0.0, "latent": 0.0}
        fm = unit.free_mask[rows][kn[1:]] & unit.free_mask[rows][kn[:-1]]
        if not fm.any():
            return {"total": 0.0, "latent": 0.0}
        tot = unit.analyzer.dist2(xi_rows[kn[1:]], xi_rows[kn[:-1]])
        return {"total": float(tot[fm].mean()), "latent": 0.0}

    # =================================================== control space (FIDELITY_CONTRACT.md "VAE (B)")
    def _control_available(self) -> bool:
        return (self.latent_space == "controls" and self._fragment_available()
                and self.n_ctrl_tracks > 0 and getattr(self.analyzer, "solo_pos", None) is not None)

    def _frag_pca(self, track: int) -> Dict[str, Any]:
        """PCA of one source's clip-averaged fragment features (r standardized axes).  Tracks that
        share a source share `analyzer.solo_f` (second voices of the same material), so they share
        one PCA and one coordinate array -- the cache is keyed on that array's identity."""
        an = self.analyzer
        key = id(an.solo_f[track])
        hit = self._pca_cache.get(key)
        if hit is not None:
            return hit
        F = np.asarray(an.frag_f[track], dtype=np.float64)
        mu = F.mean(axis=0)
        X = F - mu[None, :]
        _u, sv, Vt = np.linalg.svd(X, full_matrices=False)
        r0 = int(min(self.r_pca, Vt.shape[0]))
        C = np.zeros((len(F), self.r_pca))
        C[:, :r0] = X @ Vt[:r0].T
        sd = np.ones(self.r_pca)
        sd[:r0] = np.maximum(C[:, :r0].std(axis=0), 1e-9)
        ok = np.asarray(an.frag_E[track]) >= an.silence_energy
        if not ok.any():
            ok = np.ones(len(F), dtype=bool)
        out = {"coords": C / sd[None, :], "V": Vt[:r0], "mu": mu, "sd": sd, "ok": ok, "axes": r0,
               "pos": np.asarray(an.solo_pos[track]),
               "variance_share": float((sv[:r0] ** 2).sum() / max(1e-300, float((sv ** 2).sum())))}
        self._pca_cache[key] = out
        return out

    def _controls(self, levels: Sequence[float], positions: Sequence[int]) -> np.ndarray:
        """u = [level of every material track | r PCA coords of the fragment it plays]."""
        an = self.analyzer
        u = np.zeros(self.d_u)
        for i in range(1, self.M):
            p = self._frag_pca(i)
            u[i - 1] = float(levels[i])
            u[self._u_pca[i]] = p["coords"][an.fragment_index_at(i, int(positions[i]))]
        return u

    def _encode_controls(self, u: np.ndarray) -> np.ndarray:
        """z = atanh( Pᵀ (u − ū) / s ) in standardized control coordinates (eq. 34 analogue)."""
        us = (np.asarray(u, dtype=np.float64) - self.u_mean) / self.u_std
        w = np.clip((self.P.T @ us) / self.s_c, -0.999, 0.999)
        return np.arctanh(w)

    def _decode_controls(self, z: np.ndarray):
        """u(z) = ū + P (s ⊙ tanh z) -> clipped levels + the nearest fragment of every track.
        Every decoded point is a legal (levels, fragments) state: nothing is projected afterwards."""
        u = self.u_mean + self.u_std * (self.P @ (self.s_c * np.tanh(np.asarray(z, dtype=np.float64))))
        levels = np.zeros(self.M)
        frag = np.zeros(self.M, dtype=np.int64)
        for i in range(1, self.M):
            levels[i] = float(np.clip(u[i - 1], 0.0, 1.0))
            p = self._frag_pca(i)
            d = ((p["coords"] - u[self._u_pca[i]][None, :]) ** 2).sum(axis=1)
            d[~p["ok"]] += 1e6
            frag[i] = int(np.argmin(d))
        return levels, frag, u

    def _build_control_basis(self, history) -> None:
        """PLS-SVD of standardized random realizable CONTROLS against their exact compositions:
        the k control directions that explain the most composition variance.  Fixed for the job."""
        an = self.analyzer
        st = history.mode_state.get("vae") if hasattr(history, "mode_state") else None
        cb = (st or {}).get("control_basis")
        if cb and int(cb.get("d_u", -1)) == self.d_u and int(cb.get("k", -1)) == self.k_c:
            self.P = np.array(cb["P"], dtype=np.float64)
            self.s_c = np.array(cb["s"], dtype=np.float64)
            self.u_mean = np.array(cb["u_mean"], dtype=np.float64)
            self.u_std = np.array(cb["u_std"], dtype=np.float64)
            self.loadings_c = np.array(cb["loadings"], dtype=np.float64)
            self.control_basis_info = cb.get("info")
            self.control_origin = "restored_from_history"
            self.control_ready = True
            return
        seed = (int(self.cfg.get("seed", 0)) + self.control_seed_offset) % (2 ** 32)
        rng = np.random.default_rng(seed)
        offsets = np.arange(self.control_offsets, dtype=np.int64) * int(an.hop)
        n_fit, n_test = int(self.n_states), int(self.n_holdout)
        U = np.zeros((n_fit + n_test, self.d_u))
        XI = np.zeros((n_fit + n_test, self.d_xi))
        keep: List[Tuple[np.ndarray, np.ndarray]] = []
        for j in range(n_fit + n_test):
            xi, _p, meta = an.random_fragment_composition(self.sources, rng, offsets, goal_level=0.0)
            lv = np.asarray(meta["levels"], dtype=np.float64)
            po = np.asarray(meta["positions"], dtype=np.int64)
            U[j] = self._controls(lv, po)
            XI[j] = xi.mean(axis=0)
            if j >= n_fit:
                keep.append((lv, po))
        Uf, Xf = U[:n_fit], XI[:n_fit]
        self.u_mean = Uf.mean(axis=0)
        self.u_std = np.maximum(Uf.std(axis=0), 1e-6)
        Us = (Uf - self.u_mean) / self.u_std
        Y = (Xf - Xf.mean(axis=0)) * self.sqrtW[None, :]
        Cxy = Us.T @ Y / float(n_fit)                       # (d_u, d_xi) cross-covariance
        A, sv, _Bt = np.linalg.svd(Cxy, full_matrices=False)
        k = int(min(self.k_c, A.shape[1]))
        P = A[:, :k].copy()
        for c in range(k):                                  # deterministic sign: largest loading > 0
            j = int(np.argmax(np.abs(P[:, c])))
            if P[j, c] < 0:
                P[:, c] = -P[:, c]
        self.P = P
        proj = Us @ P
        q = np.percentile(np.abs(proj), self.scale_quantile, axis=0)
        self.s_c = np.maximum(q / max(1e-6, self.scale_target_tanh), 1e-6)
        # chi conditioning: "how far does raising material i push factor c" is literally P's level row
        self.loadings_c = np.zeros((self.M, k))
        self.loadings_c[1:] = P[: self.n_ctrl_tracks] / self.u_std[: self.n_ctrl_tracks, None]
        self.k_c = k
        # ---- held-out: encode -> decode -> EXACT composition of the decoded realizable state
        xbar = XI[n_fit:].mean(axis=0)
        num = den = 0.0
        lat = []
        for j, (lv, po) in enumerate(keep):
            z = self._encode_controls(U[n_fit + j])
            lv_d, fr_d, _u = self._decode_controls(z)
            pos_d = np.array([0] + [int(self._frag_pca(i)["pos"][fr_d[i]]) for i in range(1, self.M)],
                             dtype=np.int64)
            lv_d[0] = 0.0
            xi_d, _pp = an.fragment_composition(self.sources, pos_d, lv_d, offsets)
            num += float(an.dist2(xi_d.mean(axis=0)[None, :], XI[n_fit + j][None, :])[0])
            den += float(an.dist2(XI[n_fit + j][None, :], xbar[None, :])[0])
            lat.append(z)
        var_expl = float(1.0 - num / max(1e-300, den))
        lat = np.asarray(lat)
        tanh_abs = np.abs(np.tanh(proj / self.s_c[None, :]))
        info = {
            "kind": "PLS-SVD of standardized random realizable controls vs exact compositions",
            "control_layout": {"levels": f"tracks 1..{self.M - 1}", "fragment_pca_axes_per_track": self.r_pca,
                               "control_dimension": int(self.d_u)},
            "k": int(k), "states_fit": n_fit, "states_holdout": n_test, "seed": int(seed),
            "singular_values": sv[: min(len(sv), 8)].tolist(),
            "cross_covariance_share_captured_by_k": float((sv[:k] ** 2).sum() / max(1e-300, float((sv ** 2).sum()))),
            "holdout_composition_variance_explained_through_exact_decoder": var_expl,
            "holdout_mean_dist2_decoded_vs_true": float(num / max(1, len(keep))),
            "holdout_mean_dist2_true_vs_mean": float(den / max(1, len(keep))),
            "scales": self.s_c.tolist(),
            "basis_rows_saturated_fraction": float((tanh_abs > SAT_HIGH).mean()),
            "basis_rows_underused_fraction": float((tanh_abs < SAT_LOW).mean()),
            "latent_std_of_holdout": (lat.std(axis=0).tolist() if len(lat) else None),
            "fragment_pca_variance_share": {str(i): self._frag_pca(i)["variance_share"] for i in range(1, self.M)},
            "axis_loadings": {
                "level_block_per_axis": [P[: self.n_ctrl_tracks, c].tolist() for c in range(k)],
                "level_block_norm_share": [float((P[: self.n_ctrl_tracks, c] ** 2).sum()) for c in range(k)],
                "fragment_block_norm_share": [float((P[self.n_ctrl_tracks:, c] ** 2).sum()) for c in range(k)],
                "note": "a level_block_norm_share near 1 on one axis means that axis is 'overall level'"},
            "note": "u = [levels | fragment PCA coords]; the decoder maps z to a legal control state, so "
                    "the ideal is realizable by construction (no xi-space projection)",
        }
        self.control_basis_info = info
        self.control_origin = "built (PLS-SVD of random realizable controls)"
        self.control_ready = True
        self.control_traces.append(info)

    # ------------------------------------------------------- hold mode: decode onto the realizable set
    def _levels_from_decoder(self, D_blk: np.ndarray, gains: np.ndarray, e_rows: np.ndarray,
                             lv: np.ndarray) -> Optional[np.ndarray]:
        """Least-squares-style warm start for the level search: the material levels whose
        contributions c and total energy reproduce the decoder target when the cross terms of the
        mixture are ignored.  The per-track window energy at gain 1 is read back from the exact
        rows the plan evaluator just returned (e_i = g_i^2 diag_i), so no extra Gram work is
        needed; tracks that are silent in those rows keep their level (their diag is unknown)."""
        an = self.analyzer
        M = self.M
        c_t = np.clip(D_blk[:, self.d_phi: self.d_phi + M].mean(axis=0), 0.0, None)
        logE = float(D_blk[:, 0].mean()) * float(an.norm_std[0]) + float(an.norm_mean[0])
        E_t = float(np.expm1(min(60.0, logE))) * float(an.E_ref)
        if not np.isfinite(E_t) or E_t <= 0.0 or float(c_t[1:].sum()) <= 1e-9:
            return None
        out = np.asarray(lv, dtype=np.float64).copy()
        changed = False
        for i in range(1, M):
            m = gains[:, i] > 1e-6
            if not m.any():
                continue
            diag = float((e_rows[m, i] / (gains[m, i] ** 2)).mean())
            if not np.isfinite(diag) or diag <= 1e-30:
                continue
            out[i] = float(np.clip(np.sqrt(max(0.0, c_t[i] * E_t / diag)), 0.0, 1.0))
            changed = True
        return out if changed else None

    def _prepare_reference_hold(self, unit: UnitContext, rows: np.ndarray, xi_anchor: np.ndarray,
                                z0: np.ndarray, rs: Dict[str, Any], n_proposals: int) -> List[Target]:
        """The VAE law, decoded onto the realizable set (HOLD_CONTRACT R2/R3, "VAE" paragraph).

        Unchanged: the fragment SVD basis, the latent OU path on the model-step knots anchored at
        the encoding of the realized anchor, and the decoder D(z) = xi_G + sum_k s_k tanh(z_k) v_k.
        New: D(z) is no longer published directly.  For every model step of the hold (the commit
        frames `frame + k*commit_frames`) the plan step -- end levels of the four material tracks
        plus jumps of the tracks that may jump, to fragment candidates matched to D(z)'s band
        profile -- whose EXACT rows (`realizer_state["plan_rows"]`) are closest to D(z) in the
        composition metric is searched, and the exact rows of the resulting plan are published.
        d^2(plan rows, D(z) rows) is recorded as the projection residual: the part of the manifold
        move that no gain / position choice can produce.  The plan holds after its last step."""
        an = self.analyzer
        t0, commit = int(rs["frame"]), int(rs["commit_frames"])
        hold_f, min_clip = int(rs["hold_frames"]), int(rs["min_clip_frames"])
        search_end = int(rs["search_end_frame"])
        jumps_enabled = bool(rs["jumps_enabled"])
        plan_eval = rs["plan_rows"]
        lv_start = np.asarray(rs["levels"], dtype=np.float64)
        next_jump = [int(x) for x in rs["next_jump_frame"]]
        centers_rows = np.asarray(unit.centers)[rows]
        free = unit.free_mask[rows]
        n_steps = max(1, int(round(hold_f / float(commit))))
        blocks: List[Tuple[int, np.ndarray]] = []
        for k in range(n_steps):
            f = int(t0 + k * commit)
            if f >= search_end:
                break
            blk = np.where((centers_rows >= f) & (centers_rows < f + commit))[0]
            if len(blk) == 0:
                break
            blocks.append((f, blk))
        # innovation amplitude of the latent path: HOLD_INNOVATION_GAIN x `hold_move_scale` x
        # Cholesky(Sigma_H); openness still multiplies it row by row inside `_ou_path`
        gain = HOLD_INNOVATION_GAIN * self.move_scale
        Lm = self.L if gain == 1.0 else gain * self.L
        out: List[Target] = []
        for a_ in range(max(1, n_proposals)):
            z, kn = self._ou_path(unit, rows, z0, L=Lm)
            D = self.decode(unit, z, rows)                       # the free decoder path on the rows
            calls = 0
            budget = self.plan_calls_max
            steps: List[Dict[str, Any]] = []
            lv = lv_start.copy()
            plan_last_jump = {i: -(1 << 60) for i in range(self.M)}
            dropped_total = 0
            step_logs: List[Dict[str, Any]] = []

            def evaluate(plan, sub_abs, target):
                """d^2(exact plan rows, D(z) rows) on a small row subset -- one candidate plan."""
                xi_p, parts_p, info_p = plan_eval(plan, sub_abs)
                return float(an.dist2(xi_p, target).mean()), parts_p, info_p

            for (f, blk) in blocks:
                if budget <= 0:
                    break
                sub = blk if len(blk) <= self.fit_rows else blk[np.linspace(0, len(blk) - 1, self.fit_rows).astype(int)]
                sub_abs = rows[sub]
                # the plan's Q5 ramp only lands `ramp_seconds` after the commit frame, so the levels
                # of step k are fitted to where the decoder path will be by then (lead in rows)
                Y = D[np.minimum(sub + self.target_lead, len(rows) - 1)]
                step = {"frame": int(f), "jumps": {}, "levels": lv.tolist()}
                cost, parts0, info0 = evaluate(steps + [step], sub_abs, Y)
                calls += 1
                budget -= 1
                cost0 = cost
                pos_now = np.asarray(info0["positions"])
                # ---- jumps: fragment candidates for D(z)'s band profile on this block
                can_jump = jumps_enabled and getattr(an, "frag_f", None) is not None
                allowed = [i for i in range(1, self.M)
                           if can_jump and f >= next_jump[i] and f - plan_last_jump[i] >= min_clip]
                prof = Y[:, 1:1 + an.nb].mean(axis=0)
                n_jump_eval = 0
                for i in allowed:
                    if budget <= 0 or self.n_frag_cand <= 0:
                        break
                    cand = an.fragment_candidates(i, prof, self.n_frag_cand, exclude_near=int(pos_now[0, i]),
                                                  exclude_frames=int(2 * min_clip))
                    best_p, best_c = None, float("inf")
                    for p in cand:
                        if budget <= 0:
                            break
                        jm = dict(step["jumps"])
                        jm[int(i)] = int(p)
                        trial = {"frame": int(f), "jumps": jm, "levels": list(step["levels"])}
                        c_, _pt, _it = evaluate(steps + [trial], sub_abs, Y)
                        calls += 1
                        budget -= 1
                        n_jump_eval += 1
                        if c_ < best_c:
                            best_c, best_p = c_, int(p)
                    # a jump is kept only when it clearly lowers the distance to D(z)
                    if best_p is not None and best_c < cost * (1.0 - self.jump_margin):
                        step["jumps"][int(i)] = best_p
                        plan_last_jump[i] = f
                        cost = best_c
                # ---- levels: warm start + coordinate search on the four material end levels
                cost_jumps = cost
                cur = lv.copy()
                warm = self._levels_from_decoder(Y, np.asarray(info0["gains"]), np.asarray(parts0["e"]), lv)
                if warm is not None and budget > 0:
                    step["levels"] = warm.tolist()
                    c_, _pt, _it = evaluate(steps + [step], sub_abs, Y)
                    calls += 1
                    budget -= 1
                    if c_ < cost - 1e-12:
                        cost, cur = c_, warm
                step["levels"] = cur.tolist()
                stp = self.level_step0
                left = min(self.level_evals, budget)
                while left > 0 and stp >= self.level_min_step:
                    improved = False
                    for i in range(1, self.M):
                        for sign in (1.0, -1.0):
                            if left <= 0:
                                break
                            trial = cur.copy()
                            trial[i] = float(np.clip(trial[i] + sign * stp, 0.0, 1.0))
                            if abs(trial[i] - cur[i]) < 1e-9:
                                continue
                            step["levels"] = trial.tolist()
                            c_, _pt, _it = evaluate(steps + [step], sub_abs, Y)
                            calls += 1
                            left -= 1
                            budget -= 1
                            if c_ < cost - 1e-12:
                                cost, cur, improved = c_, trial, True
                    if not improved:
                        stp *= 0.5
                # a level change costs the sound its own fast motion (the ramp lives inside one commit
                # block): keep it only when it clearly lowers the distance to D(z).  Marginal moves are
                # held, so the latent drift accumulates into fewer, decisive steps.
                kept = bool(cost < cost_jumps * (1.0 - self.step_margin))
                if not kept:
                    cur, cost = lv.copy(), cost_jumps
                step["levels"] = cur.tolist()
                step_logs.append({"frame": int(f), "block_rows": int(len(blk)), "fit_rows": int(len(sub)),
                                  "jumps": {int(k_): int(v_) for k_, v_ in step["jumps"].items()},
                                  "jump_candidates_evaluated": int(n_jump_eval),
                                  "level_change_kept": kept,
                                  "level_change_max": float(np.abs(cur[1:] - lv[1:]).max()),
                                  "block_dist2_hold": float(cost0), "block_dist2_plan": float(cost)})
                self._u_level_moves.append(float(np.abs(cur[1:] - lv[1:]).max()))
                steps.append(step)
                lv = cur
            # ---- the chosen plan, evaluated once on ALL rows: these exact rows are published (R2)
            xi_rows, _parts, info = plan_eval(steps, rows)
            calls += 1
            dropped_total = len(info.get("dropped_jumps") or [])
            hm = unit.hold_mask[rows]
            if hm.any():
                xi_rows = np.array(xi_rows, copy=True)
                xi_rows[hm] = unit.xi_goal[rows][hm]
            xi_hat = unit.xi_goal.copy()
            xi_hat[rows] = xi_rows
            # ---- statistics: how much of the manifold move survives the projection
            m_ = free if free.any() else np.ones(len(rows), dtype=bool)
            resid = float(an.dist2(xi_rows[m_], D[m_]).mean())
            d_hold = float(an.dist2(D[m_], np.asarray(xi_anchor)[None, :]).mean())
            end_blk = blocks[-1][1] if blocks else np.arange(len(rows))[-1:]
            anchor = np.asarray(xi_anchor, dtype=np.float64)[None, :]
            req_plan = float(an.dist2(xi_rows[end_blk].mean(axis=0)[None, :], anchor)[0])
            req_dec = float(an.dist2(D[end_blk].mean(axis=0)[None, :], anchor)[0])
            n_jumps = int(sum(len(s["jumps"]) for s in steps))
            self.hold_count += 1
            self.unit_hold_count += 1
            self.plan_calls_total += calls
            self._u_proj_resid.append(resid)
            self._u_proj_share.append(float(resid / d_hold) if d_hold > 1e-12 else 0.0)
            self._u_req_plan.append(req_plan)
            self._u_req_dec.append(req_dec)
            self._u_plan_calls.append(int(calls))
            self._u_plan_steps.append(int(len(steps)))
            self._u_jumps_planned += n_jumps
            self._u_jumps_dropped += dropped_total
            hold_tr = {"unit": unit.index, "hold": int(self.unit_hold_count), "proposal": int(a_),
                       "t0_seconds": t0 / float(self.fs), "rows": [int(rows[0]), int(rows[-1])],
                       "openness": float(unit.o[rows[0]]), "plan_steps": int(len(steps)),
                       "candidate_plans_evaluated": int(calls), "plan_rows_calls": int(calls),
                       "projection_residual_dist2": resid,
                       "decoder_move_from_anchor_dist2": d_hold,
                       "projection_residual_share_of_decoder_move": self._u_proj_share[-1],
                       "requested_dist2_decoder_path": req_dec, "requested_dist2_projected_plan": req_plan,
                       "jumps_planned": n_jumps, "jumps_dropped_by_min_clip": int(dropped_total),
                       "steps": step_logs,
                       "note": "candidate plans are internal iterations, not musical time; musical time of "
                               "this hold is plan_steps x commit_seconds"}
            self.hold_traces.append(hold_tr)
            out.append(Target(f"vae:u{unit.index}:hold{self.unit_step_count}:{a_}", xi_hat,
                              meta={"plan": steps, "z0": z0.tolist(), "z_mean": z.mean(axis=0).tolist(),
                                    "basis_hash": self.basis_hash, "tau_z_seconds": self.tau_z,
                                    "model_step_seconds": self.model_step,
                                    "ar_coefficient": ar1_rho_for(self.model_step, self.tau_z),
                                    "hold_move_scale": self.move_scale,
                                    "projection_residual_dist2": resid,
                                    "projection_residual_share_of_decoder_move": self._u_proj_share[-1],
                                    "requested_dist2_decoder_path": req_dec,
                                    "requested_dist2_projected_plan": req_plan,
                                    "candidate_plans_evaluated": int(calls), "plan_steps": int(len(steps)),
                                    "jumps_planned": n_jumps,
                                    "step_displacement": self._step_displacement(unit, xi_rows, rows, kn, z)}))
        self.unit_step_count += 1
        self.hold_traces = self.hold_traces[-96:]
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
        # A held reference is observed once per commit (HOLD_CONTRACT: `prepare_reference` runs once
        # per hold).  The per-ideal displacement statistics are therefore accumulated per *ideal*,
        # not per commit; legacy traces (no hold keys in `stats`) keep the per-commit accumulation.
        held_mode = "reference_is_new" in stats
        fresh_ideal = bool(stats.get("reference_is_new")) if held_mode else True
        disp_ref = None
        if reference is not None:
            disp_ref = reference.meta.get("step_displacement")
            if isinstance(disp_ref, dict) and fresh_ideal:
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
        ret = {"rho": rho, "z_committed_mean": z_bar.tolist(), "decoder_residual": resid,
               "tanh_saturated_fraction": float((tz > SAT_HIGH).mean()),
               "tanh_underused_fraction": float((tz < SAT_LOW).mean()),
               "tanh_saturated_per_factor": sat.tolist(), "tanh_underused_per_factor": low.tolist(),
               "mean_abs_dz_per_model_step": dz_step, "ideal_step_displacement_dist2": disp_ref}
        # --- control latent: encode the COMMITTED control state (levels + played fragments) and let
        # mu_C / Sigma_C learn from it at the history rate; this is the z the latent path is compared
        # with in `latent_tracking`.
        z_ctrl = None
        if self.control_ready and stats.get("levels") is not None and stats.get("positions") is not None:
            u_c = self._controls(np.asarray(stats["levels"], dtype=np.float64),
                                 np.asarray(stats["positions"], dtype=np.int64))
            z_ctrl = self._encode_controls(u_c)
            dz = z_ctrl - self.mu_C
            self.mu_C = (1.0 - rho) * self.mu_C + rho * z_ctrl
            self.Sigma_C = (1.0 - rho) * self.Sigma_C + rho * (dz[:, None] * dz[None, :])
            self.mean_zc = unit.o[:, None] * (self.mu_C[None, :] + self.K_F * self.chi_c)
            self._refresh_LC()
            self._u_ctrl_tanh.append(np.abs(np.tanh(z_ctrl)))
            if getattr(self, "_hold_open", None) is not None:
                self._hold_open["z_last"] = z_ctrl.copy()
        if held_mode:
            meta = reference.meta if reference is not None else {}
            hold_info = {"reference_age_steps": int(stats.get("reference_age_steps", 0)),
                         "control_z_committed": (z_ctrl.tolist() if z_ctrl is not None else None),
                         "control_realization": meta.get("control_realization"),
                         "reference_is_new": bool(stats.get("reference_is_new", False)),
                         "projection_residual_dist2": meta.get("projection_residual_dist2"),
                         "requested_dist2_decoder_path": meta.get("requested_dist2_decoder_path"),
                         "requested_dist2_projected_plan": meta.get("requested_dist2_projected_plan"),
                         "candidate_plans_evaluated": meta.get("candidate_plans_evaluated"),
                         "plan_steps": meta.get("plan_steps")}
            ret["held_reference"] = hold_info
            self.step_traces[-1]["held_reference"] = hold_info
        return ret

    def end_unit(self, unit: UnitContext, history, chosen: Realization, alternatives) -> None:
        self._close_hold_track()
        zs = np.concatenate(self.unit_fits, axis=0) if self.unit_fits else np.zeros((0, self.d_z))
        st = dict(history.mode_state.get("vae", {}))
        st["mu_H"] = self.mu_H.tolist()
        st["Sigma_H"] = self.Sigma_H.tolist()
        if self.control_ready:
            st["mu_C"] = self.mu_C.tolist()
            st["Sigma_C"] = self.Sigma_C.tolist()
            st["control_basis"] = {"P": self.P.tolist(), "s": self.s_c.tolist(), "u_mean": self.u_mean.tolist(),
                                   "u_std": self.u_std.tolist(), "loadings": self.loadings_c.tolist(),
                                   "d_u": int(self.d_u), "k": int(self.k_c), "info": self.control_basis_info}
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
                                    "reference_holds": int(self.unit_hold_count),
                                    "candidate_plans_evaluated": int(sum(self._u_plan_calls)),
                                    "candidate_plans_per_hold_mean": (float(np.mean(self._u_plan_calls))
                                                                      if self._u_plan_calls else None),
                                    "candidate_plans_per_hold_max": (int(max(self._u_plan_calls))
                                                                     if self._u_plan_calls else None),
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
        if self.control_ready and self.unit_hold_count:
            req = np.asarray([a for (a, _b) in self._u_track_pairs])
            com = np.asarray([b for (_a, b) in self._u_track_pairs])
            ct = np.asarray(self._u_ctrl_tanh) if self._u_ctrl_tanh else np.zeros((0, self.k_c))
            cosines = None
            if len(req):
                nn = np.sqrt((req ** 2).sum(axis=1) * (com ** 2).sum(axis=1)) + 1e-30
                cosines = ((req * com).sum(axis=1) / nn)
            stats["latent_tracking"] = {
                "holds_paired": int(len(req)),
                "cosine_requested_vs_committed_dz_mean": (float(cosines.mean()) if cosines is not None and len(cosines) else None),
                "correlation_requested_vs_committed_dz": (float(np.corrcoef(req.ravel(), com.ravel())[0, 1])
                                                          if len(req) > 1 else None),
                "mean_requested_dz_norm": (float(np.linalg.norm(req, axis=1).mean()) if len(req) else None),
                "mean_committed_dz_norm": (float(np.linalg.norm(com, axis=1).mean()) if len(com) else None),
                "per_axis_correlation": ([float(np.corrcoef(req[:, c], com[:, c])[0, 1]) if len(req) > 1 else None
                                          for c in range(self.k_c)]),
                "note": "Δz of the hold (z at the last knot − z0) vs Δz of the committed sound, both through "
                        "the same control encoder; internal iterations are not musical time"}
            stats["control_realization"] = {
                "mean": float(np.mean(self._u_ctrl_real)),
                "jump_legal_share_mean": float(np.mean(self._u_ctrl_jump_legal)),
                "encoder_control_share_captured_mean": float(np.mean(self._u_enc_share)),
                "jumps_planned": int(self._u_jumps_planned), "jumps_dropped_by_min_clip": int(self._u_jumps_dropped),
                "note": "share of the decoded control change (levels before the [0,1] clip + fragment moves in "
                        "PCA units) that was legal to apply in the plan"}
            stats["control_latent"] = {
                "k": int(self.k_c), "control_dimension": int(self.d_u),
                "state_origin": self.control_state_origin,
                "tanh_saturated_fraction": (float((ct > SAT_HIGH).mean()) if len(ct) else None),
                "tanh_underused_fraction": (float((ct < SAT_LOW).mean()) if len(ct) else None),
                "mean_abs_tanh_per_axis": (ct.mean(axis=0).tolist() if len(ct) else None),
                "mu_C_after": self.mu_C.tolist(), "Sigma_C_after": self.Sigma_C.tolist()}
        if self.unit_hold_count:
            stats["projection_onto_realizable_set"] = {
                "holds": int(self.unit_hold_count), "hold_move_scale": self.move_scale,
                "plan_steps_per_hold_mean": float(np.mean(self._u_plan_steps)) if self._u_plan_steps else None,
                "projection_residual_mean_dist2": float(np.mean(self._u_proj_resid)),
                "projection_residual_share_of_decoder_move_mean": float(np.mean(self._u_proj_share)),
                "requested_dist2_decoder_path_mean": float(np.mean(self._u_req_dec)),
                "requested_dist2_projected_plan_mean": float(np.mean(self._u_req_plan)),
                "requested_plan_over_decoder": (float(np.mean(self._u_req_plan) / np.mean(self._u_req_dec))
                                                if np.mean(self._u_req_dec) > 1e-12 else None),
                "jumps_planned": int(self._u_jumps_planned),
                "jumps_dropped_by_min_clip": int(self._u_jumps_dropped),
                "mean_max_level_change_per_step": (float(np.mean(self._u_level_moves))
                                                   if self._u_level_moves else None),
                "note": "projection residual = mean d_xi^2 between the published EXACT plan rows and the free "
                        "decoder path D(z) on the same rows: the part of the latent manifold move that no gain "
                        "or playback-position choice can produce",
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
                "control_latent_space": {
                    "active": bool(self.control_ready), "requested": self.latent_space,
                    "law": "u = [levels of every material track | r PCA coords of the fragment it plays]; "
                           "k axes from a PLS-SVD of standardized random realizable controls against their "
                           "exact compositions; decoder u(z) = u_mean + P (s * tanh z) -> clipped levels + "
                           "nearest fragment per track -> exact rows via plan_rows (realizable by construction)",
                    "settings": {"hold_latent_space": self.latent_space, "control_latent_dim": self.k_c,
                                 "control_fragment_pca": self.r_pca, "control_basis_states": self.n_states,
                                 "control_holdout_states": self.n_holdout, "control_basis_offsets": self.control_offsets,
                                 "control_jump_min_pca_dist": self.jump_min_dist,
                                 "control_max_jumps_per_step": self.max_jumps,
                                 "control_sigma_floor": self.sigma_C_floor, "hold_move_scale": self.move_scale,
                                 "control_seed_offset": self.control_seed_offset},
                    "basis": self.control_basis_info,
                    "state_origin": getattr(self, "control_state_origin", None)},
                "hold_projection": {
                    "active": bool(self.hold_count),
                    "law": "latent OU path (anchored at the encoding of the realized anchor) decoded by "
                           "D(z) = xi_G + sum_k s_k tanh(z_k) v_k, then PROJECTED onto the realizable set: "
                           "per model step of the hold the plan step (material end levels + jumps of the "
                           "tracks that may jump, to fragment candidates for D(z)'s band profile) whose exact "
                           "rows minimise d_xi^2 to D(z); the exact rows of that plan are published",
                    "holds": int(self.hold_count), "candidate_plans_evaluated": int(self.plan_calls_total),
                    "settings": {"hold_move_scale": self.move_scale, "hold_fit_rows": self.fit_rows,
                                 "hold_level_step": self.level_step0, "hold_level_min_step": self.level_min_step,
                                 "hold_level_evals": self.level_evals, "hold_jump_candidates": self.n_frag_cand,
                                 "hold_jump_margin": self.jump_margin, "hold_step_margin": self.step_margin,
                                 "hold_target_lead_rows": self.target_lead,
                                 "hold_plan_calls_max": self.plan_calls_max,
                                 "innovation_gain_constant": HOLD_INNOVATION_GAIN,
                                 "effective_innovation_gain": HOLD_INNOVATION_GAIN * self.move_scale},
                    "holds_recorded": self.hold_traces},
                "unit_statistics": self.unit_stats,
                "latent_means_covariances": self.latent_traces, "basis_definition": self.basis_traces,
                "fitted_actual_latent_summary": self.fit_traces, "decoder": "xi_G + sum_k s_k tanh(z_k) v_k",
                "warnings": list(self.warnings), "units": self.unit_traces, "steps": self.step_traces[-64:]}

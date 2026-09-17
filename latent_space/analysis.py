"""Acoustic-composition analysis (spec §4).

For every analysis window (fixed grid over the whole piece) we precompute, for the actual
looped PCM of every track (scaled by its base gain b_i):
  G0[j]    : (M,M) time-domain Gram matrix   mean_{n,c} x_i x_k          (unwindowed)
  Gb[j,b]  : (M,M) band Gram matrices        Re sum_{f in band b, c} X_i conj(X_k)  (Hann FFT)
Because eq. (1) is linear and the gain of a track is held at its window-centre value inside an
analysis window (window_frames / fs seconds, e.g. 46 ms at 44.1 kHz, 93 ms at 22.05 kHz), the
energy / band powers of the *actual summed PCM* under a gain vector a are the exact quadratic
forms a^T G a — no per-candidate FFT is needed and cross terms are included.  The window-centre
gain hold is the approximation of the proxy; float32 working arrays add ordinary rounding; the
feature reduction (energy, 8 bands, contribution proxies, pair relations) is a modelling choice,
not a Gram-form error.  The engine re-analyses the rendered output of the chosen candidate on the
same windows (composition_from_render) and records proxy / target / rendered distances separately.
Analysis windows never extend past the end of the piece (no boundary-condition mismatch between
the looped proxy and the finite rendered file).

Composition state (eq. 17):  xi = [phi_norm (2+bands) | c (M) | upper(R) (M(M-1)/2)]
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Analyzer:
    def __init__(self, sources: Sequence[np.ndarray], base_gains: Sequence[float], fs: int,
                 total_frames: int, cfg: dict):
        a = cfg["analysis"]
        obj = cfg["objective"]
        self.fs = int(fs)
        self.M = len(sources)
        self.C = int(sources[0].shape[1])
        self.W = int(a["window_frames"])
        self.hop = max(1, int(round(float(a["hop_seconds"]) * fs)))
        self.nb = int(a["spectral_bands"])
        self.silence_energy = float(a["silence_energy"])
        self.eps = float(a["feature_epsilon"])
        self.std_floor = float(a["feature_std_floor"])
        self.sigma_F = float(a.get("sigma_F", cfg["mode_defaults"]["transformer"].get("sigma_F", 1.0)))
        self.w_phi = float(obj["w_phi"])
        self.w_c = float(obj["w_c"])
        self.w_R = float(obj["w_R"])
        self.b = np.asarray(base_gains, dtype=np.float64)
        self.d_phi = 2 + self.nb
        self.iu = np.triu_indices(self.M, 1)
        self.n_pairs = len(self.iu[0])
        self.d_xi = self.d_phi + self.M + self.n_pairs
        self.total_frames = int(total_frames)
        self.source_lengths = [int(x.shape[0]) for x in sources]
        # bands: fixed intervals over [0, Nyquist]
        F = self.W // 2 + 1
        bin_hz = fs / float(self.W)
        edges = list(a["band_edges_hz"])[: self.nb]
        while len(edges) < self.nb:
            edges.append(edges[-1] * 2.0)
        bins = [min(F - 1, int(round(e / bin_hz))) for e in edges] + [F]
        for k in range(1, len(bins)):
            if bins[k] <= bins[k - 1]:
                bins[k] = bins[k - 1] + 1
        bins = [min(b_, F) for b_ in bins]
        self.band_bins: List[Tuple[int, int]] = [(bins[k], bins[k + 1]) for k in range(self.nb)]
        self.band_edges_hz = [bb[0] * bin_hz for bb in self.band_bins] + [fs / 2.0]
        # analysis grid: window [centre - W/2, centre + W/2)
        half = self.W // 2
        # windows [c - W/2, c - W/2 + W) must lie entirely inside the rendered piece
        last_center = self.total_frames - (self.W - half)
        self.centers = np.arange(half, max(half + 1, last_center + 1), self.hop, dtype=np.int64)
        self.starts = self.centers - half
        self.J = int(len(self.centers))
        self.G0 = np.zeros((self.J, self.M, self.M), dtype=np.float64)
        self.Gb = np.zeros((self.J, self.nb, self.M, self.M), dtype=np.float64)
        self._compute_grams(sources)
        self._calibrate()
        self._material_features()
        e0 = np.zeros(self.M)
        e0[0] = 1.0
        self.xi_goal_all, self.goal_parts_all = self.composition(
            np.broadcast_to(e0, (self.J, self.M)).copy(), np.arange(self.J))

    # ------------------------------------------------------------------ precomputation
    def _compute_grams(self, sources: Sequence[np.ndarray]) -> None:
        M, W, C = self.M, self.W, self.C
        hann = np.hanning(W).astype(np.float32)
        chunk = 96
        ar = np.arange(W, dtype=np.int64)
        for j0 in range(0, self.J, chunk):
            st = self.starts[j0:j0 + chunk]
            n = len(st)
            Xt = np.empty((n, M, W, C), dtype=np.float32)
            for i, x in enumerate(sources):
                idx = (st[:, None] + ar[None, :]) % x.shape[0]
                Xt[:, i] = x[idx] * np.float32(self.b[i])
            flat = Xt.reshape(n, M, W * C)
            self.G0[j0:j0 + n] = np.matmul(flat, flat.transpose(0, 2, 1)) / float(W * C)
            Xf = np.fft.rfft(Xt * hann[None, None, :, None], axis=2)  # (n, M, F, C) complex
            for bi, (f0, f1) in enumerate(self.band_bins):
                Xb = Xf[:, :, f0:f1, :].reshape(n, M, -1)
                G = np.matmul(Xb, np.conj(Xb).transpose(0, 2, 1)).real
                self.Gb[j0:j0 + n, bi] = G
        self.Gb = np.maximum(self.Gb, 0.0) if False else self.Gb  # keep raw (symmetric PSD)

    def _phi_raw(self, gains: np.ndarray, idx: np.ndarray):
        a = np.asarray(gains, dtype=np.float64)
        G0 = self.G0[idx]
        Gb = self.Gb[idx]
        E_y = np.einsum("jm,jmk,jk->j", a, G0, a)
        E_y = np.maximum(E_y, 0.0)
        diag = np.einsum("jmm->jm", G0)
        e = a * a * diag
        Pb = np.einsum("jm,jbmk,jk->jb", a, Gb, a)
        Pb = np.maximum(Pb, 0.0)
        Ptot = Pb.sum(axis=1, keepdims=True)
        ratios = Pb / (Ptot + self.eps)
        silent = E_y < self.silence_energy
        ratios[silent] = 0.0
        flux = np.zeros(len(E_y))
        if len(E_y) > 1:
            d = ratios[1:] - ratios[:-1]
            flux[1:] = (np.maximum(d, 0.0) ** 2).sum(axis=1)
        logE = np.log1p(E_y / self.E_ref)
        phi_raw = np.concatenate([logE[:, None], ratios, flux[:, None]], axis=1)
        return phi_raw, e, E_y, silent

    def _calibrate(self) -> None:
        diag = np.einsum("jmm->jm", self.G0)  # (J, M) energies of each track alone
        meds = []
        for i in range(self.M):
            v = diag[:, i]
            v = v[v >= self.silence_energy]
            meds.append(float(np.median(v)) if len(v) else self.silence_energy)
        self.E_ref = max(float(np.mean(meds)), self.silence_energy)
        idx = np.arange(self.J)
        samples = []
        for i in range(self.M):
            g = np.zeros((self.J, self.M))
            g[:, i] = 1.0
            samples.append(self._phi_raw(g, idx)[0])
        anchor = np.full((self.J, self.M), 0.4)
        anchor[:, 0] = 0.1
        samples.append(self._phi_raw(anchor, idx)[0])
        samples.append(self._phi_raw(np.ones((self.J, self.M)), idx)[0])
        allp = np.concatenate(samples, axis=0)
        self.norm_mean = allp.mean(axis=0)
        self.norm_std = np.maximum(allp.std(axis=0), self.std_floor)

    def _material_features(self) -> None:
        idx = np.arange(self.J)
        diag = np.einsum("jmm->jm", self.G0)
        f_raw = np.zeros((self.J, self.M, self.d_phi))
        for i in range(self.M):
            g = np.zeros((self.J, self.M))
            g[:, i] = 1.0
            f_raw[:, i, :] = self._phi_raw(g, idx)[0]
        self.f_raw = f_raw
        self.f_mat = (f_raw - self.norm_mean) / self.norm_std
        self.silent_mat = diag < self.silence_energy
        logE = f_raw[:, :, 0]
        chi = np.zeros((self.J, self.M))
        chi[1:] = np.tanh(logE[1:] - logE[:-1])
        self.chi = chi
        diff = self.f_mat[:, :, None, :] - self.f_mat[:, None, :, :]
        self.S = np.exp(-(diff ** 2).sum(axis=-1) / (2.0 * self.sigma_F ** 2))

    # ------------------------------------------------------------------ hires extension
    def build_solo_bank(self, sources: Sequence[np.ndarray], hop_seconds: float = 0.1) -> None:
        """Per-source solo features on a position grid over the whole source (for jump candidates)."""
        self.solo_hop = max(1, int(round(hop_seconds * self.fs)))
        self.solo_pos: List[np.ndarray] = []
        self.solo_f: List[np.ndarray] = []      # normalized features (P_i, d_phi)
        self.solo_E: List[np.ndarray] = []
        hann = np.hanning(self.W).astype(np.float32)
        ar = np.arange(self.W, dtype=np.int64)
        for i, x in enumerate(sources):
            L = x.shape[0]
            pos = np.arange(0, L, self.solo_hop, dtype=np.int64)
            rows = []
            Es = []
            chunk = 512
            for p0 in range(0, len(pos), chunk):
                st = pos[p0:p0 + chunk]
                idx = (st[:, None] + ar[None, :]) % L
                Xt = x[idx] * np.float32(self.b[i])                      # (n, W, C)
                E = (Xt.astype(np.float64) ** 2).mean(axis=(1, 2))
                Xf = np.fft.rfft(Xt * hann[None, :, None], axis=1)       # (n, F, C)
                P = (np.abs(Xf) ** 2).sum(axis=2)
                Pb = np.stack([P[:, f0:f1].sum(axis=1) for (f0, f1) in self.band_bins], axis=1)
                ratios = Pb / (Pb.sum(axis=1, keepdims=True) + self.eps)
                ratios[E < self.silence_energy] = 0.0
                rows.append(np.concatenate([np.log1p(E / self.E_ref)[:, None], ratios], axis=1))
                Es.append(E)
            raw = np.concatenate(rows, axis=0)
            flux = np.zeros(len(raw))
            if len(raw) > 1:
                d = raw[1:, 1:] - raw[:-1, 1:]
                flux[1:] = (np.maximum(d, 0.0) ** 2).sum(axis=1)
            raw = np.concatenate([raw, flux[:, None]], axis=1)
            self.solo_pos.append(pos)
            self.solo_f.append((raw - self.norm_mean) / self.norm_std)
            self.solo_E.append(np.concatenate(Es))

    def build_fragment_bank(self, clip_seconds: float) -> None:
        """Fragment vocabulary: for every solo-bank position, the features averaged over the
        clip_seconds that follow it (what actually sounds after a jump), circular over the source."""
        n = max(1, int(round(clip_seconds * self.fs / self.solo_hop)))
        self.frag_rows = n
        self.frag_f: List[np.ndarray] = []
        self.frag_E: List[np.ndarray] = []
        for i in range(self.M):
            f = self.solo_f[i]
            E = self.solo_E[i]
            P = len(f)
            idx = (np.arange(P)[:, None] + np.arange(n)[None, :]) % P
            self.frag_f.append(f[idx].mean(axis=1))
            self.frag_E.append(E[idx].mean(axis=1))

    def fragment_composition(self, sources: Sequence[np.ndarray], positions: np.ndarray, levels: np.ndarray,
                             offsets: np.ndarray):
        """Exact composition rows of the mixture in which track i plays from source position
        positions[i] at level levels[i]; one row per offset (frames after the positions)."""
        offsets = np.asarray(offsets, dtype=np.int64)
        pos = np.stack([(int(positions[i]) + offsets) % sources[i].shape[0] for i in range(self.M)], axis=1)
        G0, Gb = self.grams_at_positions(sources, offsets, pos)
        _f, S, _chi = self.material_features_at(pos)
        gains = np.broadcast_to(np.asarray(levels, dtype=np.float64), (len(offsets), self.M)).copy()
        return self.composition_from_grams(gains, G0, Gb, S)

    def random_fragment_composition(self, sources: Sequence[np.ndarray], rng: np.random.Generator, offsets: np.ndarray,
                                    goal_level: float = 0.0, levels: Optional[np.ndarray] = None):
        """A random legal fragment composition (random positions, random material levels)."""
        positions = np.array([int(rng.integers(0, sources[i].shape[0])) for i in range(self.M)])
        if levels is None:
            levels = rng.uniform(0.0, 1.0, self.M)
            levels[0] = goal_level
        xi, parts = self.fragment_composition(sources, positions, levels, offsets)
        return xi, parts, {"positions": positions.tolist(), "levels": np.asarray(levels).tolist()}

    def fragment_candidates(self, track: int, target_ratios: np.ndarray, n: int, exclude_near: Optional[int] = None,
                            exclude_frames: int = 0) -> List[int]:
        """Top-n fragment positions of `track` whose clip-averaged band profile is closest to the
        target (normalized band block), silent fragments penalised."""
        F = self.frag_f[track]
        d = ((F[:, 1:1 + self.nb] - np.asarray(target_ratios)[None, :]) ** 2).sum(axis=1)
        d = d + 4.0 * (self.frag_E[track] < self.silence_energy)
        if exclude_near is not None and exclude_frames > 0:
            near = np.abs(self.solo_pos[track] - exclude_near) < exclude_frames
            d = d + 1e6 * near
        order = np.argsort(d)[: max(1, n)]
        return [int(self.solo_pos[track][k]) for k in order]

    def grams_at_positions(self, sources: Sequence[np.ndarray], starts: np.ndarray, positions: np.ndarray):
        """Window Gram matrices for explicit per-track source positions (positions: (n, M) source
        frames of the window start for each track).  Exact for the summed PCM of those windows."""
        n = len(starts)
        M, W, C = self.M, self.W, self.C
        hann = np.hanning(W).astype(np.float32)
        ar = np.arange(W, dtype=np.int64)
        Xt = np.empty((n, M, W, C), dtype=np.float32)
        for i, x in enumerate(sources):
            idx = (positions[:, i][:, None] + ar[None, :]) % x.shape[0]
            Xt[:, i] = x[idx] * np.float32(self.b[i])
        flat = Xt.reshape(n, M, W * C)
        G0 = np.matmul(flat, flat.transpose(0, 2, 1)) / float(W * C)
        Xf = np.fft.rfft(Xt * hann[None, None, :, None], axis=2)
        Gb = np.zeros((n, self.nb, M, M), dtype=np.float64)
        for bi, (f0, f1) in enumerate(self.band_bins):
            Xb = Xf[:, :, f0:f1, :].reshape(n, M, -1)
            Gb[:, bi] = np.matmul(Xb, np.conj(Xb).transpose(0, 2, 1)).real
        return G0, Gb

    def composition_from_grams(self, gains: np.ndarray, G0: np.ndarray, Gb: np.ndarray, S_rows: np.ndarray):
        """Composition state from explicit Grams (rows consecutive) and material similarity rows."""
        a = np.asarray(gains, dtype=np.float64)
        E_y = np.maximum(np.einsum("jm,jmk,jk->j", a, G0, a), 0.0)
        diag = np.einsum("jmm->jm", G0)
        e = a * a * diag
        Pb = np.maximum(np.einsum("jm,jbmk,jk->jb", a, Gb, a), 0.0)
        ratios = Pb / (Pb.sum(axis=1, keepdims=True) + self.eps)
        silent = E_y < self.silence_energy
        ratios[silent] = 0.0
        flux = np.zeros(len(E_y))
        if len(E_y) > 1:
            d = ratios[1:] - ratios[:-1]
            flux[1:] = (np.maximum(d, 0.0) ** 2).sum(axis=1)
        phi_raw = np.concatenate([np.log1p(E_y / self.E_ref)[:, None], ratios, flux[:, None]], axis=1)
        phi = (phi_raw - self.norm_mean) / self.norm_std
        c = e / (e.sum(axis=1, keepdims=True) + self.eps)
        R = S_rows * c[:, :, None] * c[:, None, :]
        Rup = R[:, self.iu[0], self.iu[1]]
        xi = np.concatenate([phi, c, Rup], axis=1)
        return xi, {"phi": phi, "phi_raw": phi_raw, "c": c, "R": R, "silent": silent, "E": E_y, "e": e}

    def material_features_at(self, positions: np.ndarray):
        """Normalized solo features f (n, M, d_phi) at explicit positions (nearest solo-bank row)
        and the similarity S (n, M, M) and log-energy change chi (n, M)."""
        n = positions.shape[0]
        f = np.zeros((n, self.M, self.d_phi))
        for i in range(self.M):
            k = np.clip(np.round(positions[:, i] / float(self.solo_hop)).astype(np.int64), 0, len(self.solo_pos[i]) - 1)
            f[:, i, :] = self.solo_f[i][k]
        diff = f[:, :, None, :] - f[:, None, :, :]
        S = np.exp(-(diff ** 2).sum(axis=-1) / (2.0 * self.sigma_F ** 2))
        raw_logE = f[:, :, 0] * self.norm_std[0] + self.norm_mean[0]
        chi = np.zeros((n, self.M))
        if n > 1:
            chi[1:] = np.tanh(raw_logE[1:] - raw_logE[:-1])
        return f, S, chi

    # ------------------------------------------------------------------ public API
    def composition(self, gains: np.ndarray, idx: np.ndarray):
        """gains (J', M) at consecutive grid indices idx -> (xi (J', d_xi), parts)."""
        idx = np.asarray(idx)
        phi_raw, e, E_y, silent = self._phi_raw(gains, idx)
        phi = (phi_raw - self.norm_mean) / self.norm_std
        c = e / (e.sum(axis=1, keepdims=True) + self.eps)
        S = self.S[idx]
        R = S * c[:, :, None] * c[:, None, :]
        Rup = R[:, self.iu[0], self.iu[1]]
        xi = np.concatenate([phi, c, Rup], axis=1)
        parts = {"phi": phi, "phi_raw": phi_raw, "c": c, "R": R, "silent": silent, "E": E_y, "e": e}
        return xi, parts

    def split(self, xi: np.ndarray):
        return (xi[..., : self.d_phi], xi[..., self.d_phi: self.d_phi + self.M], xi[..., self.d_phi + self.M:])

    def dist2(self, xi_a: np.ndarray, xi_b: np.ndarray) -> np.ndarray:
        """Eq. (22) per row."""
        d = xi_a - xi_b
        dp, dc, dr = self.split(d)
        out = self.w_phi * (dp ** 2).mean(axis=-1) + self.w_c * (dc ** 2).mean(axis=-1)
        if self.n_pairs > 0:
            out = out + self.w_R * (dr ** 2).mean(axis=-1)
        return out

    def weight_vector(self) -> np.ndarray:
        """Diagonal W such that dist2 = d^T W d."""
        w = np.concatenate([np.full(self.d_phi, self.w_phi / self.d_phi),
                            np.full(self.M, self.w_c / self.M),
                            np.full(self.n_pairs, self.w_R / max(1, self.n_pairs))])
        return w

    def grid_indices(self, start: int, end: int) -> np.ndarray:
        return np.where((self.centers >= start) & (self.centers < end))[0]

    def phi_from_pcm(self, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """True mixture features of rendered PCM y (frames, C) at grid indices idx (for verification)."""
        hann = np.hanning(self.W)
        rows = []
        prev = None
        for j in idx:
            s = int(self.starts[j])
            seg = y[s: s + self.W].astype(np.float64)
            if seg.shape[0] < self.W:
                seg = np.pad(seg, ((0, self.W - seg.shape[0]), (0, 0)))
            E = float((seg ** 2).mean())
            X = np.fft.rfft(seg * hann[:, None], axis=0)
            P = (np.abs(X) ** 2).sum(axis=1)
            Pb = np.array([P[f0:f1].sum() for (f0, f1) in self.band_bins])
            ratios = Pb / (Pb.sum() + self.eps)
            if E < self.silence_energy:
                ratios = np.zeros(self.nb)
            flux = 0.0 if prev is None else float((np.maximum(ratios - prev, 0.0) ** 2).sum())
            prev = ratios
            rows.append(np.concatenate([[np.log1p(E / self.E_ref)], ratios, [flux]]))
        raw = np.array(rows)
        return (raw - self.norm_mean) / self.norm_std

    def composition_from_render(self, y: np.ndarray, sources: Sequence[np.ndarray], base_gains: Sequence[float],
                                curves, idx: np.ndarray):
        """True composition state of the rendered PCM y on *consecutive* grid rows idx: mixture
        features from y itself (time-varying gains), per-track contributions from the exact
        component signals b_i g_i(n) x_i[n mod L_i] (the sources and the official curves are known,
        no source separation), relations from the same S.  Flux is computed along the given
        consecutive sequence (its first row has flux 0, like any sequence start)."""
        idx = np.asarray(idx)
        hann = np.hanning(self.W)
        rows_phi = []
        e_rows = []
        prev = None
        for j in idx:
            s0 = int(self.starts[j])
            frames = np.arange(s0, s0 + self.W, dtype=np.int64)
            seg = y[s0: s0 + self.W].astype(np.float64)
            if seg.shape[0] < self.W:
                seg = np.pad(seg, ((0, self.W - seg.shape[0]), (0, 0)))
            E = float((seg ** 2).mean())
            X = np.fft.rfft(seg * hann[:, None], axis=0)
            P = (np.abs(X) ** 2).sum(axis=1)
            Pb = np.array([P[f0:f1].sum() for (f0, f1) in self.band_bins])
            ratios = Pb / (Pb.sum() + self.eps)
            if E < self.silence_energy:
                ratios = np.zeros(self.nb)
            flux = 0.0 if prev is None else float((np.maximum(ratios - prev, 0.0) ** 2).sum())
            prev = ratios
            rows_phi.append(np.concatenate([[np.log1p(E / self.E_ref)], ratios, [flux]]))
            e = []
            for i, (x, b, cv) in enumerate(zip(sources, base_gains, curves)):
                g = cv.values(frames)
                v = float(b) * g[:, None] * x[cv.positions(frames, x.shape[0])].astype(np.float64)
                e.append(float((v ** 2).mean()))
            e_rows.append(e)
        phi_raw = np.array(rows_phi)
        phi = (phi_raw - self.norm_mean) / self.norm_std
        e = np.array(e_rows)
        c = e / (e.sum(axis=1, keepdims=True) + self.eps)
        S = self.S[idx]
        R = S * c[:, :, None] * c[:, None, :]
        Rup = R[:, self.iu[0], self.iu[1]]
        xi = np.concatenate([phi, c, Rup], axis=1)
        return xi, {"phi": phi, "phi_raw": phi_raw, "c": c, "R": R, "e": e}

    def to_trace(self) -> dict:
        return {
            "window_frames": self.W, "window_seconds": self.W / self.fs,
            "hop_frames": self.hop, "hop_seconds": self.hop / self.fs,
            "grid_points": self.J, "band_edges_hz": [float(x) for x in self.band_edges_hz],
            "band_bins": [list(map(int, bb)) for bb in self.band_bins],
            "E_ref": self.E_ref, "feature_norm_mean": self.norm_mean.tolist(),
            "feature_norm_std": self.norm_std.tolist(), "sigma_F": self.sigma_F,
            "d_phi": self.d_phi, "d_xi": self.d_xi, "n_pairs": self.n_pairs,
            "distance_weights": {"w_phi": self.w_phi, "w_c": self.w_c, "w_R": self.w_R},
            "window_gain_hold_note": f"gains held at window-centre value inside each {self.W / self.fs * 1000:.1f} ms "
                                     "analysis window; mixture quadratic forms are exact for the summed PCM under "
                                     "that hold; windows never cross the end of the rendered piece",
        }

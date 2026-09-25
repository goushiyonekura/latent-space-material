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
        self.occ_threshold = float(a.get("occupancy_threshold", 0.25))
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
        seen: Dict[int, int] = {}                 # tracks that share one source array (second voices) share its bank
        for i, x in enumerate(sources):
            if id(x) in seen and float(self.b[i]) == float(self.b[seen[id(x)]]):
                j = seen[id(x)]
                self.solo_pos.append(self.solo_pos[j])
                self.solo_f.append(self.solo_f[j])
                self.solo_E.append(self.solo_E[j])
                continue
            seen[id(x)] = i
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
        # sparsity convergence: occupancy of the clip that follows each position = the share of its
        # hops whose solo energy exceeds occupancy_threshold x the source's median (non-silent) energy
        self.frag_occ: List[np.ndarray] = []
        for i in range(self.M):
            j = next((k for k in range(i) if self.solo_f[k] is self.solo_f[i]), None)
            if j is not None:                      # second voice of the same source
                self.frag_f.append(self.frag_f[j])
                self.frag_E.append(self.frag_E[j])
                self.frag_occ.append(self.frag_occ[j])
                continue
            f = self.solo_f[i]
            E = self.solo_E[i]
            P = len(f)
            idx = (np.arange(P)[:, None] + np.arange(n)[None, :]) % P
            self.frag_f.append(f[idx].mean(axis=1))
            self.frag_E.append(E[idx].mean(axis=1))
            nz = E[E >= self.silence_energy]
            med = float(np.median(nz)) if len(nz) else float(self.silence_energy)
            self.frag_occ.append((E[idx] > self.occ_threshold * med).mean(axis=1))
        # the goal's own occupancy on the continuous clock, per analysis row (the target of the pull)
        L0 = int(self.source_lengths[0])
        k0 = np.clip(np.round((self.starts % L0) / float(self.solo_hop)).astype(np.int64), 0, len(self.frag_occ[0]) - 1)
        self.goal_occ_all = self.frag_occ[0][k0]

    def occupancy_at(self, positions: np.ndarray) -> np.ndarray:
        """Clip occupancy (n, M) of every track at explicit source positions (nearest fragment)."""
        positions = np.asarray(positions, dtype=np.int64)
        out = np.zeros(positions.shape, dtype=np.float64)
        for i in range(self.M):
            k = np.clip(np.round(positions[:, i] / float(self.solo_hop)).astype(np.int64), 0, len(self.frag_occ[i]) - 1)
            out[:, i] = self.frag_occ[i][k]
        return out

    @staticmethod
    def presence_terms(gains: np.ndarray, solo_f: np.ndarray, occ: np.ndarray, occ_goal: np.ndarray,
                       w_phi: float, w_occ: float, eps: float = 1e-6, d_empty: float = 1.7):
        """Per row, over the SOUNDING materials (gain > eps, the goal excluded): their number, the mean
        spectral distance of their solo fragment features to the goal's own solo features at the same
        row, and the mean squared occupancy difference to the goal's.  Levels do not enter beyond
        presence, so lowering a material's level changes nothing until it is silent."""
        g = np.asarray(gains, dtype=np.float64)
        on = (g[:, 1:] > eps).astype(np.float64)                      # (n, M-1)
        n_s = on.sum(axis=1)
        f = np.asarray(solo_f, dtype=np.float64)                       # (n, M, d_phi)
        d_spec_i = w_phi * ((f[:, 1:, :] - f[:, :1, :]) ** 2).mean(axis=2)          # (n, M-1)
        d_occ_i = w_occ * (np.asarray(occ)[:, 1:] - np.asarray(occ_goal)[:, None]) ** 2
        denom = np.maximum(n_s, 1.0)
        spec = (d_spec_i * on).sum(axis=1) / denom
        occ_t = (d_occ_i * on).sum(axis=1) / denom
        # an empty set is not "distance 0": silence is as far from the goal as a poor material, scaled by
        # the goal's absence (with the goal fully in, no material is needed)
        empty = n_s <= 0
        spec = np.where(empty, d_empty * (1.0 - np.clip(g[:, 0], 0.0, 1.0)), spec)
        occ_t = np.where(empty, 0.0, occ_t)
        return n_s, spec, occ_t

    @staticmethod
    def presence_level_term(gains: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Per row (goal_exposure.presence_w_level, 'B'): (1 - level of the loudest sounding material)^2,
        0 when no material sounds.  The presence measure is flat in the levels, so without this nothing
        keeps a sounding material audible; with it the loudest one is held near full level and a source
        leaves by being dropped, not by fading."""
        m = np.asarray(gains, dtype=np.float64)[:, 1:]
        top = m.max(axis=1) if m.shape[1] else np.zeros(m.shape[0])
        return np.where(top > eps, (1.0 - np.clip(top, 0.0, 1.0)) ** 2, 0.0)

    @staticmethod
    def sparsity_terms(c: np.ndarray, occ: np.ndarray, occ_goal: np.ndarray, eps: float = 1e-12):
        """Per row: effective number of sources N_eff (all tracks, goal included) and the
        contribution-weighted mixture occupancy; used by the 'sparsity' convergence measure."""
        c = np.asarray(c, dtype=np.float64)
        tot = c.sum(axis=1)
        p = c / (tot[:, None] + eps)
        neff = np.where(tot > 1e-9, 1.0 / (np.sum(p * p, axis=1) + eps), 1.0)
        occ_mix = np.where(tot > 1e-9, (c * occ).sum(axis=1) / (tot + eps), occ_goal)
        return neff, occ_mix

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
            levels = self.cap_random_levels(levels, rng)
        xi, parts = self.fragment_composition(sources, positions, levels, offsets)
        return xi, parts, {"positions": positions.tolist(), "levels": np.asarray(levels).tolist()}

    def cap_random_levels(self, levels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Polyphony cap (engine: hires.max_active_materials): a RANDOM legal state has 1..cap sounding
        materials (the voices of one material count once; a sounding material keeps at least one
        voice).  No cap configured: the levels are returned unchanged and no random number is drawn."""
        cap = int(getattr(self, "max_active_materials", 0))
        if cap <= 0:
            return levels
        groups = np.asarray(getattr(self, "material_of_track", np.arange(self.M)))
        mats = sorted({int(g) for g in groups[1:]})
        k = int(rng.integers(1, min(cap, len(mats)) + 1))
        keep = {int(x) for x in rng.choice(mats, size=k, replace=False)}
        out = np.asarray(levels, dtype=np.float64).copy()
        for m_ in mats:
            idx = np.where(groups == m_)[0]
            if m_ not in keep:
                out[idx] = 0.0
            elif len(idx) > 1:                    # each extra voice sounds with probability 1/2
                drop = [int(i) for i in idx if rng.random() < 0.5]
                if len(drop) == len(idx):
                    drop = drop[1:]
                out[drop] = 0.0
        if float(out[1:].max()) < 0.3:            # a random legal state is not near-silence
            out[1 + int(np.argmax(out[1:]))] = float(rng.uniform(0.3, 1.0))
        return out

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

    # ---- mixture-aware fragment search (2026-09-17): which fragment of ONE track brings the whole
    # MIXTURE closest to a target, given what the other tracks play.  Cross terms are ignored (they
    # average out over a clip), so this is a fast pre-ranking over every fragment of the source; the
    # caller evaluates the short list exactly (plan_rows / fragment_composition).
    def fragment_index_at(self, track: int, position: int) -> int:
        return int(np.clip(int(round(int(position) / float(self.solo_hop))), 0, len(self.solo_pos[track]) - 1))

    def fragment_band_energy(self, track: int, positions=None) -> np.ndarray:
        """Raw clip-averaged band energies (energy x band ratios) of `track` at the fragment(s)
        nearest to `positions` (int or array); all fragments (P_i, nb) when positions is None."""
        if not hasattr(self, "_frag_band"):
            self._frag_band = {}
        if track not in self._frag_band:
            ratios = self.frag_f[track][:, 1:1 + self.nb] * self.norm_std[1:1 + self.nb] + self.norm_mean[1:1 + self.nb]
            self._frag_band[track] = self.frag_E[track][:, None] * np.clip(ratios, 0.0, None)
        B = self._frag_band[track]
        if positions is None:
            return B
        if np.ndim(positions) == 0:
            return B[self.fragment_index_at(track, int(positions))]
        k = np.clip(np.round(np.asarray(positions) / float(self.solo_hop)).astype(np.int64), 0, len(B) - 1)
        return B[k]

    def mixture_band_energy(self, positions: Sequence[int], levels: Sequence[float], skip: Optional[int] = None) -> np.ndarray:
        """Predicted raw band energies (nb,) of the mixture 'track i plays its fragment at
        positions[i] with gain levels[i]' (cross terms ignored), optionally without track `skip`."""
        out = np.zeros(self.nb)
        for i in range(self.M):
            if skip is not None and i == skip:
                continue
            a = float(levels[i])
            if a > 0.0:
                out += a * a * self.fragment_band_energy(i, int(positions[i]))
        return out

    def fragment_candidates_mix(self, track: int, target_phi: np.ndarray, others_band: np.ndarray, level: float, n: int,
                                exclude_near: Optional[int] = None, exclude_frames: int = 0,
                                energy_weight: float = 1.0, penalty: Optional[np.ndarray] = None) -> List[int]:
        """Top-n fragment positions of `track` for a target MIXTURE.  `target_phi` = normalized
        [logE, band ratios (nb), ...] of the wanted mixture (the first 1+nb entries of a xi row),
        `others_band` = mixture_band_energy(..., skip=track) of what the other tracks play,
        `level` = the gain this track would have.  Ranks EVERY fragment of the source."""
        B = self.fragment_band_energy(track)                                  # (P, nb)
        cand = np.asarray(others_band, dtype=np.float64)[None, :] + float(level) ** 2 * B
        E = cand.sum(axis=1)
        ratios = cand / (E[:, None] + self.eps)
        tp = np.asarray(target_phi, dtype=np.float64)
        rn = (ratios - self.norm_mean[1:1 + self.nb]) / self.norm_std[1:1 + self.nb]
        en = (np.log1p(E / self.E_ref) - self.norm_mean[0]) / self.norm_std[0]
        d = ((rn - tp[None, 1:1 + self.nb]) ** 2).sum(axis=1) + float(energy_weight) * (en - tp[0]) ** 2
        d = d + 4.0 * (self.frag_E[track] < self.silence_energy)
        if penalty is not None:                   # e.g. fragment_visit_penalty(): keeps a mode from re-choosing the same places
            d = d + np.asarray(penalty, dtype=np.float64)
        if exclude_near is not None and exclude_frames > 0:
            d = d + 1e6 * (np.abs(self.solo_pos[track] - exclude_near) < exclude_frames)
        order = np.argsort(d)[: max(1, n)]
        return [int(self.solo_pos[track][k]) for k in order]

    def fragment_visit_penalty(self, track: int, visited_positions: Sequence[int], width_seconds: float = 2.0,
                               weight: float = 1.0, ages: Optional[Sequence[float]] = None,
                               half_life: Optional[float] = None) -> np.ndarray:
        """Additive ranking penalty (P_i,) for fragments near positions that were already played:
        a triangular bump of `width_seconds` around every visited source position (circular),
        optionally faded with the age of the visit (`ages`, same unit as `half_life`)."""
        pos = self.solo_pos[track].astype(np.float64)
        L = float(self.source_lengths[track])
        w = max(1.0, float(width_seconds) * self.fs)
        out = np.zeros(len(pos))
        for k, v in enumerate(visited_positions):
            dist = np.abs(pos - float(v))
            dist = np.minimum(dist, L - dist)
            bump = np.clip(1.0 - dist / w, 0.0, None)
            if ages is not None and half_life:
                bump = bump * 0.5 ** (float(ages[k]) / float(half_life))
            out += bump
        return float(weight) * out

    def enable_window_cache(self, max_entries: int = 256) -> None:
        """Reuse per-track PCM windows / spectra between grams_at_positions calls (hold mode)."""
        self._win_cache: Dict[tuple, tuple] = {}
        self._win_cache_max = int(max_entries)

    def grams_at_positions(self, sources: Sequence[np.ndarray], starts: np.ndarray, positions: np.ndarray):
        """Window Gram matrices for explicit per-track source positions (positions: (n, M) source
        frames of the window start for each track).  Exact for the summed PCM of those windows."""
        n = len(starts)
        M, W, C = self.M, self.W, self.C
        hann = np.hanning(W).astype(np.float32)
        ar = np.arange(W, dtype=np.int64)
        cache = getattr(self, "_win_cache", None)
        if cache is None:
            Xt = np.empty((n, M, W, C), dtype=np.float32)
            for i, x in enumerate(sources):
                idx = (positions[:, i][:, None] + ar[None, :]) % x.shape[0]
                Xt[:, i] = x[idx] * np.float32(self.b[i])
            Xf = np.fft.rfft(Xt * hann[None, None, :, None], axis=2)
        else:
            # hold mode (enable_window_cache): candidates of one commit step differ in the position of
            # one or two tracks only, so the windows / spectra of the other tracks are reused
            Xt = np.empty((n, M, W, C), dtype=np.float32)
            Xf = None
            for i, x in enumerate(sources):
                pi = np.ascontiguousarray(positions[:, i], dtype=np.int64)
                key = (id(x), float(self.b[i]), pi.tobytes())
                hit = cache.get(key)
                if hit is None:
                    idx = (pi[:, None] + ar[None, :]) % x.shape[0]
                    xt = x[idx] * np.float32(self.b[i])
                    xf = np.fft.rfft(xt * hann[None, :, None], axis=1)
                    if len(cache) >= self._win_cache_max:
                        cache.clear()
                    cache[key] = hit = (xt, xf)
                Xt[:, i] = hit[0]
                if Xf is None:
                    Xf = np.empty((n, M) + hit[1].shape[1:], dtype=hit[1].dtype)
                Xf[:, i] = hit[1]
        flat = Xt.reshape(n, M, W * C)
        G0 = np.matmul(flat, flat.transpose(0, 2, 1)) / float(W * C)
        Gb = np.zeros((n, self.nb, M, M), dtype=np.float64)
        for bi, (f0, f1) in enumerate(self.band_bins):
            Xb = Xf[:, :, f0:f1, :].reshape(n, M, -1)
            Gb[:, bi] = np.matmul(Xb, np.conj(Xb).transpose(0, 2, 1)).real
        return G0, Gb

    def composition_from_grams(self, gains: np.ndarray, G0: np.ndarray, Gb: np.ndarray, S_rows: np.ndarray,
                               prev_ratios: Optional[np.ndarray] = None):
        """Composition state from explicit Grams (rows consecutive) and material similarity rows.
        `prev_ratios` = raw band ratios of the row just before the first one: its flux is then a real
        value instead of the 0 of a sequence start (hold mode: a commit window continues the committed
        sound, and a flux that drops to 0 at every commit would be an artefact a discriminator can read)."""
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
        if prev_ratios is not None and len(E_y):
            flux[0] = float((np.maximum(ratios[0] - np.asarray(prev_ratios, dtype=np.float64), 0.0) ** 2).sum())
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

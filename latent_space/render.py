"""Official renderer: equation (1) only.  y[n,c] = sum_i b_i g_i(n/fs) x_i[n mod L_i, c]."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .curves import TrackCurve


def render_equation_1(sources: Sequence[np.ndarray], base_gains: Sequence[float], curves: List[TrackCurve],
                      total_frames: int, block_frames: int = 8192, splice_crossfade_frames: int = 0) -> np.ndarray:
    """y[n,c] = sum_i b_i g_i(n) x_i[p_i(n), c].  With an empty position map p_i(n) = n mod L_i
    (the official eq. 1).  With clips (hires extension) p_i jumps at splice points; a short
    raised-cosine crossfade between the outgoing and incoming positions of the same source
    removes the splice click (authorised cut-and-splice)."""
    C = sources[0].shape[1]
    y = np.zeros((total_frames, C), dtype=np.float32)
    ar = np.arange(block_frames, dtype=np.int64)
    for n0 in range(0, total_frames, block_frames):
        n1 = min(total_frames, n0 + block_frames)
        frames = n0 + ar[: n1 - n0]
        acc = np.zeros((n1 - n0, C), dtype=np.float64)
        for i, (x, b, cv) in enumerate(zip(sources, base_gains, curves)):
            g = cv.values(frames)  # float64, evaluated at integer sample times
            L = x.shape[0]
            pos = cv.positions(frames, L)
            sig = x[pos].astype(np.float64)
            if cv.clips and splice_crossfade_frames > 0:
                for c in cv.clips:
                    s0, s1 = c.out_start, c.out_start + splice_crossfade_frames
                    if s1 <= n0 or s0 >= n1 or s0 <= 0:
                        continue
                    lo, hi = max(s0, n0), min(s1, n1)
                    fr = np.arange(lo, hi, dtype=np.int64)
                    # outgoing stream: the previous clip continued past the splice
                    prev = [q for q in cv.clips if q.out_start < c.out_start]
                    src_prev = (prev[-1].src_start + (fr - prev[-1].out_start)) % L if prev else fr % L
                    w = 0.5 - 0.5 * np.cos(np.pi * (fr - s0) / float(splice_crossfade_frames))
                    sig[lo - n0:hi - n0] = (1.0 - w)[:, None] * x[src_prev].astype(np.float64) + w[:, None] * sig[lo - n0:hi - n0]
            acc += (float(b) * g)[:, None] * sig
        y[n0:n1] = acc.astype(np.float32)
    return y


def goal_hold_reference(source0: np.ndarray, b0: float, start: int, end: int) -> np.ndarray:
    """What eq. (13) must produce on [start, end): b_0 x_0[n mod L_0] cast to float32."""
    frames = np.arange(start, end, dtype=np.int64)
    return (float(b0) * source0[frames % source0.shape[0]].astype(np.float64)).astype(np.float32)

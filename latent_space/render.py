"""Official renderer: equation (1) only.  y[n,c] = sum_i b_i g_i(n/fs) x_i[n mod L_i, c]."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .curves import TrackCurve


def render_equation_1(sources: Sequence[np.ndarray], base_gains: Sequence[float], curves: List[TrackCurve],
                      total_frames: int, block_frames: int = 8192) -> np.ndarray:
    C = sources[0].shape[1]
    y = np.zeros((total_frames, C), dtype=np.float32)
    ar = np.arange(block_frames, dtype=np.int64)
    for n0 in range(0, total_frames, block_frames):
        n1 = min(total_frames, n0 + block_frames)
        frames = n0 + ar[: n1 - n0]
        acc = np.zeros((n1 - n0, C), dtype=np.float64)
        for i, (x, b, cv) in enumerate(zip(sources, base_gains, curves)):
            g = cv.values(frames)  # float64, evaluated at integer sample times
            idx = frames % x.shape[0]
            acc += (float(b) * g)[:, None] * x[idx].astype(np.float64)
        y[n0:n1] = acc.astype(np.float32)
    return y


def goal_hold_reference(source0: np.ndarray, b0: float, start: int, end: int) -> np.ndarray:
    """What eq. (13) must produce on [start, end): b_0 x_0[n mod L_0] cast to float32."""
    frames = np.arange(start, end, dtype=np.int64)
    return (float(b0) * source0[frames % source0.shape[0]].astype(np.float64)).astype(np.float32)

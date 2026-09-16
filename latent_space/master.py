"""Output dynamics (hires extension, user-authorised): compressor -> peak normalisation -> limiter.
Applied after the official eq. (1) render; the goal-hold exactness is verified on the raw render
and the applied static/dynamic gains are recorded in the trace."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _db(x: float) -> float:
    return 20.0 * np.log10(max(1e-12, x))


def _smooth_env(x: np.ndarray, fs: int, attack_ms: float, release_ms: float) -> np.ndarray:
    a_att = float(np.exp(-1.0 / max(1.0, attack_ms * 1e-3 * fs)))
    a_rel = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * fs)))
    out = np.empty_like(x)
    e = 0.0
    for n in range(len(x)):
        v = x[n]
        e = a_att * e + (1 - a_att) * v if v > e else a_rel * e + (1 - a_rel) * v
        out[n] = e
    return out


def compressor(y: np.ndarray, fs: int, threshold_dbfs: float, ratio: float, attack_ms: float, release_ms: float,
               knee_db: float = 6.0, block: int = 64) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Feed-forward RMS compressor on a block grid (64 frames) with smoothed gain reduction."""
    n = y.shape[0]
    nb = (n + block - 1) // block
    rms = np.zeros(nb)
    for b in range(nb):
        seg = y[b * block:(b + 1) * block]
        rms[b] = float(np.sqrt((seg.astype(np.float64) ** 2).mean() + 1e-20))
    lvl = 20.0 * np.log10(np.maximum(rms, 1e-12))
    over = lvl - threshold_dbfs
    # soft knee
    k = knee_db
    gr = np.where(over <= -k / 2, 0.0,
                  np.where(over >= k / 2, (1.0 / ratio - 1.0) * over,
                           (1.0 / ratio - 1.0) * (over + k / 2) ** 2 / (2.0 * k)))
    g_lin = 10 ** (gr / 20.0)
    env = _smooth_env(g_lin, fs / block, attack_ms, release_ms)
    gain = np.repeat(env, block)[:n]
    return y * gain[:, None].astype(y.dtype), {"max_gain_reduction_db": float(20 * np.log10(max(1e-12, env.min()))),
                                              "mean_gain_reduction_db": float(20 * np.log10(max(1e-12, env.mean())))}


def limiter(y: np.ndarray, fs: int, ceiling_dbfs: float, lookahead_ms: float, release_ms: float,
            block: int = 32) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Look-ahead peak limiter: block peak -> required gain -> attack over the look-ahead, release."""
    ceil = 10 ** (ceiling_dbfs / 20.0)
    n = y.shape[0]
    nb = (n + block - 1) // block
    peak = np.array([float(np.max(np.abs(y[b * block:(b + 1) * block]))) for b in range(nb)])
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-12))
    la = max(1, int(round(lookahead_ms * 1e-3 * fs / block)))
    # attack: the gain must reach the required value la blocks before the peak -> running minimum
    req = need.copy()
    for b in range(nb):
        lo, hi = b, min(nb, b + la + 1)
        req[b] = need[lo:hi].min()
    rel = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * fs / block)))
    g = np.empty(nb)
    cur = 1.0
    for b in range(nb):
        cur = req[b] if req[b] < cur else rel * cur + (1 - rel) * req[b]
        g[b] = min(cur, req[b])
    gain = np.repeat(g, block)[:n]
    out = y * gain[:, None].astype(y.dtype)
    return out, {"max_gain_reduction_db": float(20 * np.log10(max(1e-12, g.min()))),
                 "blocks_limited": int(np.sum(need < 1.0)), "output_peak": float(np.max(np.abs(out)))}


def master_chain(y: np.ndarray, fs: int, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"enabled": True, "input_peak": float(np.max(np.abs(y)))}
    out = y.astype(np.float64)
    c = cfg.get("compressor", {})
    if c.get("enabled", False):
        out, ci = compressor(out, fs, float(c["threshold_dbfs"]), float(c["ratio"]), float(c["attack_ms"]),
                             float(c["release_ms"]), float(c.get("knee_db", 6.0)))
        info["compressor"] = ci
    tp = cfg.get("normalize_peak_dbfs")
    if tp is not None:
        pk = float(np.max(np.abs(out)))
        g = (10 ** (float(tp) / 20.0)) / max(pk, 1e-12)
        out = out * g
        info["normalize_gain_db"] = float(20 * np.log10(g))
    l = cfg.get("limiter", {})
    if l.get("enabled", False):
        out, li = limiter(out, fs, float(l["ceiling_dbfs"]), float(l["lookahead_ms"]), float(l["release_ms"]))
        info["limiter"] = li
    info["output_peak"] = float(np.max(np.abs(out)))
    return out.astype(np.float32), info

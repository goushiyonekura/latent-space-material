"""Common soft objective (spec §5.3): E_form, E_hist, E_motion, J, selection temperature,
and the phase-relative comparison grid used by E_hist / parents / memory."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .curves import MotionLimits, TrackCurve

SLOT_OF_PHASE = {"INTRO": "OPENING", "REOPEN": "OPENING", "OPEN": "OPEN", "CONTRACT": "CONTRACT"}
SLOTS = ("OPENING", "OPEN", "CONTRACT")


def resample_rows(x: np.ndarray, P: int) -> np.ndarray:
    """Linear resampling of (n, d) rows onto P points (index-space interpolation)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros((P, x.shape[1]))
    if n == 1:
        return np.repeat(x, P, axis=0)
    src = np.linspace(0.0, n - 1.0, P)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    return x[lo] * (1.0 - w) + x[hi] * w


class Objective:
    def __init__(self, cfg: dict, lim: MotionLimits, fs: int):
        o = cfg["objective"]
        self.lim = lim
        self.fs = int(fs)
        self.w_mode = float(o["w_mode"])
        self.w_form = float(o["w_form"])
        self.w_hist = float(o["w_hist"])
        self.w_motion = float(o["w_motion"])
        self.open_target = float(o["open_contribution_target"])
        self.sigma_hist = float(o["sigma_hist"])
        self.P = int(o["comparison_points_per_phase"])
        self.temp_base = float(o["selection_temperature_base"])
        self.temp_open = float(o["selection_temperature_open"])
        self.mode_scale = dict(o["mode_error_scale"])
        self.tolerance = float(cfg["search"]["normalized_mode_tolerance"])

    # ------------------------------------------------------------------ E_form
    def e_form(self, unit, xi: np.ndarray, parts: Dict[str, np.ndarray]) -> float:
        fm = unit.free_mask
        if not fm.any():
            return 0.0
        o = unit.o
        d2_goal = unit.dist2(xi, unit.xi_goal)
        contract = unit.phase_names == "CONTRACT"
        opening = (~contract) & fm
        nongoal = parts["c"][:, 1:].sum(axis=1)
        pen = np.zeros(unit.J)
        pen[contract] = (1.0 - o[contract]) ** 2 * d2_goal[contract]
        short = np.maximum(self.open_target - nongoal[opening], 0.0)
        pen[opening] = (o[opening] ** 2) * short ** 2
        return float(pen[fm].mean())

    # ------------------------------------------------------------------ comparison grid
    def phase_blocks(self, unit, xi: np.ndarray) -> Dict[str, np.ndarray]:
        blocks: Dict[str, np.ndarray] = {}
        for slot in SLOTS:
            rows = np.array([SLOT_OF_PHASE.get(p) == slot for p in unit.phase_names]) & unit.free_mask
            if rows.any():
                blocks[slot] = resample_rows(xi[rows], self.P)
        return blocks

    def blocks_to_unit(self, unit, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Map phase-relative blocks back onto a unit's grid (GOAL_HOLD rows = xi_goal;
        slots absent in `blocks` fall back to xi_goal rows)."""
        out = unit.xi_goal.copy()
        for slot in SLOTS:
            rows = np.where(np.array([SLOT_OF_PHASE.get(p) == slot for p in unit.phase_names]) & unit.free_mask)[0]
            if len(rows) == 0 or slot not in blocks:
                continue
            out[rows] = resample_rows(blocks[slot], len(rows))
        return out

    def e_hist(self, unit, xi: np.ndarray, history) -> float:
        if not history.comparison_archive:
            return 0.0
        blocks = self.phase_blocks(unit, xi)
        vals = []
        for arch in history.comparison_archive:
            common = [s for s in SLOTS if s in blocks and s in arch["blocks"]]
            if not common:
                continue
            v = np.concatenate([blocks[s].ravel() for s in common])
            vb = np.concatenate([arch["blocks"][s].ravel() for s in common])
            vals.append(np.exp(-float(((v - vb) ** 2).mean()) / (2.0 * self.sigma_hist ** 2)))
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------ E_motion
    def e_motion(self, curves: List[TrackCurve], unit) -> float:
        n = len(curves) * max(1, unit.end - unit.start)
        return float(sum(cv.motion_energy(self.fs, self.lim) for cv in curves) / n)

    # ------------------------------------------------------------------ J and selection
    def normalize_mode_error(self, mode_cli_name: str, e: float) -> float:
        return float(e) / float(self.mode_scale.get(mode_cli_name, 1.0))

    def total(self, norm_mode_error: float, e_form: float, e_hist: float, e_motion: float) -> float:
        return (self.w_mode * norm_mode_error + self.w_form * e_form
                + self.w_hist * e_hist + self.w_motion * e_motion)

    def selection_temperature(self, mean_openness: float) -> float:
        return self.temp_base + self.temp_open * float(mean_openness)

    def select(self, totals: np.ndarray, mean_openness: float, rng: np.random.Generator,
               rule: str = "argmin", margin: float = 0.0) -> Tuple[int, np.ndarray]:
        """Final selection (audit §6): argmin by default; with rule='softmax_within_margin' a
        temperature draw restricted to candidates with J <= J_min + margin."""
        totals = np.asarray(totals, dtype=np.float64)
        k_min = int(np.argmin(totals))
        w = np.zeros(len(totals))
        if rule != "softmax_within_margin" or margin <= 0.0:
            w[k_min] = 1.0
            return k_min, w
        eligible = totals <= totals[k_min] + margin
        tau = self.selection_temperature(mean_openness)
        z = -(totals - totals[k_min]) / max(tau, 1e-12)
        w[eligible] = np.exp(z[eligible] - z[eligible].max())
        w = w / w.sum()
        k = int(rng.choice(len(totals), p=w))
        return k, w

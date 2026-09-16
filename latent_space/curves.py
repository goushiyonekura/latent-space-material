"""Legal gain curves (spec §2): Q5 ramps + holds, analytic hard-constraint checks.

A track curve is a contiguous list of segments over integer frames.  Segment kinds:
  Q5   : g(t) = a + (b-a) Q(s), Q(s) = 10s^3 - 15s^4 + 6s^5, s = (n - start)/(end - start)
  HOLD : g(t) = a  (labels: ZERO_HOLD, GOAL_HOLD, SCORED_HOLD, START_HOLD)
Every Q5 segment is exactly one monotone motion episode (eq. 6).  Consecutive Q5 segments
must reverse direction (a zero-velocity joint is only allowed as reversal/hold/start/arrival).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

LN10 = math.log(10.0)
QP_MAX = 15.0 / 8.0  # max of Q'(s) at s = 1/2


def Q(s):
    s = np.asarray(s, dtype=np.float64)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def Qp(s):
    s = np.asarray(s, dtype=np.float64)
    u = s * (1.0 - s)
    return 30.0 * u * u


def Qpp(s):
    s = np.asarray(s, dtype=np.float64)
    return 60.0 * s * (1.0 - s) * (1.0 - 2.0 * s)


def Qppp(s):
    s = np.asarray(s, dtype=np.float64)
    return 60.0 * (1.0 - 6.0 * s + 6.0 * s * s)


# mean of Q'(s)^2 and Q''(s)^2 over s in [0,1] (numeric quadrature, fixed at import)
_S = np.linspace(0.0, 1.0, 20001)
MEAN_QP2 = float(np.trapz(Qp(_S) ** 2, _S))
MEAN_QPP2 = float(np.trapz(Qpp(_S) ** 2, _S))
del _S


@dataclass
class MotionLimits:
    velocity_max: float
    velocity_min_bulk: float
    mean_velocity_min: float
    acceleration_max: float
    jerk_max: float
    regularized_db_rate_max: float
    gain_log_epsilon: float
    minimum_move_amplitude: float
    slow_fraction_max: float
    ramp_component_max_seconds: float
    nonzero_scored_holds: bool = False
    rel_margin: float = 1e-6
    bound_tol: float = 1e-9
    db_rate_subintervals: int = 128

    @classmethod
    def from_config(cls, m: dict, numerics: Optional[dict] = None) -> "MotionLimits":
        numerics = numerics or {}
        return cls(
            velocity_max=float(m["velocity_max"]),
            velocity_min_bulk=float(m["velocity_min_bulk"]),
            mean_velocity_min=float(m["mean_velocity_min"]),
            acceleration_max=float(m["acceleration_max"]),
            jerk_max=float(m["jerk_max"]),
            regularized_db_rate_max=float(m["regularized_db_rate_max"]),
            gain_log_epsilon=float(m["gain_log_epsilon"]),
            minimum_move_amplitude=float(m["minimum_move_amplitude"]),
            slow_fraction_max=float(m["slow_fraction_max"]),
            ramp_component_max_seconds=float(m["ramp_component_max_seconds"]),
            nonzero_scored_holds=bool(m.get("nonzero_scored_holds", False)),
            rel_margin=float(numerics.get("motion_relative_margin", 1e-6)),
            bound_tol=float(numerics.get("gain_bound_tolerance", 1e-9)),
            db_rate_subintervals=int(numerics.get("db_rate_subintervals", 128)),
        )


@dataclass
class Segment:
    kind: str          # 'Q5' | 'HOLD'
    start: int         # inclusive frame
    end: int           # exclusive frame
    a: float
    b: float
    label: str = ""    # MOVE | ZERO_HOLD | GOAL_HOLD | SCORED_HOLD | START_HOLD

    @property
    def frames(self) -> int:
        return self.end - self.start

    def duration(self, fs: float) -> float:
        return self.frames / float(fs)

    @property
    def delta(self) -> float:
        return abs(self.b - self.a)

    @property
    def direction(self) -> int:
        return 0 if self.b == self.a else (1 if self.b > self.a else -1)

    def to_dict(self) -> dict:
        return {"type": self.kind, "label": self.label, "start_frame": int(self.start),
                "end_frame": int(self.end), "start_gain": float(self.a), "end_gain": float(self.b)}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(d["type"], int(d["start_frame"]), int(d["end_frame"]),
                   float(d["start_gain"]), float(d["end_gain"]), d.get("label", ""))


# ----------------------------------------------------------------------------- analytic pieces

def q5_peaks(delta: float, T: float) -> Tuple[float, float, float]:
    """Eq. (8): max |g'|, |g''|, |g'''| of a Q5 ramp of amplitude delta over T seconds."""
    return (15.0 * delta / (8.0 * T),
            10.0 * math.sqrt(3.0) * delta / (3.0 * T * T),
            60.0 * delta / (T ** 3))


def q5_min_duration_from_peaks(delta: float, lim: MotionLimits) -> float:
    """Eq. (9) lower bound (velocity / acceleration / jerk)."""
    return max(15.0 * delta / (8.0 * lim.velocity_max),
               math.sqrt(10.0 * math.sqrt(3.0) * delta / (3.0 * lim.acceleration_max)),
               (60.0 * delta / lim.jerk_max) ** (1.0 / 3.0))


def q5_slow_zone(delta: float, T: float, vmin: float) -> Optional[float]:
    """Eq. (10): s_- such that |g'| < vmin for s < s_- (and symmetric at the end).
    Returns None when the whole ramp is slower than vmin (max speed <= vmin)."""
    if delta <= 0.0:
        return None
    r = vmin * T / delta
    if r >= QP_MAX:
        return None
    return 0.5 * (1.0 - math.sqrt(1.0 - 4.0 * math.sqrt(r / 30.0)))


def q5_db_rate_bound(a: float, b: float, T: float, eps_g: float, K: int = 128) -> float:
    """Eq. (11): conservative upper bound of |d lambda/dt| over the ramp, K sub-intervals."""
    delta = abs(b - a)
    if delta == 0.0:
        return 0.0
    s = np.linspace(0.0, 1.0, K + 1)
    g = a + (b - a) * Q(s)
    s0, s1 = s[:-1], s[1:]
    qp_max = np.where(s1 <= 0.5, Qp(s1), np.where(s0 >= 0.5, Qp(s0), QP_MAX))
    g_min = np.minimum(g[:-1], g[1:])
    ratio = np.max(qp_max / (g_min + eps_g))
    return float((20.0 * delta / (T * LN10)) * ratio)


def _slow_rules_ok(delta: float, T: float, lim: MotionLimits) -> bool:
    s_minus = q5_slow_zone(delta, T, lim.velocity_min_bulk)
    if s_minus is None:
        return False
    m = 1.0 + lim.rel_margin
    return (2.0 * s_minus <= lim.slow_fraction_max * m) and (T * s_minus <= lim.ramp_component_max_seconds * m)


def move_bounds(a: float, b: float, lim: MotionLimits) -> Optional[Tuple[float, float]]:
    """Feasible duration interval [T_lo, T_hi] (seconds) for a Q5 move a->b, or None."""
    delta = abs(b - a)
    if delta < lim.minimum_move_amplitude * (1.0 - lim.rel_margin):
        return None
    T_lo = q5_min_duration_from_peaks(delta, lim)
    bound = q5_db_rate_bound(a, b, T_lo, lim.gain_log_epsilon, lim.db_rate_subintervals)
    if bound > lim.regularized_db_rate_max:
        T_lo = T_lo * bound / lim.regularized_db_rate_max  # bound is exactly proportional to 1/T
    T_hi = delta / lim.mean_velocity_min
    if T_hi < T_lo:
        return None
    if not _slow_rules_ok(delta, T_lo, lim):
        return None
    if _slow_rules_ok(delta, T_hi, lim):
        return (T_lo, T_hi)
    lo, hi = T_lo, T_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _slow_rules_ok(delta, mid, lim):
            lo = mid
        else:
            hi = mid
    return (T_lo, lo)


def feasible_delta_range(T: float, lim: MotionLimits, start: float, direction: int,
                         n_grid: int = 60) -> Optional[Tuple[float, float]]:
    """Range of amplitudes delta such that a Q5 move of duration T from `start` in
    `direction` is legal (scanned on a grid; conservative)."""
    max_delta = (1.0 - start) if direction > 0 else start
    if max_delta < lim.minimum_move_amplitude:
        return None
    deltas = np.linspace(lim.minimum_move_amplitude, max_delta, n_grid)
    ok = []
    for d in deltas:
        end = start + direction * d
        bnd = move_bounds(start, end, lim)
        ok.append(bnd is not None and bnd[0] * (1.0 + 1e-9) <= T <= bnd[1] * (1.0 - 1e-9))
    ok = np.array(ok)
    if not ok.any():
        return None
    idx = np.where(ok)[0]
    return (float(deltas[idx[0]]), float(deltas[idx[-1]]))


# ----------------------------------------------------------------------------- segment checks

def check_segment(seg: Segment, fs: float, lim: MotionLimits) -> List[str]:
    v: List[str] = []
    tol = lim.bound_tol
    if seg.end <= seg.start:
        v.append("empty_segment")
        return v
    for x, nm in ((seg.a, "start_gain"), (seg.b, "end_gain")):
        if not (-tol <= x <= 1.0 + tol) or not math.isfinite(x):
            v.append(f"{nm}_out_of_range:{x}")
    if seg.kind == "HOLD":
        if seg.a != seg.b:
            v.append("hold_not_constant")
        if seg.label == "GOAL_HOLD" or seg.label == "START_HOLD":
            pass
        elif abs(seg.a) <= tol:
            pass  # ZERO_HOLD
        elif not lim.nonzero_scored_holds:
            v.append(f"nonzero_hold_not_allowed:{seg.label}:{seg.a}")
        return v
    if seg.kind != "Q5":
        v.append(f"unknown_kind:{seg.kind}")
        return v
    delta = seg.delta
    T = seg.duration(fs)
    m = 1.0 + lim.rel_margin
    if delta < lim.minimum_move_amplitude / m:
        v.append(f"move_amplitude_below_min:{delta}")
    vmax, amax, jmax = q5_peaks(delta, T)
    if vmax > lim.velocity_max * m:
        v.append(f"velocity_max_exceeded:{vmax}")
    if amax > lim.acceleration_max * m:
        v.append(f"acceleration_max_exceeded:{amax}")
    if jmax > lim.jerk_max * m:
        v.append(f"jerk_max_exceeded:{jmax}")
    if delta / T < lim.mean_velocity_min / m:
        v.append(f"mean_velocity_below_min:{delta / T}")
    s_minus = q5_slow_zone(delta, T, lim.velocity_min_bulk)
    if s_minus is None:
        v.append("peak_velocity_not_above_vmin_bulk")
    else:
        if 2.0 * s_minus > lim.slow_fraction_max * m:
            v.append(f"slow_fraction_exceeded:{2.0 * s_minus}")
        if T * s_minus > lim.ramp_component_max_seconds * m:
            v.append(f"ramp_component_exceeded:{T * s_minus}")
    dbr = q5_db_rate_bound(seg.a, seg.b, T, lim.gain_log_epsilon, lim.db_rate_subintervals)
    if dbr > lim.regularized_db_rate_max * m:
        v.append(f"db_rate_exceeded:{dbr}")
    return v


@dataclass
class TrackCurve:
    segments: List[Segment] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.segments[0].start

    @property
    def end(self) -> int:
        return self.segments[-1].end

    def values(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        out = np.empty(frames.shape, dtype=np.float64)
        out.fill(np.nan)
        starts = np.array([s.start for s in self.segments])
        idx = np.searchsorted(starts, frames, side="right") - 1
        idx = np.clip(idx, 0, len(self.segments) - 1)
        for k, seg in enumerate(self.segments):
            mask = idx == k
            if not mask.any():
                continue
            if seg.kind == "HOLD":
                out[mask] = seg.a
            else:
                s = (frames[mask] - seg.start) / float(seg.frames)
                out[mask] = seg.a + (seg.b - seg.a) * Q(np.clip(s, 0.0, 1.0))
        return out

    def value_at(self, frame: int) -> float:
        return float(self.values(np.array([frame]))[0])

    def velocity_after(self, frame: int) -> float:
        """Direction/speed of the segment containing `frame` (0 for holds)."""
        for seg in self.segments:
            if seg.start <= frame < seg.end:
                return float(seg.direction)
        return 0.0

    def moves(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "Q5"]

    def check(self, fs: float, lim: MotionLimits) -> List[str]:
        v: List[str] = []
        segs = self.segments
        if not segs:
            return ["no_segments"]
        for k, seg in enumerate(segs):
            for msg in check_segment(seg, fs, lim):
                v.append(f"seg{k}:{msg}")
            if k > 0:
                prev = segs[k - 1]
                if prev.end != seg.start:
                    v.append(f"seg{k}:not_contiguous")
                if abs(prev.b - seg.a) > lim.bound_tol:
                    v.append(f"seg{k}:value_discontinuity:{prev.b}->{seg.a}")
                if prev.kind == "Q5" and seg.kind == "Q5" and prev.direction == seg.direction:
                    v.append(f"seg{k}:same_direction_stop")
        return v

    def to_list(self) -> List[dict]:
        return [s.to_dict() for s in self.segments]

    @classmethod
    def from_list(cls, lst: List[dict]) -> "TrackCurve":
        return cls([Segment.from_dict(d) for d in lst])

    def motion_energy(self, fs: float, lim: MotionLimits) -> float:
        """Sum over segments of frames * mean[(g'/Vmax)^2 + (g''/Amax)^2]  (for E_motion)."""
        tot = 0.0
        for seg in self.segments:
            if seg.kind != "Q5":
                continue
            T = seg.duration(fs)
            d = seg.delta
            tot += seg.frames * ((d / T / lim.velocity_max) ** 2 * MEAN_QP2
                                 + (d / (T * T) / lim.acceleration_max) ** 2 * MEAN_QPP2)
        return tot


def curve_from_waypoints(points: List[Tuple[int, float, str]]) -> TrackCurve:
    """points: list of (frame, value, label_of_segment_starting_here).  Consecutive equal
    values become HOLD segments (label from the point), unequal become Q5 MOVE."""
    segs: List[Segment] = []
    for (f0, v0, lab), (f1, v1, _) in zip(points[:-1], points[1:]):
        if f1 <= f0:
            continue
        if v0 == v1:
            segs.append(Segment("HOLD", int(f0), int(f1), float(v0), float(v0), lab or "ZERO_HOLD"))
        else:
            segs.append(Segment("Q5", int(f0), int(f1), float(v0), float(v1), "MOVE"))
    return TrackCurve(segs)

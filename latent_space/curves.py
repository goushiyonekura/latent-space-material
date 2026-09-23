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


@dataclass
class Bump:
    """Local additive correction with zero value / velocity / acceleration at both ends
    (audit C6.1):  dg(t) = amplitude * 64 s^3 (1-s)^3,  s = (n - start)/(end - start)."""
    start: int
    end: int
    amplitude: float

    @property
    def frames(self) -> int:
        return self.end - self.start

    def _s(self, frames: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(frames, dtype=np.float64) - self.start) / float(self.frames), 0.0, 1.0)

    def values(self, frames: np.ndarray) -> np.ndarray:
        s = self._s(frames)
        inside = (frames >= self.start) & (frames < self.end)
        return np.where(inside, self.amplitude * 64.0 * s ** 3 * (1.0 - s) ** 3, 0.0)

    def derivatives(self, frames: np.ndarray, fs: float):
        """(dg, dg1, dg2, dg3): value and first three time derivatives (gain, gain/s, gain/s^2, gain/s^3)."""
        s = self._s(frames)
        inside = (frames >= self.start) & (frames < self.end)
        h = self.frames / float(fs)
        a = self.amplitude
        f0 = 64.0 * s ** 3 * (1.0 - s) ** 3
        f1 = 192.0 * s ** 2 * (1.0 - s) ** 2 * (1.0 - 2.0 * s)
        f2 = 384.0 * s * (1.0 - s) * (1.0 - 5.0 * s + 5.0 * s * s)
        f3 = 384.0 * ((1.0 - 2.0 * s) * (1.0 - 5.0 * s + 5.0 * s * s) + (s - s * s) * (-5.0 + 10.0 * s))
        z = np.zeros_like(s)
        return (np.where(inside, a * f0, z), np.where(inside, a * f1 / h, z),
                np.where(inside, a * f2 / (h * h), z), np.where(inside, a * f3 / (h ** 3), z))

    def to_dict(self) -> dict:
        return {"start_frame": int(self.start), "end_frame": int(self.end), "amplitude": float(self.amplitude)}

    @classmethod
    def from_dict(cls, d: dict) -> "Bump":
        return cls(int(d["start_frame"]), int(d["end_frame"]), float(d["amplitude"]))


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
class Clip:
    """Playback-position map piece (hires extension, user-authorised): from output frame
    `out_start` the source plays from source frame `src_start` (advancing 1:1, modulo length)."""
    out_start: int
    src_start: int

    def to_dict(self) -> dict:
        return {"out_start": int(self.out_start), "src_start": int(self.src_start)}


@dataclass
class TrackCurve:
    segments: List[Segment] = field(default_factory=list)
    bumps: List[Bump] = field(default_factory=list)
    clips: List[Clip] = field(default_factory=list)      # empty = continuous clock (position = output frame)

    @property
    def start(self) -> int:
        return self.segments[0].start

    @property
    def end(self) -> int:
        return self.segments[-1].end

    def base_values(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        out = np.empty(frames.shape, dtype=np.float64)
        out.fill(np.nan)
        starts = np.array([s.start for s in self.segments])
        idx = np.searchsorted(starts, frames, side="right") - 1
        idx = np.clip(idx, 0, len(self.segments) - 1)
        for k in np.unique(idx):                      # only the segments the frames actually hit
            seg = self.segments[int(k)]
            mask = idx == k
            if seg.kind == "HOLD":
                out[mask] = seg.a
            else:
                s = (frames[mask] - seg.start) / float(seg.frames)
                out[mask] = seg.a + (seg.b - seg.a) * Q(np.clip(s, 0.0, 1.0))
        return out

    def _bumps_overlapping(self, frames: np.ndarray) -> List["Bump"]:
        if not self.bumps or len(frames) == 0:
            return []
        fmin, fmax = int(frames.min()), int(frames.max())
        return [b for b in self.bumps if b.end > fmin and b.start <= fmax]

    def values(self, frames: np.ndarray) -> np.ndarray:
        """Composite gain: base segments plus local bump corrections."""
        frames = np.asarray(frames)
        out = self.base_values(frames)
        for b in self._bumps_overlapping(frames):
            out = out + b.values(frames)
        return out

    def value_at(self, frame: int) -> float:
        return float(self.values(np.array([frame]))[0])

    def derivatives(self, frames: np.ndarray, fs: float):
        """Analytic (g, g1, g2, g3) of the composite curve at integer frames: gain, gain/s,
        gain/s^2, gain/s^3 (audit B12: real derivatives, not segment directions)."""
        frames = np.asarray(frames)
        g = np.empty(frames.shape, dtype=np.float64)
        v = np.zeros(frames.shape, dtype=np.float64)
        a = np.zeros(frames.shape, dtype=np.float64)
        j = np.zeros(frames.shape, dtype=np.float64)
        starts = np.array([sg.start for sg in self.segments])
        idx = np.clip(np.searchsorted(starts, frames, side="right") - 1, 0, len(self.segments) - 1)
        for k in np.unique(idx):                      # only the segments the frames actually hit
            seg = self.segments[int(k)]
            mask = idx == k
            if seg.kind == "HOLD":
                g[mask] = seg.a
                continue
            T = seg.frames / float(fs)
            s = np.clip((frames[mask] - seg.start) / float(seg.frames), 0.0, 1.0)
            d = seg.b - seg.a
            g[mask] = seg.a + d * Q(s)
            v[mask] = d * Qp(s) / T
            a[mask] = d * Qpp(s) / (T * T)
            j[mask] = d * Qppp(s) / (T ** 3)
        for b in self._bumps_overlapping(frames):
            d0, d1, d2, d3 = b.derivatives(frames, fs)
            g, v, a, j = g + d0, v + d1, a + d2, j + d3
        return g, v, a, j

    def state_at(self, frame: int, fs: float):
        """(gain, velocity, acceleration) at an integer frame."""
        g, v, a, _ = self.derivatives(np.array([frame]), fs)
        return float(g[0]), float(v[0]), float(a[0])

    def direction_after(self, frame: int) -> int:
        """Direction (+1/-1/0) of the *base* segment containing `frame` (not a velocity)."""
        for seg in self.segments:
            if seg.start <= frame < seg.end:
                return int(seg.direction)
        return 0

    def add_bump(self, bump: "Bump") -> "TrackCurve":
        return TrackCurve(self.segments, self.bumps + [bump], self.clips)

    def positions(self, frames: np.ndarray, length: int) -> np.ndarray:
        """Source position (frame index into the source, modulo its length) for output frames."""
        frames = np.asarray(frames, dtype=np.int64)
        if not self.clips:
            return frames % length
        starts = np.array([c.out_start for c in self.clips], dtype=np.int64)
        idx = np.searchsorted(starts, frames, side="right") - 1
        out = np.empty(frames.shape, dtype=np.int64)
        for k in np.unique(idx):                      # only the clips the frames actually hit
            if k < 0:
                continue
            c = self.clips[int(k)]
            m = idx == k
            out[m] = (c.src_start + (frames[m] - c.out_start)) % length
        m = idx < 0
        if m.any():
            out[m] = frames[m] % length
        return out

    def check_hires(self, fs: float, lim: MotionLimits, cap_intervals: Optional[List[Tuple[int, int, float]]] = None) -> List[str]:
        """Hires checks (user-authorised profile): gain bounds, value continuity at joints, Q5/HOLD
        kinds only, goal caps; the slow-motion rules are waived by authorisation."""
        v: List[str] = []
        segs = self.segments
        if not segs:
            return ["no_segments"]
        tol = lim.bound_tol
        for k, seg in enumerate(segs):
            if seg.kind not in ("Q5", "HOLD"):
                v.append(f"seg{k}:unknown_kind:{seg.kind}")
            if seg.end <= seg.start:
                v.append(f"seg{k}:empty_segment")
            for x, nm in ((seg.a, "start_gain"), (seg.b, "end_gain")):
                if not (-tol <= x <= 1.0 + tol) or not math.isfinite(x):
                    v.append(f"seg{k}:{nm}_out_of_range:{x}")
            if seg.kind == "HOLD" and seg.a != seg.b:
                v.append(f"seg{k}:hold_not_constant")
            if k > 0:
                prev = segs[k - 1]
                if prev.end != seg.start:
                    v.append(f"seg{k}:not_contiguous")
                if abs(prev.b - seg.a) > tol:
                    v.append(f"seg{k}:value_discontinuity:{prev.b}->{seg.a}")
        if cap_intervals:
            for (a0, a1, cap) in cap_intervals:
                fr = np.arange(a0, a1, max(1, int(fs // 100)), dtype=np.int64)
                if len(fr) and float(self.values(fr).max()) > cap + tol:
                    v.append(f"cap_exceeded:{a0}")
        return v

    def base_boundaries(self) -> List[int]:
        return [self.segments[0].start] + [sg.end for sg in self.segments]

    def check_composite(self, fs: float, lim: MotionLimits, start: Optional[int] = None,
                        end: Optional[int] = None, sample_hz: float = 100.0,
                        cap_intervals: Optional[List[Tuple[int, int, float]]] = None,
                        max_extension_seconds: float = 40.0) -> List[str]:
        """Sampled hard checks of the composite curve on [start, end) extended to the enclosing
        base-segment boundaries: bounds, velocity / acceleration / jerk / relative dB rate, and
        the motion-episode rules (minimum amplitude, mean velocity, slow zones) applied to the
        *actual* monotone episodes delimited by sign changes of the composite velocity."""
        v: List[str] = []
        if not self.segments:
            return ["no_segments"]
        c0, c1 = self.start, self.end
        if start is None:
            start = c0
        if end is None:
            end = c1
        bounds = self.base_boundaries()
        # span = nearest base boundaries around [start, end) that are NOT straddled by a bump (the
        # composite velocity is exactly zero there), searched within a bounded distance (audit-2
        # runtime fix: no unbounded chain through earlier committed bumps).  If no such boundary
        # exists within the bound, the edge is partial and its cut episode is not judged here.
        ext = int(round(max_extension_seconds * fs))
        lo_cands = [b for b in bounds if start - ext <= b <= start and not any(bp.start < b < bp.end for bp in self.bumps)]
        hi_cands = [b for b in bounds if end <= b <= end + ext and not any(bp.start < b < bp.end for bp in self.bumps)]
        if c0 >= start - ext:
            lo_cands.append(c0)
        if c1 <= end + ext:
            hi_cands.append(c1)
        partial_lo = len(lo_cands) == 0
        partial_hi = len(hi_cands) == 0
        lo = max(lo_cands) if lo_cands else max(c0, start - ext)
        hi = min(hi_cands) if hi_cands else min(c1, end + ext)
        partial_edges = (partial_lo, partial_hi)
        step = max(1, int(round(fs / sample_hz)))
        frames = np.arange(lo, hi, step, dtype=np.int64)
        if len(frames) < 3:
            return v
        g, vel, acc, jerk = self.derivatives(frames, fs)
        m = 1.0 + lim.rel_margin
        tol = lim.bound_tol
        # The derivatives of a bump peak exactly at its endpoints (s = 0 / s = 1: |g'''| = 384|a|/h^3).
        # A uniform sample grid only sees those spikes when its phase happens to align, so the local
        # check of the realizer ([t0, t1) extended to base boundaries) and this final check (the whole
        # curve) could disagree and an illegal composite could be accepted.  The pointwise limits are
        # therefore evaluated on the sample grid PLUS the bump endpoints; the motion-episode analysis
        # below stays on the uniform grid, whose constant spacing its durations assume.
        pf, pg, pv, pa, pj = frames, g, vel, acc, jerk
        if self.bumps:
            ext = np.array(sorted({int(f) for b in self.bumps for f in (b.start, b.end - 1)}), dtype=np.int64)
            ext = np.setdiff1d(ext[(ext >= lo) & (ext < hi)], frames)
            if ext.size:
                eg, ev, ea, ej = self.derivatives(ext, fs)
                pf = np.concatenate([frames, ext])
                pg, pv = np.concatenate([g, eg]), np.concatenate([vel, ev])
                pa, pj = np.concatenate([acc, ea]), np.concatenate([jerk, ej])
        if not np.all(np.isfinite(pg)) or pg.max() > 1.0 + tol or pg.min() < -tol:
            v.append(f"composite_gain_out_of_bounds:[{pg.min():.4g},{pg.max():.4g}]")
        if np.max(np.abs(pv)) > lim.velocity_max * m:
            v.append(f"composite_velocity_max_exceeded:{np.max(np.abs(pv)):.4g}")
        if np.max(np.abs(pa)) > lim.acceleration_max * m:
            v.append(f"composite_acceleration_max_exceeded:{np.max(np.abs(pa)):.4g}")
        if np.max(np.abs(pj)) > lim.jerk_max * m:
            v.append(f"composite_jerk_max_exceeded:{np.max(np.abs(pj)):.4g}")
        dbr = (20.0 / LN10) * np.abs(pv) / (np.clip(pg, 0.0, None) + lim.gain_log_epsilon)
        if np.max(dbr) > lim.regularized_db_rate_max * m:
            v.append(f"composite_db_rate_exceeded:{np.max(dbr):.4g}")
        if cap_intervals:
            for (a0, a1, cap) in cap_intervals:
                mk = (pf >= a0) & (pf < a1)
                if mk.any() and pg[mk].max() > cap + tol:
                    v.append(f"composite_cap_exceeded:{a0}:{pg[mk].max():.4g}>{cap}")
        sign = np.sign(vel)
        sign[np.abs(vel) <= 1e-9] = 0
        dt = step / float(fs)
        k = 0
        n = len(frames)
        while k < n:
            if sign[k] == 0:
                k += 1
                continue
            k2 = k
            while k2 + 1 < n and sign[k2 + 1] == sign[k]:
                k2 += 1
            ks, ke = max(0, k - 1), min(n - 1, k2 + 1)
            touches_edge = (k == 0 and partial_edges[0]) or (k2 == n - 1 and partial_edges[1])
            amp = abs(float(g[ke] - g[ks]))
            T = (ke - ks) * dt
            if T > 0 and not touches_edge:
                if amp < lim.minimum_move_amplitude / m:
                    v.append(f"episode_amplitude_below_min@{int(frames[ks])}:{amp:.4g}")
                if amp / T < lim.mean_velocity_min / m:
                    v.append(f"episode_mean_velocity_below_min@{int(frames[ks])}:{amp / T:.4g}")
                sp = np.abs(vel[ks:ke + 1])
                if sp.max() <= lim.velocity_min_bulk:
                    v.append(f"episode_peak_velocity_not_above_vmin@{int(frames[ks])}")
                slow = sp < lim.velocity_min_bulk
                if slow.mean() > lim.slow_fraction_max * m + 1e-9:
                    v.append(f"episode_slow_fraction_exceeded@{int(frames[ks])}:{slow.mean():.3f}")
                run = 0
                for z in slow:
                    run = run + 1 if z else 0
                    if run * dt > lim.ramp_component_max_seconds * m:
                        v.append(f"episode_ramp_component_exceeded@{int(frames[ks])}")
                        break
            k = k2 + 1
        return v

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
        out = [s.to_dict() for s in self.segments]
        if self.bumps:
            out.append({"type": "BUMPS", "bumps": [b.to_dict() for b in self.bumps]})
        if self.clips:
            out.append({"type": "CLIPS", "clips": [c.to_dict() for c in self.clips]})
        return out

    @classmethod
    def from_list(cls, lst: List[dict]) -> "TrackCurve":
        segs = [Segment.from_dict(d) for d in lst if d.get("type") not in ("BUMPS", "CLIPS")]
        bumps = [Bump.from_dict(b) for d in lst if d.get("type") == "BUMPS" for b in d["bumps"]]
        clips = [Clip(int(c["out_start"]), int(c["src_start"])) for d in lst if d.get("type") == "CLIPS" for c in d["clips"]]
        return cls(segs, bumps, clips)

    def motion_energy(self, fs: float, lim: MotionLimits) -> float:
        """Sum over segments of frames * mean[(g1/Vmax)^2 + (g2/Amax)^2]  (for E_motion).
        With bumps the composite is sampled; otherwise the analytic Q5 integrals are used."""
        if self.bumps:
            step = max(1, int(round(fs / 50.0)))
            fr = np.arange(self.start, self.end, step, dtype=np.int64)
            _g, vel, acc, _j = self.derivatives(fr, fs)
            return float(((vel / lim.velocity_max) ** 2 + (acc / lim.acceleration_max) ** 2).mean() * (self.end - self.start))
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

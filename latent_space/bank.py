"""Candidate bank (spec §12): small sets of *legal* joint gain trajectories built from Q5
ramps and zero/goal holds, plus mutation operators (levels, reversal times, joint
sync/counter structure between tracks) used by the bounded search.

Audit revision (2026-09-16):
  * goal-exposure policy (form.goal_exposure): the goal track stays at/below explicit caps in
    INTRO/OPEN, rises continuously inside CONTRACT to exactly 1 at the goal time, inherits 1 at
    REOPEN and descends legally (never an instantaneous cut);
  * motion distribution: the number of moves per track is derived from the available time and
    the legal move durations (no small fixed maximum, no bias toward few moves); amplitudes come
    from an explicit small/medium/large mixture and each move's duration is drawn inside its
    legal range with a short bias, so short and long motions coexist instead of a few motions
    being stretched over the whole phase.  Hard limits and the exact goal tail are unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .curves import (MotionLimits, TrackCurve, curve_from_waypoints, feasible_delta_range,
                     move_bounds)
from .types import Candidate, UnitContext

Point = Tuple[int, float, str]
Hint = Tuple[int, int, str]   # (source track, derived track, 'sync' | 'counter')

_MARGIN = 1e-4


class Bank:
    def __init__(self, unit: UnitContext, lim: MotionLimits, cfg: dict, rng: np.random.Generator,
                 objective, history):
        self.unit = unit
        self.lim = lim
        self.rng = rng
        self.objective = objective
        self.history = history
        self.fs = unit.fs
        self.M = unit.M
        self.tol = lim.bound_tol
        self.dmin = lim.minimum_move_amplitude
        s = cfg["search"]
        self.attempts = int(s["bank_generation_attempts"])
        self.t0 = unit.start
        self.t1 = unit.goal_arrival if unit.goal_arrival is not None else unit.end
        self.t_end = unit.end
        self.has_goal = unit.goal_arrival is not None
        self.D = (self.t1 - self.t0) / float(self.fs)
        self.hold_max = min(20.0, 0.5 * self.D)
        # audit B2: the move count follows from the available time and the shortest legal move,
        # never from a fixed per-cycle maximum; the configured value is only a floor of the cap
        b_min = move_bounds(0.0, max(lim.minimum_move_amplitude, 0.05), lim)
        t_shortest = b_min[0] if b_min is not None else 2.0
        self.K_cap = max(int(s["max_moves_per_track"]), int(self.D / max(1e-6, t_shortest)) + 2)
        # goal exposure policy
        ge = dict(cfg["form"].get("goal_exposure", {}))
        self.goal_policy = str(ge.get("policy", "contract_only"))
        self.intro_max = float(ge.get("intro_max", 0.0))
        self.open_max = float(ge.get("open_max", 0.0))
        self.descend_within_reopen = bool(ge.get("reopen_descend_within_reopen", True))
        pf = unit.phase_frames
        self.contract_start = pf["CONTRACT"][0] if "CONTRACT" in pf else None
        self.reopen_end = pf["REOPEN"][1] if "REOPEN" in pf else None
        self.intro_end = pf["INTRO"][1] if "INTRO" in pf else None
        # amplitude mixture and duration preference
        am = dict(cfg["search"].get("amplitude_mixture", {}))
        self.amp_probs = np.array(am.get("probabilities", [0.35, 0.40, 0.25]), dtype=np.float64)
        self.amp_probs = self.amp_probs / self.amp_probs.sum()
        self.amp_ranges = am.get("ranges", [[0.05, 0.2], [0.2, 0.5], [0.5, 1.0]])
        self.duration_beta = tuple(cfg["search"].get("duration_preference_beta", [1.0, 3.0]))
        self.n_generated = 0
        self.n_failed = 0
        self.next_id = 0
        self.notes: List[str] = []
        self.rel_stats: Dict[str, int] = {"attempted": 0, "applied": 0, "infeasible_boundary": 0, "infeasible_other": 0}

    # ------------------------------------------------------------------ helpers
    def goal_value(self, i: int) -> float:
        return 1.0 if i == 0 else 0.0

    @property
    def goal_constrained(self) -> bool:
        return self.goal_policy != "free"

    def _first_dirs(self, v0: float, cap: float = 1.0) -> List[int]:
        dirs = []
        if v0 + self.dmin <= cap + self.tol:
            dirs.append(1)
        if v0 - self.dmin >= -self.tol:
            dirs.append(-1)
        return dirs

    def _last_dir(self, v_end: Optional[float], cap: float = 1.0) -> Optional[int]:
        if v_end is None:
            return None
        if v_end >= cap - self.tol:
            return 1
        if v_end <= self.tol:
            return -1
        return None

    def _sec_to_frame(self, sec: float, origin: int) -> int:
        return origin + int(round(sec * self.fs))

    def _sample_amplitude(self) -> float:
        k = int(self.rng.choice(len(self.amp_probs), p=self.amp_probs))
        lo, hi = self.amp_ranges[k]
        return float(self.rng.uniform(lo, hi))

    def _sample_duration(self, lo: float, hi: float) -> float:
        a, b = self.duration_beta
        return float(lo + self.rng.beta(a, b) * (hi - lo))

    # ------------------------------------------------------------------ generic bounded segment
    def _segment_plan(self, v0: float, t_start: int, t_end: int, v_end: Optional[float],
                      cap: float = 1.0, allow_initial_hold: bool = True,
                      first_dir_required: Optional[int] = None,
                      last_dir_required: Optional[int] = None) -> Optional[List[Point]]:
        """Waypoints (frame, value, label) on [t_start, t_end] starting at v0 (zero velocity),
        ending exactly at v_end at t_end (or a free level when v_end is None), all levels in
        [0, cap].  Moves alternate direction; zero holds are allowed at level 0.  Number of
        moves follows from the available time and the legal durations."""
        rng = self.rng
        D = (t_end - t_start) / float(self.fs)
        if D <= 0:
            return None
        if cap < self.dmin:  # no motion possible: hold at v0 (must be 0)
            if v0 > self.tol or (v_end is not None and v_end > self.tol):
                return None
            return [(t_start, v0, "ZERO_HOLD"), (t_end, v0, "")]
        first_dirs = self._first_dirs(v0, cap)
        if first_dir_required is not None:
            first_dirs = [d for d in first_dirs if d == first_dir_required]
        if not first_dirs:
            return None
        ld = self._last_dir(v_end, cap)
        if ld is None and last_dir_required is not None:
            ld = int(last_dir_required)
        hold_max = min(self.hold_max, 0.5 * D)
        for _ in range(80):
            levels = [v0]
            prefs: List[Tuple[float, float, float]] = []
            holds_after: Dict[int, float] = {}
            h0 = 0.0
            if allow_initial_hold and v0 <= self.tol and rng.random() < 0.6:
                h0 = float(rng.uniform(0.0, min(hold_max, 0.3 * D)))
            total = h0
            d = int(rng.choice(first_dirs))
            ok = False
            fail = False
            while True:
                prev = levels[-1]
                # can we finish now with one legal move to v_end?
                if v_end is not None:
                    dv = v_end - prev
                    if dv * d > 0 and abs(dv) >= self.dmin and (ld is None or d == ld):
                        bnd = move_bounds(prev, v_end, self.lim)
                        rem = D - total
                        if bnd is not None and bnd[0] * (1 + _MARGIN) <= rem <= bnd[1] * (1 - _MARGIN):
                            levels.append(v_end)
                            prefs.append((rem, bnd[0] * (1 + _MARGIN), bnd[1] * (1 - _MARGIN)))
                            total += rem
                            ok = True
                            break
                        if bnd is not None and rem < bnd[0] * (1 + _MARGIN):
                            fail = True   # overshot: not enough time left for the final approach
                            break
                elif total >= D:
                    ok = True
                    break
                if len(prefs) >= self.K_cap:
                    fail = True
                    break
                # sample the next move (amplitude mixture, direction d, inside [0, cap])
                nxt = None
                bnd = None
                for _t in range(6):
                    amp = self._sample_amplitude()
                    cand = prev + d * amp
                    cand = min(cap, max(0.0, cand))
                    if d < 0 and cand < 0.03:
                        cand = 0.0
                    if v_end is not None and abs(cand - v_end) < self.dmin and cand != v_end:
                        continue
                    if abs(cand - prev) < self.dmin:
                        continue
                    bnd = move_bounds(prev, cand, self.lim)
                    if bnd is not None:
                        nxt = cand
                        break
                if nxt is None:
                    fail = True
                    break
                lo, hi = bnd[0] * (1 + _MARGIN), bnd[1] * (1 - _MARGIN)
                T = self._sample_duration(lo, hi)
                levels.append(nxt)
                prefs.append((T, lo, hi))
                total += T
                if nxt <= self.tol and rng.random() < 0.5:
                    h = float(rng.uniform(0.0, hold_max))
                    holds_after[len(levels) - 1] = h
                    total += h
                d = -d
            if fail or not ok:
                continue
            if ld is not None and v_end is not None:
                # direction of the final move must match the parity implied by v_end
                if (levels[-1] - levels[-2]) * ld <= 0:
                    continue
            pts = self._fit_to_duration(levels, prefs, holds_after, h0, t_start, t_end, D)
            if pts is not None:
                return pts
        return None

    def _fit_to_duration(self, levels, prefs, holds_after, h0, t_start, t_end, D) -> Optional[List[Point]]:
        """Common scale lambda on the preferred durations (each clamped to its legal range) and
        on the holds so that the plan fills exactly D seconds; bisection on lambda."""
        T_pref = np.array([p[0] for p in prefs])
        lo = np.array([p[1] for p in prefs])
        hi = np.array([p[2] for p in prefs])
        hold_keys = sorted(holds_after)
        H = np.array([holds_after[k] for k in hold_keys]) if hold_keys else np.zeros(0)

        def total_for(lam: float) -> float:
            return float(np.clip(lam * T_pref, lo, hi).sum() + lam * (h0 + H.sum()))

        if not (total_for(1e-6) - 1e-9 <= D <= total_for(1e6) + 1e-9):
            return None
        a, b = 1e-6, 1e6
        for _ in range(80):
            m = 0.5 * (a + b)
            if total_for(m) < D:
                a = m
            else:
                b = m
        lam = 0.5 * (a + b)
        T = np.clip(lam * T_pref, lo, hi)
        Hs = lam * H
        h0s = lam * h0
        pts: List[Point] = []
        cursor = 0.0
        if h0s * self.fs >= 1.0:
            pts.append((t_start, levels[0], "ZERO_HOLD"))
            cursor += h0s
            pts.append((self._sec_to_frame(cursor, t_start), levels[0], "MOVE"))
        else:
            pts.append((t_start, levels[0], "MOVE"))
        hi_idx = 0
        K = len(T)
        for k in range(K):
            cursor += float(T[k])
            f = t_end if k == K - 1 else self._sec_to_frame(cursor, t_start)
            level = levels[k + 1]
            key = k + 1
            if key in holds_after and k < K - 1:
                h = float(Hs[hold_keys.index(key)])
                if h * self.fs >= 1.0:
                    pts.append((f, level, "ZERO_HOLD"))
                    cursor += h
                    pts.append((self._sec_to_frame(cursor, t_start), level, "MOVE"))
                else:
                    pts.append((f, level, "MOVE"))
            else:
                pts.append((f, level, "MOVE"))
        # strictly increasing frames and legal rounded durations
        for p, q in zip(pts[:-1], pts[1:]):
            if q[0] <= p[0]:
                return None
        pts[-1] = (t_end, pts[-1][1], "")
        return pts if self._plan_moves_legal(pts) else None

    # ------------------------------------------------------------------ per-track plans
    def random_track_plan(self, i: int, allow_initial_hold: bool = True) -> Optional[List[Point]]:
        if i == 0 and self.goal_constrained:
            return self._goal_track_plan()
        v0 = float(self.unit.start_gains[i])
        vG = self.goal_value(i) if self.has_goal else None
        pts = self._segment_plan(v0, self.t0, self.t1, vG, allow_initial_hold=allow_initial_hold)
        if pts is None:
            return None
        return self._finish_plan(pts, vG)

    def _goal_track_plan(self) -> Optional[List[Point]]:
        """Goal track under form.goal_exposure (policy != 'free')."""
        rng = self.rng
        v0 = float(self.unit.start_gains[0])
        if self.has_goal:
            rise_lo_hi = move_bounds(0.0, 1.0, self.lim)
            if rise_lo_hi is None:
                return None
            lo, hi = rise_lo_hi[0] * (1 + _MARGIN), rise_lo_hi[1] * (1 - _MARGIN)
            c_start = self.contract_start if self.contract_start is not None else self.t0
            contract_secs = (self.t1 - c_start) / float(self.fs)
            if contract_secs >= lo:
                T_rise = float(rng.uniform(lo, min(hi, contract_secs)))
            else:
                T_rise = lo
                self.notes.append("goal rise started before CONTRACT (contract shorter than the legal 0->1 ramp)")
            t_rise = self.t1 - int(round(T_rise * self.fs))
            t_rise = max(self.t0 + 1, t_rise)
        else:
            t_rise = self.t_end
        pts: List[Point] = []
        cursor = self.t0
        if v0 >= 1.0 - self.tol:
            # inherit 1 from GOAL_HOLD and descend legally
            v_open = 0.0 if self.open_max < self.dmin else float(rng.uniform(0.0, self.open_max))
            bnd = move_bounds(1.0, v_open, self.lim)
            if bnd is None:
                return None
            lo_d, hi_d = bnd[0] * (1 + _MARGIN), bnd[1] * (1 - _MARGIN)
            limit = (self.reopen_end - self.t0) / float(self.fs) if (self.reopen_end and self.descend_within_reopen) else (t_rise - self.t0) / float(self.fs)
            limit = min(limit, (t_rise - self.t0) / float(self.fs))
            if limit >= lo_d:
                T_d = float(rng.uniform(lo_d, min(hi_d, limit)))
            else:
                T_d = lo_d
                self.notes.append("goal descent could not finish inside REOPEN; it ends inside OPEN")
            t_d = self.t0 + int(round(T_d * self.fs))
            if t_d >= t_rise:
                return None
            pts.append((self.t0, 1.0, "MOVE"))
            cursor = t_d
            v_cur = v_open
            first_dirs_cap = self.open_max
            if v_cur > self.tol:
                seg = self._segment_plan(v_cur, cursor, t_rise, 0.0, cap=self.open_max, allow_initial_hold=False)
                if seg is None:
                    return None
                pts.extend(seg)
            else:
                seg = self._segment_plan(0.0, cursor, t_rise, 0.0 if self.has_goal else None,
                                         cap=self.open_max, allow_initial_hold=True)
                if seg is None:
                    return None
                pts.extend(seg)
        else:
            # from silence: bounded activity (intro/open caps) ending at 0 at the rise start
            cap = max(self.intro_max, self.open_max)
            seg = self._segment_plan(v0, self.t0, t_rise, 0.0, cap=cap, allow_initial_hold=True)
            if seg is None:
                return None
            if self.intro_max < self.open_max and self.intro_end is not None:
                # enforce the intro cap separately
                tc = curve_from_waypoints(seg)
                fr = np.arange(self.t0, min(self.intro_end, t_rise))
                if tc.segments and len(fr) and float(tc.values(fr).max()) > self.intro_max + self.tol:
                    return None
            pts.extend(seg)
        if self.has_goal:
            f_last, v_last, _ = pts[-1]
            pts[-1] = (f_last, v_last, "MOVE")
            pts.append((self.t1, 1.0, "GOAL_HOLD"))
            if self.t_end > self.t1:
                pts.append((self.t_end, 1.0, ""))
            else:
                pts[-1] = (self.t1, 1.0, "")
        else:
            f_last, v_last, _ = pts[-1]
            pts[-1] = (f_last, v_last, "")
        # dedupe consecutive identical frames created by segment joins
        clean: List[Point] = []
        for p in pts:
            if clean and clean[-1][0] == p[0]:
                clean[-1] = (p[0], p[1], p[2] if p[2] else clean[-1][2])
                continue
            clean.append(p)
        return clean if self._plan_moves_legal(clean) else None

    def _finish_plan(self, pts: List[Point], vG: Optional[float]) -> Optional[List[Point]]:
        for a, b in zip(pts[:-1], pts[1:]):
            if b[0] <= a[0]:
                return None
        if pts[-1][0] != self.t1:
            return None
        if self.has_goal:
            f, v, _ = pts[-1]
            pts[-1] = (f, vG, "GOAL_HOLD")
            if self.t_end > self.t1:
                pts.append((self.t_end, vG, ""))
        else:
            f, v, _ = pts[-1]
            pts[-1] = (f, v, "")
        return pts

    # ------------------------------------------------------------------ derived (joint) plans
    def _parse_base(self, base: List[Point]):
        segs = []
        for (f0, a, lab), (f1, b, _) in zip(base[:-1], base[1:]):
            if f0 >= self.t1:
                break
            if a == b:
                segs.append(("HOLD", f0, min(f1, self.t1), 0))
            else:
                segs.append(("MOVE", f0, f1, 1 if b > a else -1))
        return segs

    def derived_track_plan(self, j: int, base: List[Point], relation: str) -> Optional[List[Point]]:
        """Track j related to `base`.  'sync': same move/hold timing and directions over the whole
        unit (falls back to a local interior relation when the boundary makes it infeasible);
        'counter': opposite direction on one interior interval of the base (audit C3: start-from-0
        and end-at-0 moves cannot be reversed, so the relation is local, never whole-span)."""
        self.rel_stats["attempted"] += 1
        if j == 0 and self.goal_constrained:
            self.rel_stats["infeasible_boundary"] += 1
            return None
        out = None
        if relation == "sync":
            out = self._full_sync_plan(j, base)
            if out is None:
                out = self._local_relation_plan(j, base, +1)
        else:
            out = self._local_relation_plan(j, base, -1)
        if out is not None:
            self.rel_stats["applied"] += 1
        return out

    def _full_sync_plan(self, j: int, base: List[Point]) -> Optional[List[Point]]:
        rng = self.rng
        v0 = float(self.unit.start_gains[j])
        vG = self.goal_value(j) if self.has_goal else None
        segs = self._parse_base(base)
        moves = [sg for sg in segs if sg[0] == "MOVE"]
        if not moves:
            return None
        first_dirs = self._first_dirs(v0)
        if moves[0][3] not in first_dirs:
            self.rel_stats["infeasible_boundary"] += 1
            return None
        ld = self._last_dir(vG)
        if ld is not None and moves[-1][3] != ld:
            self.rel_stats["infeasible_boundary"] += 1
            return None
        pts: List[Point] = []
        level = v0
        n_moves = len(moves)
        mi = 0
        for (kind, f0, f1, d) in segs:
            if kind == "HOLD":
                if level > self.tol:
                    return None
                pts.append((f0, level, "ZERO_HOLD"))
                continue
            T = (f1 - f0) / float(self.fs)
            is_last = mi == n_moves - 1
            if is_last and vG is not None:
                if (vG - level) * d <= 0 or abs(vG - level) < self.dmin:
                    return None
                bnd = move_bounds(level, vG, self.lim)
                if bnd is None or not (bnd[0] * (1 + 1e-9) <= T <= bnd[1] * (1 - 1e-9)):
                    return None
                new_level = vG
            else:
                rg = feasible_delta_range(T, self.lim, level, d)
                if rg is None:
                    return None
                dlo, dhi = rg
                if mi == n_moves - 2 and vG is not None:
                    # audit C2: bound the *next gain* first, then convert to an amplitude range
                    next_lo, next_hi = 0.0, 1.0
                    if ld is not None and ld > 0:      # final move rises to the goal value
                        next_hi = min(next_hi, vG - self.dmin)
                    else:                              # final move descends to the goal value
                        next_lo = max(next_lo, vG + self.dmin)
                    if d > 0:
                        dlo = max(dlo, next_lo - level)
                        dhi = min(dhi, next_hi - level)
                    else:
                        dlo = max(dlo, level - next_hi)
                        dhi = min(dhi, level - next_lo)
                    if dlo > dhi:
                        return None
                delta = float(rng.uniform(dlo, dhi))
                new_level = float(min(1.0, max(0.0, level + d * delta)))
            pts.append((f0, level, "MOVE"))
            level = new_level
            mi += 1
        pts.append((self.t1, level, ""))
        out = self._finish_plan(pts, vG)
        if out is not None and not self._plan_moves_legal(out):
            return None
        return out

    def _local_relation_plan(self, j: int, base: List[Point], sgn: int) -> Optional[List[Point]]:
        """Relation on one interior move of the base (same direction sgn=+1, opposite sgn=-1);
        before and after it track j moves independently and legally."""
        rng = self.rng
        v0 = float(self.unit.start_gains[j])
        vG = self.goal_value(j) if self.has_goal else None
        moves = [sg for sg in self._parse_base(base) if sg[0] == "MOVE"]
        interior = moves[1:-1] if vG is not None else moves[1:]
        if not interior:
            self.rel_stats["infeasible_boundary"] += 1
            return None
        for _ in range(6):
            _k, f0, f1, d_b = interior[int(rng.integers(len(interior)))]
            d_j = sgn * d_b
            T = (f1 - f0) / float(self.fs)
            v_e = float(rng.uniform(0.0, 1.0 - self.dmin)) if d_j > 0 else float(rng.uniform(self.dmin, 1.0))
            rg = feasible_delta_range(T, self.lim, v_e, d_j)
            if rg is None:
                continue
            delta = float(rng.uniform(rg[0], rg[1]))
            v_x = float(min(1.0, max(0.0, v_e + d_j * delta)))
            pre = self._segment_plan(v0, self.t0, f0, v_e, allow_initial_hold=True,
                                     last_dir_required=(-d_j if v_e > self.tol else None))
            if pre is None:
                continue
            post = self._segment_plan(v_x, f1, self.t1, vG, allow_initial_hold=(v_x <= self.tol),
                                      first_dir_required=-d_j)
            if post is None:
                continue
            pts = pre[:-1] + [(f0, v_e, "MOVE")] + post
            pts[-1] = (self.t1, pts[-1][1], "")
            if not self._plan_moves_legal(pts):
                continue
            out = self._finish_plan(pts, vG)
            if out is not None:
                return out
        self.rel_stats["infeasible_other"] += 1
        return None

    # ------------------------------------------------------------------ mutation
    def perturb_levels(self, pts: List[Point], i: int) -> Optional[List[Point]]:
        if i == 0 and self.goal_constrained:
            return None
        rng = self.rng
        pts = list(pts)
        last_free = len(pts) - 1
        if self.has_goal:
            last_free = next(k for k, p in enumerate(pts) if p[0] == self.t1)
        changed = False
        for k in range(1, last_free):
            f, v, lab = pts[k]
            if v <= self.tol and (lab == "ZERO_HOLD" or pts[k - 1][1] == v):
                continue
            if rng.random() < 0.6:
                nv = float(min(1.0, max(0.0, v + rng.normal(0.0, 0.15))))
                pts[k] = (f, nv, lab)
                changed = True
        if not changed:
            return None
        if not self.has_goal and rng.random() < 0.5:
            f, v, lab = pts[-1]
            pts[-1] = (f, float(min(1.0, max(0.0, v + rng.normal(0.0, 0.15)))), lab)
        return pts if self._plan_moves_legal(pts) else None

    def perturb_times(self, pts: List[Point], i: int) -> Optional[List[Point]]:
        if i == 0 and self.goal_constrained:
            return None
        rng = self.rng
        pts = list(pts)
        last_free = len(pts) - 1
        if self.has_goal:
            last_free = next(k for k, p in enumerate(pts) if p[0] == self.t1)
        if last_free < 2:
            return None
        k = int(rng.integers(1, last_free))
        f_prev, f_next = pts[k - 1][0], pts[k + 1][0]
        room = min(pts[k][0] - f_prev, f_next - pts[k][0])
        shift = int(round(rng.uniform(0.05, 0.25) * room)) * int(rng.choice([-1, 1]))
        if shift == 0:
            return None
        f, v, lab = pts[k]
        pts[k] = (f + shift, v, lab)
        if pts[k][0] <= f_prev or pts[k][0] >= f_next:
            return None
        return pts if self._plan_moves_legal(pts) else None

    def _plan_moves_legal(self, pts: List[Point]) -> bool:
        prev_dir = 0
        prev_kind = None
        for (f0, a, lab), (f1, b, _) in zip(pts[:-1], pts[1:]):
            if f1 <= f0:
                return False
            if a == b:
                if a > self.tol and lab != "GOAL_HOLD":
                    return False
                prev_kind = "HOLD"
                continue
            d = 1 if b > a else -1
            if prev_kind == "MOVE" and d == prev_dir:
                return False
            bnd = move_bounds(a, b, self.lim)
            T = (f1 - f0) / float(self.fs)
            if bnd is None or not (bnd[0] <= T <= bnd[1]):
                return False
            prev_dir = d
            prev_kind = "MOVE"
        return True

    # ------------------------------------------------------------------ candidates
    def make_candidate(self, plans: List[List[Point]], origin: str,
                       parent_ids: Optional[List] = None) -> Optional[Candidate]:
        curves = [curve_from_waypoints(p) for p in plans]
        viol: List[str] = []
        for i, cv in enumerate(curves):
            if not cv.segments:
                viol.append(f"track{i}:empty")
                continue
            viol += [f"track{i}:{m}" for m in cv.check(self.fs, self.lim)]
            if cv.start != self.t0 or cv.end != self.t_end:
                viol.append(f"track{i}:span_mismatch")
            if self.has_goal and self.t_end > self.t1:
                seg = cv.segments[-1]
                if not (seg.kind == "HOLD" and seg.start == self.t1 and seg.label == "GOAL_HOLD"
                        and seg.a == self.goal_value(i)):
                    viol.append(f"track{i}:goal_hold_missing")
        if viol:
            self.n_failed += 1
            return None
        gains = np.stack([cv.values(self.unit.centers) for cv in curves], axis=1)
        xi, parts = self.unit.composition(gains)
        cand = Candidate(id=self.next_id, plan=[list(p) for p in plans], curves=curves, gains=gains,
                         xi=xi, parts=parts, origin=origin, parent_ids=list(parent_ids or []))
        self.next_id += 1
        cand.e_form = self.objective.e_form(self.unit, xi, parts)
        cand.e_hist = self.objective.e_hist(self.unit, xi, self.history)
        cand.e_motion = self.objective.e_motion(curves, self.unit)
        self.n_generated += 1
        return cand

    def random_candidate(self, hints: Optional[Sequence[Hint]] = None, origin: str = "bank") -> Optional[Candidate]:
        rng = self.rng
        use_hints = [h for h in (hints or []) if not (self.goal_constrained and (h[0] == 0 or h[1] == 0))]
        for _ in range(8):
            plans: List[Optional[List[Point]]] = [None] * self.M
            for (a, b, rel) in use_hints:
                if a == b or a >= self.M or b >= self.M or rng.random() > 0.7:
                    continue
                if plans[a] is None:
                    plans[a] = self.random_track_plan(a)
                if plans[a] is None or plans[b] is not None:
                    continue
                derived = None
                for _t in range(4):
                    derived = self.derived_track_plan(b, plans[a], rel)
                    if derived is not None:
                        break
                plans[b] = derived
            # audit E3: from silence, at least two materials start without an initial zero hold
            no_hold = set()
            if self.unit.starts_from_silence and self.M > 2:
                no_hold = set(int(x) for x in rng.choice(np.arange(1, self.M), size=min(2, self.M - 1), replace=False))
            for i in range(self.M):
                if plans[i] is None:
                    plans[i] = self.random_track_plan(i, allow_initial_hold=(i not in no_hold))
            if any(p is None for p in plans):
                continue
            cand = self.make_candidate(plans, origin)  # type: ignore[arg-type]
            if cand is not None:
                return cand
        return None

    def mutate(self, cand: Candidate, hints: Optional[Sequence[Hint]] = None,
               kind: Optional[str] = None) -> Optional[Candidate]:
        rng = self.rng
        kinds = ["regen", "levels", "times", "relate"]
        if kind is None:
            kind = str(rng.choice(kinds, p=[0.3, 0.3, 0.2, 0.2]))
        plans = [list(p) for p in cand.plan]
        i = int(rng.integers(self.M))
        if i == 0 and self.goal_constrained and kind in ("levels", "times", "relate"):
            kind = "regen"
        new = None
        if kind == "regen":
            new = self.random_track_plan(i)
        elif kind == "levels":
            new = self.perturb_levels(plans[i], i)
        elif kind == "times":
            new = self.perturb_times(plans[i], i)
        elif kind == "relate":
            use_hints = [h for h in (hints or []) if not (self.goal_constrained and (h[0] == 0 or h[1] == 0))]
            if use_hints:
                a, b, rel = use_hints[int(rng.integers(len(use_hints)))]
            else:
                lo_track = 1 if self.goal_constrained else 0
                if self.M - lo_track < 2:
                    return None
                a, b = rng.choice(np.arange(lo_track, self.M), 2, replace=False)
                rel = str(rng.choice(["sync", "counter"]))
            i = int(b)
            new = self.derived_track_plan(int(b), plans[int(a)], rel)
        if new is None:
            return None
        plans[i] = new
        return self.make_candidate(plans, f"mutate:{kind}", [cand.id])

    def stats(self) -> Dict[str, object]:
        return {"generated": self.n_generated, "rejected_illegal": self.n_failed,
                "goal_policy": self.goal_policy, "notes": sorted(set(self.notes)),
                "relations": dict(self.rel_stats)}

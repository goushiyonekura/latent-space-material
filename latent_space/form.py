"""Form: INTRO -> OPEN -> CONTRACT -> GOAL_HOLD -> REOPEN -> OPEN ... (spec §3).

Planning units are bounded by goal states (all-zero velocity, g = e_0), which are the only
natural zero-velocity boundaries:
  unit 0        : INTRO, OPEN, CONTRACT, GOAL_HOLD
  unit r >= 1   : REOPEN, OPEN, CONTRACT, GOAL_HOLD
  final unit    : REOPEN                     (the last re-opening; the piece ends there)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .curves import MotionLimits, Q, move_bounds

PHASE_NAMES = ("INTRO", "OPEN", "CONTRACT", "GOAL_HOLD", "REOPEN")


class FormInfeasible(Exception):
    pass


@dataclass
class Phase:
    name: str
    start: int
    end: int
    cycle: int
    unit: int

    @property
    def frames(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"name": self.name, "start_frame": int(self.start), "end_frame": int(self.end),
                "cycle": int(self.cycle), "unit": int(self.unit)}


@dataclass
class Unit:
    index: int
    phases: List[Phase]

    @property
    def start(self) -> int:
        return self.phases[0].start

    @property
    def end(self) -> int:
        return self.phases[-1].end

    @property
    def goal_arrival(self) -> Optional[int]:
        for p in self.phases:
            if p.name == "GOAL_HOLD":
                return p.start
        return None

    @property
    def has_goal(self) -> bool:
        return self.goal_arrival is not None

    @property
    def starts_from_silence(self) -> bool:
        return self.phases[0].name == "INTRO"

    @property
    def movable_end(self) -> int:
        return self.goal_arrival if self.has_goal else self.end

    def phase(self, name: str) -> Optional[Phase]:
        for p in self.phases:
            if p.name == name:
                return p
        return None


@dataclass
class FormPlan:
    fs: int
    phases: List[Phase]
    units: List[Unit]
    resolved_seconds: dict = field(default_factory=dict)

    @property
    def total_frames(self) -> int:
        return self.phases[-1].end

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, form_cfg: dict, fs: int, lim: MotionLimits) -> "FormPlan":
        cycles = int(form_cfg["cycles"])
        secs = {k: float(form_cfg[k]) for k in ("intro_seconds", "open_seconds", "contract_seconds",
                                                 "goal_hold_seconds", "reopen_seconds")}
        adaptive = form_cfg.get("timing_policy", "adaptive") == "adaptive"
        s_max = float(form_cfg.get("adaptive_stage_seconds_max", 90))
        # minimal movable time per unit: the goal track must leave and return (two ramps)
        b_full = move_bounds(0.0, 1.0, lim)
        b_half = move_bounds(1.0, 0.5, lim)
        if b_full is None or b_half is None:
            raise FormInfeasible("motion profile admits no legal 0->1 or 1->0.5 ramp")
        need = b_full[0] + b_half[0]
        for _ in range(64):
            unit0 = secs["intro_seconds"] + secs["open_seconds"] + secs["contract_seconds"]
            unit_r = secs["reopen_seconds"] + secs["open_seconds"] + secs["contract_seconds"]
            if unit0 >= need and unit_r >= need:
                break
            if not adaptive:
                raise FormInfeasible(
                    f"fixed stage times give {min(unit0, unit_r):.1f}s of motion per cycle but the "
                    f"motion limits need at least {need:.1f}s (timing_policy=fixed)")
            grown = False
            for k in ("contract_seconds", "open_seconds"):
                if secs[k] < s_max:
                    secs[k] = min(s_max, secs[k] + 5.0)
                    grown = True
                    break
            if not grown:
                raise FormInfeasible("adaptive timing exhausted adaptive_stage_seconds_max")
        else:
            raise FormInfeasible("adaptive timing did not converge")
        phases: List[Phase] = []
        units: List[Unit] = []
        cursor = 0

        def add(name: str, seconds: float, cycle: int, unit: int) -> Phase:
            nonlocal cursor
            n = int(round(seconds * fs))
            p = Phase(name, cursor, cursor + n, cycle, unit)
            cursor += n
            phases.append(p)
            return p

        for c in range(cycles):
            ps: List[Phase] = []
            if c == 0:
                ps.append(add("INTRO", secs["intro_seconds"], c, c))
            else:
                ps.append(add("REOPEN", secs["reopen_seconds"], c, c))
            ps.append(add("OPEN", secs["open_seconds"], c, c))
            ps.append(add("CONTRACT", secs["contract_seconds"], c, c))
            ps.append(add("GOAL_HOLD", secs["goal_hold_seconds"], c, c))
            units.append(Unit(c, ps))
        if form_cfg.get("final_reopen", True):
            p = add("REOPEN", secs["reopen_seconds"], cycles, cycles)
            units.append(Unit(cycles, [p]))
        plan = cls(fs=fs, phases=phases, units=units, resolved_seconds=dict(secs))
        return plan

    # ------------------------------------------------------------------ queries
    def phase_at(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        starts = np.array([p.start for p in self.phases])
        idx = np.searchsorted(starts, frames, side="right") - 1
        return np.clip(idx, 0, len(self.phases) - 1)

    def phase_names_at(self, frames: np.ndarray) -> np.ndarray:
        idx = self.phase_at(frames)
        names = np.array([p.name for p in self.phases])
        return names[idx]

    def openness(self, frames: np.ndarray) -> np.ndarray:
        """o(t): INTRO/REOPEN 0->1 (Q5), OPEN 1, CONTRACT 1->0 (Q5), GOAL_HOLD 0."""
        frames = np.asarray(frames, dtype=np.float64)
        idx = self.phase_at(frames.astype(np.int64))
        o = np.zeros(frames.shape, dtype=np.float64)
        for k, p in enumerate(self.phases):
            m = idx == k
            if not m.any():
                continue
            if p.frames <= 0:
                continue
            s = np.clip((frames[m] - p.start) / float(p.frames), 0.0, 1.0)
            if p.name in ("INTRO", "REOPEN"):
                o[m] = Q(s)
            elif p.name == "OPEN":
                o[m] = 1.0
            elif p.name == "CONTRACT":
                o[m] = 1.0 - Q(s)
            else:
                o[m] = 0.0
        return o

    def goal_hold_mask(self, frames: np.ndarray) -> np.ndarray:
        return self.phase_names_at(frames) == "GOAL_HOLD"

    def to_trace(self) -> dict:
        return {
            "resolved_stage_seconds": self.resolved_seconds,
            "total_frames": int(self.total_frames),
            "total_seconds": self.total_frames / float(self.fs),
            "phase_intervals_in_integer_frames": [p.to_dict() for p in self.phases],
            "units": [{"index": u.index, "start_frame": u.start, "end_frame": u.end,
                       "goal_arrival_frame": u.goal_arrival,
                       "phases": [p.name for p in u.phases]} for u in self.units],
        }

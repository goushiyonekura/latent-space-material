"""Shared data structures used by the engine, the bank/realizer and the four modes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .curves import TrackCurve


@dataclass
class Target:
    """An ideal acoustic-composition trajectory proposed by a mode (eq. 20).
    xi_hat has shape (J, d_xi) on the unit's analysis grid; GOAL_HOLD rows must equal xi_goal."""
    id: str
    xi_hat: np.ndarray
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """A legal joint gain trajectory for one planning unit and its realized composition."""
    id: int
    plan: List[List[Tuple[int, float, str]]]   # per track: (frame, value, label) waypoints
    curves: List[TrackCurve]
    gains: np.ndarray                           # (J, M) gains at analysis-window centers
    xi: np.ndarray                              # (J, d_xi) realized composition state
    parts: Dict[str, np.ndarray]                # phi (J,10), c (J,M), R (J,M,M), silent (J,)
    e_form: float = 0.0
    e_hist: float = 0.0
    e_motion: float = 0.0
    origin: str = "bank"
    hard_ok: bool = True
    violations: List[str] = field(default_factory=list)
    parent_ids: List[Any] = field(default_factory=list)


@dataclass
class Realization:
    """A candidate evaluated against one target (the realizer's output, eq. 21/46)."""
    candidate: Candidate
    target: Target
    mode_error: float
    normalized_mode_error: float
    e_form: float
    e_hist: float
    e_motion: float
    total: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitContext:
    """Everything a mode / the realizer needs about one planning unit (spec §5.1)."""
    index: int
    start: int
    end: int
    goal_arrival: Optional[int]
    starts_from_silence: bool
    idx: np.ndarray            # (J,) indices into the analyzer's global grid
    centers: np.ndarray        # (J,) window-center frames
    seconds: np.ndarray        # (J,) window-center seconds
    o: np.ndarray              # (J,) openness
    phase_names: np.ndarray    # (J,) str
    hold_mask: np.ndarray      # (J,) bool, True inside GOAL_HOLD
    xi_goal: np.ndarray        # (J, d) goal-only composition at the same absolute times
    f_mat: np.ndarray          # (J, M, 10) normalized material features
    S: np.ndarray              # (J, M, M) material similarity
    chi: np.ndarray            # (J, M) bounded log-energy change of each material
    start_gains: np.ndarray    # (M,) gains at unit start (zero velocity)
    fs: int
    M: int
    d_xi: int
    analyzer: Any              # Analyzer
    phase_frames: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def J(self) -> int:
        return int(len(self.idx))

    @property
    def N(self) -> int:
        return self.M - 1

    @property
    def free_mask(self) -> np.ndarray:
        return ~self.hold_mask

    @property
    def mean_openness(self) -> float:
        fm = self.free_mask
        return float(self.o[fm].mean()) if fm.any() else 0.0

    def composition(self, gains: np.ndarray):
        """gains (J, M) -> (xi (J,d), parts)."""
        return self.analyzer.composition(gains, self.idx)

    def probe(self, gain_vector: np.ndarray):
        """Constant gain vector across the unit -> (xi (J,d), parts)."""
        g = np.broadcast_to(np.asarray(gain_vector, dtype=np.float64), (self.J, self.M)).copy()
        return self.analyzer.composition(g, self.idx)

    def dist2(self, xi_a: np.ndarray, xi_b: np.ndarray) -> np.ndarray:
        return self.analyzer.dist2(xi_a, xi_b)

    def mean_dist2(self, xi_a: np.ndarray, xi_b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        d = self.analyzer.dist2(xi_a, xi_b)
        if mask is None:
            mask = self.free_mask
        return float(d[mask].mean()) if mask.any() else 0.0

    def split(self, xi: np.ndarray):
        return self.analyzer.split(xi)

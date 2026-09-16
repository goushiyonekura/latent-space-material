"""Mode-controller interface shared by the four modes (spec §5.2, §7-§10).

A mode never touches gains directly.  It (1) conditions on the current material features F,
the history H and the openness o, (2) proposes ideal composition trajectories (Targets, eq. 20),
(3) scores realized compositions (E_m, default = mean d_xi^2 over non-hold windows), (4) runs
its own required internal update from the realized compositions, and (5) writes its summary
back to the history at the end of a unit (eq. 27).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import Candidate, Realization, Target, UnitContext


def ar1_noise(rng: np.random.Generator, J: int, d: int, rho: float) -> np.ndarray:
    """Time-correlated unit-variance Gaussian noise (J, d) with AR(1) coefficient rho."""
    z = rng.standard_normal((J, d))
    out = np.empty_like(z)
    s = np.sqrt(max(1e-12, 1.0 - rho * rho))
    out[0] = z[0]
    for j in range(1, J):
        out[j] = rho * out[j - 1] + s * z[j]
    return out


def fix_hold_rows(unit: UnitContext, xi_hat: np.ndarray) -> np.ndarray:
    """GOAL_HOLD rows of an ideal trajectory are pinned to the goal composition (eq. 12/13)."""
    out = np.array(xi_hat, dtype=np.float64, copy=True)
    out[unit.hold_mask] = unit.xi_goal[unit.hold_mask]
    return out


class ModeController:
    cli_name: str = ""
    internal_id: str = ""

    def __init__(self, cfg: dict, analyzer, fs: int, rng: np.random.Generator, objective):
        self.cfg = cfg
        self.analyzer = analyzer
        self.fs = int(fs)
        self.rng = rng
        self.objective = objective
        self.M = int(analyzer.M)
        self.N = self.M - 1
        self.d_xi = int(analyzer.d_xi)
        self.d_phi = int(analyzer.d_phi)
        self.warnings: List[str] = []
        self.unit_traces: List[Dict[str, Any]] = []

    # -------------------------------------------------------------- required hooks
    def begin_unit(self, unit: UnitContext, history) -> None:
        """Condition on F_k, H_k, o (eq. 20 / 27).  Must read the history."""
        raise NotImplementedError

    def hints(self, unit: UnitContext, history) -> List[tuple]:
        """Optional joint-structure hints for the bank: (track_i, track_j, 'sync'|'counter')."""
        return []

    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        """Ideal composition trajectories for this search round."""
        raise NotImplementedError

    def mode_error(self, unit: UnitContext, cand: Candidate, target: Target) -> float:
        """E_m for a realized candidate against a target (default: <d_xi^2> on free windows)."""
        return unit.mean_dist2(cand.xi, target.xi_hat)

    def update(self, unit: UnitContext, history, realizations: Sequence[Realization],
               round_index: int) -> Dict[str, Any]:
        """The mode's own required internal update from realized compositions."""
        return {}

    def end_unit(self, unit: UnitContext, history, chosen: Realization,
                 alternatives: Sequence[Realization]) -> None:
        """Write the mode-specific history summary (parents, latent stats, memory ...)."""
        return None

    def signature(self) -> np.ndarray:
        """Vector summarising the current ideal distribution / internal parameters (check B)."""
        raise NotImplementedError

    def trace(self) -> Dict[str, Any]:
        return {"warnings": list(self.warnings), "units": self.unit_traces}

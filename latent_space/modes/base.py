"""Mode-controller interface shared by the four modes (spec §5.2, §7-§10; audit-2 §C4).

A mode never touches gains directly.  It defines the *law* of the acoustic composition (targets,
relations, possibilities); the realization layer improves the legal gain trajectory against a
reference that is FROZEN while it is being realized.  Per planning unit and per commit step:

  begin_unit(unit, history)                 condition on F, H, o at the unit start
  propose(unit, history, 0, n)              full-unit ideal trajectories (warm-start reference)
  prepare_reference(unit, history, rows, xi_current, n)
                                            ideal trajectories for the next window (rows), from the
                                            committed history and the realized current composition
  window_error(unit, xi_rows, rows, target) E_m of a realized composition on the window rows
  relation_terms(unit)                      optional (omega, s, dstar, lag) for the soft relation term
  observe_committed(unit, history, rows, xi_rows, parts_rows, reference, stats)
                                            the mode's own learning from the *committed* composition
                                            (GAN D/G steps, diffusion direction, VAE latent path,
                                            transformer memory); never called during realization
  end_unit(unit, history, chosen, alternatives)   persist the mode summary into history.mode_state

The old `update(unit, history, realizations, round)` hook (target adaptation toward realized
candidates during a search) is no longer called by the engine: a reference must not move while
the trajectory is being scored against it (audit B5/C4).
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


def ar1_rho_for(step_seconds: float, tau_seconds: float) -> float:
    """AR(1) coefficient for a correlation time tau at a given step (rho = exp(-dt/tau))."""
    return float(np.exp(-float(step_seconds) / max(1e-9, float(tau_seconds))))


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
        self.step_traces: List[Dict[str, Any]] = []
        self.extra_candidate_evaluations = 0

    # -------------------------------------------------------------- unit level
    def begin_unit(self, unit: UnitContext, history) -> None:
        """Condition on F_k, H_k, o (eq. 20 / 27).  Must read the history."""
        raise NotImplementedError

    def hints(self, unit: UnitContext, history) -> List[tuple]:
        """Optional joint-structure hints for the bank: (track_i, track_j, 'sync'|'counter')."""
        return []

    def propose(self, unit: UnitContext, history, round_index: int, n_targets: int) -> List[Target]:
        """Full-unit ideal composition trajectories (used to pick the legal warm start)."""
        raise NotImplementedError

    def mode_error(self, unit: UnitContext, cand: Candidate, target: Target) -> float:
        """E_m for a realized candidate against a full-unit target (default: <d_xi^2> on free windows)."""
        return unit.mean_dist2(cand.xi, target.xi_hat)

    # -------------------------------------------------------------- step level (audit-2)
    def prepare_reference(self, unit: UnitContext, history, rows: np.ndarray, xi_current: np.ndarray,
                          n_proposals: int) -> List[Target]:
        """Ideal trajectories for the next window.  Default: full-unit proposals (the engine
        slices `rows`).  Modes override to start from the realized current composition."""
        return self.propose(unit, history, 0, n_proposals)

    def window_error(self, unit: UnitContext, xi_rows: np.ndarray, rows: np.ndarray, target: Target) -> float:
        """E_m of a realized composition on the window rows (default: mean d_xi^2 on free rows)."""
        m = unit.free_mask[rows]
        if not m.any():
            return 0.0
        d = unit.analyzer.dist2(xi_rows, target.xi_hat[rows])
        return float(d[m].mean())

    def relation_terms(self, unit: UnitContext) -> Optional[Dict[str, Any]]:
        """Optional soft relation term parameters for the realization objective (audit C3):
        {"omega": (M,M) weights, "s": (M,M) signs, "dstar": (M,M) targets, "lag_seconds": float}."""
        return None

    def observe_committed(self, unit: UnitContext, history, rows: np.ndarray, xi_rows: np.ndarray,
                          parts_rows: Dict[str, np.ndarray], reference: Optional[Target],
                          stats: Dict[str, Any]) -> Dict[str, Any]:
        """Mode-specific learning from the committed (realized, adopted) composition."""
        return {}

    def update(self, unit: UnitContext, history, realizations: Sequence[Realization],
               round_index: int) -> Dict[str, Any]:
        """Deprecated: not called by the engine (references are frozen during realization)."""
        return {}

    def end_unit(self, unit: UnitContext, history, chosen: Realization,
                 alternatives: Sequence[Realization]) -> None:
        """Write the mode-specific history summary (parents, latent stats, memory ...)."""
        return None

    def signature(self) -> np.ndarray:
        """Vector summarising the current ideal distribution / internal parameters (check B)."""
        raise NotImplementedError

    def trace(self) -> Dict[str, Any]:
        return {"warnings": list(self.warnings), "units": self.unit_traces, "steps": self.step_traces}

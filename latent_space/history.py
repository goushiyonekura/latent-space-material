"""History H (spec §6): bounded event list, composition exposure (h_c), covariation (M_H),
mode-specific summaries, comparison archive for E_hist and lineage for the GAN mode.
Updated only from *realized* (selected + legal) compositions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class History:
    def __init__(self, M: int, d_xi: int, cfg: dict):
        h = cfg["history"]
        self.M = int(M)
        self.d_xi = int(d_xi)
        self.rho = float(h["update_rate"])
        self.tau_H = float(h.get("time_constant_seconds", 60.0))     # audit C7: memory time constant (s)
        self.cov_reg = float(h["cov_regularization"])
        self.dc_cov = np.zeros((self.M, self.M))                        # EMA of composition-change covariance
        self.commits = 0
        self.committed_seconds = 0.0
        self.recent_xi: List[np.ndarray] = []                          # short list of committed rows (bounded)
        self.event_capacity = int(h["recent_event_capacity"])
        self.parent_capacity = int(h["parent_capacity"])
        self.h_c = np.zeros(self.M)
        self.M_H = np.zeros((self.M, self.M))
        self.n_updates = 0
        self.xi_mean = np.zeros(self.d_xi)
        self.events: List[Dict[str, Any]] = []
        self.comparison_archive: List[Dict[str, Any]] = []
        self.change_direction = np.zeros(self.M)
        self.change_directions: List[np.ndarray] = []
        self.transitions: List[Dict[str, Any]] = []
        self.motion_summaries: List[Dict[str, Any]] = []
        self.hold_seconds_total = 0.0
        self.units_completed = 0
        self.mode_state: Dict[str, Any] = {}
        self.parents: List[Dict[str, Any]] = []
        self.selected_parent_ids: List[Any] = []
        self.update_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ queries
    def corr(self) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(self.M_H), 0.0, None) + self.cov_reg)
        return self.M_H / (d[:, None] * d[None, :])

    def has_history(self) -> bool:
        return self.n_updates > 0

    def recent_events(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        ev = self.events if n is None else self.events[-n:]
        return list(ev)

    # ------------------------------------------------------------------ commit-level update (audit C6/C7)
    def rho_dt(self, dt_seconds: float) -> float:
        """EMA rate for a committed block of dt seconds: rho = 1 - exp(-dt / tau_H)."""
        return float(1.0 - np.exp(-max(0.0, float(dt_seconds)) / max(1e-9, self.tau_H)))

    def observe_committed(self, unit, rows: np.ndarray, xi_rows: np.ndarray, parts_rows: Dict[str, np.ndarray],
                          dt_seconds: float, new_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enter the *committed* composition of one block into the long-term summaries
        (h_c, M_H, xi_mean, change covariance) with the time-constant rate, append the completed
        events (time-ordered, bounded) and keep a short list of recent committed rows.
        GOAL_HOLD rows are excluded from the motion summaries (spec §6.2)."""
        fm = unit.free_mask[rows]
        rho = self.rho_dt(dt_seconds)
        log: Dict[str, Any] = {"unit": unit.index, "rows": int(len(rows)), "free_rows": int(fm.sum()),
                               "dt_seconds": float(dt_seconds), "rho": rho}
        if fm.any():
            c = parts_rows["c"][fm]
            xi = xi_rows[fm]
            c_bar = c.mean(axis=0)
            dc = c - c_bar
            Mc = dc.T @ dc / float(len(c))
            self.h_c = (1.0 - rho) * self.h_c + rho * c_bar
            self.M_H = (1.0 - rho) * self.M_H + rho * Mc
            self.xi_mean = (1.0 - rho) * self.xi_mean + rho * xi.mean(axis=0)
            if len(c) > 1:
                d_c = np.diff(c, axis=0)
                cov = d_c.T @ d_c / float(len(d_c))
                self.dc_cov = (1.0 - rho) * self.dc_cov + rho * cov
                w, V = np.linalg.eigh(self.dc_cov)
                v = V[:, -1]
                if v @ (c[-1] - c[0]) < 0:
                    v = -v
                self.change_direction = v
                self.change_directions.append(v.copy())
                self.change_directions = self.change_directions[-self.event_capacity:]
            self.transitions.append({"unit": unit.index, "c_first": c[0].tolist(), "c_last": c[-1].tolist(),
                                     "c_mean": c_bar.tolist()})
            self.transitions = self.transitions[-self.event_capacity:]
            self.recent_xi.append(xi.copy())
            self.recent_xi = self.recent_xi[-self.event_capacity:]
            self.n_updates += 1
            log["c_bar"] = c_bar.tolist()
        key = lambda e: (int(e["end_frame"]), int(e["start_frame"]), int(e["track"]))  # noqa: E731
        self.events = sorted(self.events + list(new_events), key=key)[-self.event_capacity:]
        self.commits += 1
        self.committed_seconds += float(dt_seconds)
        self.update_log.append(log)
        return log

    def update_unit_archive(self, unit, chosen_candidate, comparison_blocks: Dict[str, np.ndarray], fs: int) -> None:
        """Unit-level archive (comparison grid for E_hist, motion summary, hold time)."""
        cand = chosen_candidate
        peaks = [int(np.argmax(cand.gains[:, i])) for i in range(self.M)]
        self.motion_summaries.append({
            "unit": unit.index,
            "moves_per_track": [len(cv.moves()) for cv in cand.curves],
            "bumps_per_track": [len(cv.bumps) for cv in cand.curves],
            "mean_amplitude_per_track": [float(np.mean([s.delta for s in cv.moves()])) if cv.moves() else 0.0
                                         for cv in cand.curves],
            "peak_order": [int(t) for t in np.argsort(peaks)],
        })
        self.motion_summaries = self.motion_summaries[-self.event_capacity:]
        self.hold_seconds_total += float(unit.hold_mask.sum()) * unit.analyzer.hop / fs
        self.comparison_archive.append({"unit": unit.index, "blocks": comparison_blocks})
        self.comparison_archive = self.comparison_archive[-self.event_capacity:]
        self.units_completed += 1

    # ------------------------------------------------------------------ legacy unit-level update
    def update(self, unit, chosen_candidate, comparison_blocks: Dict[str, np.ndarray],
               fs: int) -> Dict[str, Any]:
        """Eq. (25)-(26) from the non-hold windows of the realized composition, plus events."""
        cand = chosen_candidate
        fm = unit.free_mask
        c = cand.parts["c"][fm]
        xi = cand.xi[fm]
        log: Dict[str, Any] = {"unit": unit.index, "free_windows": int(fm.sum())}
        if len(c) > 0:
            c_bar = c.mean(axis=0)
            dc = c - c_bar
            Mc = dc.T @ dc / float(len(c))
            self.h_c = (1.0 - self.rho) * self.h_c + self.rho * c_bar
            self.M_H = (1.0 - self.rho) * self.M_H + self.rho * Mc
            self.xi_mean = (1.0 - self.rho) * self.xi_mean + self.rho * xi.mean(axis=0)
            # principal direction of the realized composition change (for diffusion M_D etc.)
            d_c = np.diff(c, axis=0)
            if len(d_c) > 1:
                cov = d_c.T @ d_c / float(len(d_c))
                w, V = np.linalg.eigh(cov)
                v = V[:, -1]
                if v @ (c[-1] - c[0]) < 0:
                    v = -v
                self.change_direction = v
                self.change_directions.append(v.copy())
                self.change_directions = self.change_directions[-self.event_capacity:]
            self.transitions.append({"unit": unit.index, "c_first": c[0].tolist(), "c_last": c[-1].tolist(),
                                     "c_mean": c_bar.tolist()})
            self.transitions = self.transitions[-self.event_capacity:]
            log["c_bar"] = c_bar.tolist()
            self.n_updates += 1
        # events: one per realized move segment, ordered by event time (end_frame, start_frame,
        # track) before the capacity cut so that "recent" means recent in time (audit §11.1)
        centers = unit.centers
        new_events = []
        for i, curve in enumerate(cand.curves):
            for seg in curve.moves():
                j0 = int(np.clip(np.searchsorted(centers, seg.start), 0, unit.J - 1))
                j1 = int(np.clip(np.searchsorted(centers, seg.end) - 1, 0, unit.J - 1))
                xi0 = cand.xi[j0]
                xi1 = cand.xi[j1]
                new_events.append({
                    "unit": unit.index, "track": i, "start_frame": int(seg.start), "end_frame": int(seg.end),
                    "start_seconds": seg.start / fs, "end_seconds": seg.end / fs,
                    "direction": int(seg.direction), "start_gain": float(seg.a), "end_gain": float(seg.b),
                    "phase": str(unit.phase_names[j1]), "xi_start": xi0.copy(), "xi_end": xi1.copy(),
                    "dxi": (xi1 - xi0).copy(), "c_end": cand.parts["c"][j1].copy(),
                })
        key = lambda e: (int(e["end_frame"]), int(e["start_frame"]), int(e["track"]))  # noqa: E731
        self.events = sorted(self.events + new_events, key=key)[-self.event_capacity:]
        # motion summary (order of peaks, amplitudes)
        peaks = [int(np.argmax(cand.gains[:, i])) for i in range(self.M)]
        self.motion_summaries.append({
            "unit": unit.index,
            "moves_per_track": [len(cv.moves()) for cv in cand.curves],
            "mean_amplitude_per_track": [float(np.mean([s.delta for s in cv.moves()])) if cv.moves() else 0.0
                                         for cv in cand.curves],
            "peak_order": [int(t) for t in np.argsort(peaks)],
        })
        self.motion_summaries = self.motion_summaries[-self.event_capacity:]
        self.hold_seconds_total += float(unit.hold_mask.sum()) * unit.analyzer.hop / fs
        self.comparison_archive.append({"unit": unit.index, "blocks": comparison_blocks})
        self.comparison_archive = self.comparison_archive[-self.event_capacity:]
        self.units_completed += 1
        self.update_log.append(log)
        return log

    def add_parent(self, parent: Dict[str, Any]) -> None:
        self.parents.append(parent)
        self.parents = self.parents[-self.parent_capacity:]

    # ------------------------------------------------------------------ trace
    def to_trace(self) -> dict:
        def _ev(e):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()
                    if k not in ("xi_start", "xi_end")}
        return {
            "n_updates": self.n_updates, "units_completed": self.units_completed,
            "commits": self.commits, "committed_seconds": self.committed_seconds, "tau_H_seconds": self.tau_H,
            "dc_cov": self.dc_cov.tolist(),
            "h_c": self.h_c.tolist(), "M_H": self.M_H.tolist(), "corr_M_H": self.corr().tolist(),
            "xi_mean": self.xi_mean.tolist(), "change_direction": self.change_direction.tolist(),
            "events": [_ev(e) for e in self.events], "transitions": self.transitions,
            "motion_summaries": self.motion_summaries, "hold_seconds_total": self.hold_seconds_total,
            "comparison_archive_units": [a["unit"] for a in self.comparison_archive],
            "parents": [{k: v for k, v in p.items() if k not in ("blocks", "xi")} for p in self.parents],
            "selected_parent_ids": self.selected_parent_ids,
            "update_log": self.update_log,
        }

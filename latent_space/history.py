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
        self.cov_reg = float(h["cov_regularization"])
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

    # ------------------------------------------------------------------ update
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
            "h_c": self.h_c.tolist(), "M_H": self.M_H.tolist(), "corr_M_H": self.corr().tolist(),
            "xi_mean": self.xi_mean.tolist(), "change_direction": self.change_direction.tolist(),
            "events": [_ev(e) for e in self.events], "transitions": self.transitions,
            "motion_summaries": self.motion_summaries, "hold_seconds_total": self.hold_seconds_total,
            "comparison_archive_units": [a["unit"] for a in self.comparison_archive],
            "parents": [{k: v for k, v in p.items() if k not in ("blocks", "xi")} for p in self.parents],
            "selected_parent_ids": self.selected_parent_ids,
            "update_log": self.update_log,
        }

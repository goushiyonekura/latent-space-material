"""Job engine (spec §12.4; audit-2 §C4-C7): WAV -> legal joint gain trajectories -> WAV.

Per planning unit (bounded by goal states):
  1. legal warm start: the bank builds a few legal full-unit plans; the mode proposes full-unit
     ideal trajectories; one proposal is frozen as the unit reference R0 (chosen by the best
     candidate's joint score *before* freezing) and the best bank candidate against R0 is the
     warm-start tail (bounded mutations against the same frozen R0 are allowed).
  2. commit steps (commit_seconds) inside the unit: for the next lookahead window the mode prepares
     a reference from the committed history and the realized current composition; the reference is
     FROZEN (hashed); the tail is improved on that window by local bump corrections (zero value /
     velocity / acceleration at both ends) chosen by a finite-difference coordinate search on the
     joint objective computed from the actual mixed composition; only improvements are accepted;
     the first commit_seconds are committed, their realized composition enters the history with the
     time-constant rate, completed motion events are appended, and the mode observes the committed
     block.  Reference values used for committed rows are stored for the full-grid residual.
  3. the final composite trajectory is checked (analytic Q5 checks + sampled composite checks on the
     actual motion episodes), rendered by eq. (1) only, and re-analysed on a spread of windows.
"""
from __future__ import annotations

import copy
import hashlib
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import Analyzer
from .audio_io import read_wav
from .bank import Bank
from .config import MODE_IDS, SPEC_VERSION
from .curves import Bump, MotionLimits, TrackCurve
from .form import FormInfeasible, FormPlan
from .history import History
from .modes import IncompleteImplementation, make_mode
from .objective import Objective
from .render import goal_hold_reference, render_equation_1
from .trace import write_gain_csv, write_result_wav, write_trace
from .types import Candidate, Realization, Target, UnitContext

IMPLEMENTATION_STYLE = ("H5 structure-corresponding controllers over acoustic-composition space; "
                        "frozen-reference realization in short commit steps; numpy only; "
                        "gain is the sole actuator (eq. 1)")


class HardConstraintFailure(Exception):
    pass


class InputError(Exception):
    pass


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


class Job:
    def __init__(self, cfg: dict, output_dir: str, config_path: Optional[str] = None,
                 mode_override: Optional[str] = None):
        self.cfg = cfg
        self.output_dir = output_dir
        self.config_path = config_path
        self.mode_override = mode_override
        self.mode_name = cfg["mode"]
        self.warnings: List[str] = []
        self.implementation_gaps: List[str] = []
        self.t_start = time.time()
        self.rng = np.random.default_rng(int(cfg["seed"]))
        self.unit_reports: List[Dict[str, Any]] = []
        self.unit_chosen: List[tuple] = []
        self.unit_refs: Dict[int, np.ndarray] = {}      # frozen reference rows actually used per unit

    # ------------------------------------------------------------------ inputs
    def _resolve(self, p: str) -> str:
        base = os.path.dirname(os.path.abspath(self.config_path)) if self.config_path else os.getcwd()
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    def load_inputs(self) -> None:
        cfg = self.cfg
        paths = [cfg["goal"]] + list(cfg["materials"])
        self.input_paths = [self._resolve(p) for p in paths]
        self.sources: List[np.ndarray] = []
        self.infos = []
        for p in self.input_paths:
            if not os.path.exists(p):
                raise InputError(f"input not found: {p}")
            data, info = read_wav(p)
            if data.shape[0] == 0:
                raise InputError(f"empty audio: {p}")
            self.sources.append(data)
            self.infos.append(info)
        fs_set = {i.sample_rate for i in self.infos}
        ch_set = {i.channels for i in self.infos}
        if len(fs_set) != 1:
            raise InputError(f"sample rates differ across inputs: {sorted(fs_set)} (no auto-resampling)")
        if len(ch_set) != 1:
            raise InputError(f"channel counts differ across inputs: {sorted(ch_set)} (no auto-conversion)")
        self.fs = int(self.infos[0].sample_rate)
        self.C = int(self.infos[0].channels)
        if self.C not in (1, 2):
            raise InputError(f"channels must be 1 or 2, got {self.C}")
        self.M = len(self.sources)
        self.N = self.M - 1
        if self.N < 2:
            raise InputError("need at least 2 materials (N >= 2)")
        goal = self.sources[0]
        if float(np.max(np.abs(goal))) <= 1e-7:
            raise InputError("goal audio is entirely silent; not usable as the goal")
        for k, (x, info) in enumerate(zip(self.sources, self.infos)):
            secs = x.shape[0] / self.fs
            if k == 0 and not (10.0 <= secs <= 30.0):
                self.warnings.append(f"goal length {secs:.1f}s outside the expected 10-30s range")
            if k > 0 and not (180.0 <= secs <= 600.0):
                self.warnings.append(f"material {k} length {secs:.1f}s outside the expected 180-600s range")
            jump = float(np.max(np.abs(x[-1].astype(np.float64) - x[0].astype(np.float64))))
            if jump > 0.1:
                self.warnings.append(f"track {k}: loop boundary jump {jump:.3f} (not repaired, by design)")
        bl = cfg["base_level"]
        peaks = np.array([float(np.max(np.abs(x))) for x in self.sources])
        if bl.get("policy", "shared_static_peak_bound") != "shared_static_peak_bound":
            self.warnings.append(f"unknown base_level.policy {bl.get('policy')}; using shared_static_peak_bound")
        B = min(1.0, float(bl["sample_peak_cap"]) / (float(peaks.sum()) + float(bl.get("epsilon", 1e-9))))
        self.peaks = peaks
        self.base_gains = [B] * self.M
        self.track_names = ["goal"] + [os.path.splitext(os.path.basename(p))[0] for p in self.input_paths[1:]]
        self.spec_document_sha256 = None
        sd = cfg.get("spec_document")
        if sd:
            sp = self._resolve(str(sd))
            if os.path.exists(sp):
                h = hashlib.sha256()
                with open(sp, "rb") as f:
                    h.update(f.read())
                self.spec_document_sha256 = h.hexdigest()
            else:
                self.warnings.append(f"spec_document not found: {sp}")

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        cfg = self.cfg
        self.lim = MotionLimits.from_config(cfg["motion"], cfg["numerics"])
        self.form = FormPlan.build(cfg["form"], self.fs, self.lim)
        self.total_frames = self.form.total_frames
        t0 = time.time()
        self.analyzer = Analyzer(self.sources, self.base_gains, self.fs, self.total_frames, cfg)
        self.analysis_seconds = time.time() - t0
        self.objective = Objective(cfg, self.lim, self.fs)
        self.history = History(self.M, self.analyzer.d_xi, cfg)
        self.mode = make_mode(self.mode_name, cfg, self.analyzer, self.fs, self.rng, self.objective)
        w_ms = self.analyzer.W / self.fs * 1000.0
        loops = {self.track_names[k]: self.total_frames / float(x.shape[0]) for k, x in enumerate(self.sources)}
        self.source_loop_counts = loops
        self.implementation_gaps += [
            "finite realization: reported scores are best-so-far over the finite set of legal trajectories examined",
            f"analysis holds each gain at its window-centre value inside a {w_ms:.1f} ms window; proxy-vs-rendered "
            "composition distance is measured on the same windows and recorded (not a hard constraint)",
            "SCORED_HOLD segments are accepted by the checker when motion.nonzero_scored_holds=true but never generated",
            "motion limits and perceptual behaviour are uncalibrated on real materials (calibration_status=UNVERIFIED)",
            "mode fits are structure-corresponding controllers with bounded iterations; no trained DDPM/VAE/LLM/GAN and "
            "no claim of convergence or distribution matching",
            "the acoustic representation (energy, 8 bands, flux, contribution proxies, pair relations) is a reduction; "
            "different sounds can share a proxy score",
            "local realization uses additive bump corrections on a legal base trajectory (one curve family, audit C6.1); "
            "the base reversal times are fixed at the warm start",
        ]

    def unit_context(self, u) -> UnitContext:
        an = self.analyzer
        idx = an.grid_indices(u.start, u.end)
        centers = an.centers[idx]
        phase_names = self.form.phase_names_at(centers)
        e0 = np.zeros(self.M)
        e0[0] = 1.0
        return UnitContext(
            index=u.index, start=u.start, end=u.end, goal_arrival=u.goal_arrival,
            starts_from_silence=u.starts_from_silence, idx=idx, centers=centers,
            seconds=centers / float(self.fs), o=self.form.openness(centers), phase_names=phase_names,
            hold_mask=(phase_names == "GOAL_HOLD"), xi_goal=an.xi_goal_all[idx], f_mat=an.f_mat[idx],
            S=an.S[idx], chi=an.chi[idx], start_gains=(np.zeros(self.M) if u.starts_from_silence else e0),
            fs=self.fs, M=self.M, d_xi=an.d_xi, analyzer=an,
            phase_frames={p.name: (p.start, p.end) for p in u.phases})

    # ------------------------------------------------------------------ scoring helpers
    def _score(self, unit: UnitContext, cand: Candidate, target: Target, mode=None) -> Realization:
        mode = mode or self.mode
        em = float(mode.mode_error(unit, cand, target))
        fit = float(unit.mean_dist2(cand.xi, target.xi_hat))
        nem = self.objective.normalize_mode_error(self.mode_name, em)
        total = self.objective.total(nem, cand.e_form, cand.e_hist, cand.e_motion)
        r = Realization(cand, target, em, nem, cand.e_form, cand.e_hist, cand.e_motion, total)
        r.extra["target_fit_error"] = fit
        r.extra["mode_penalty"] = em - fit
        r.extra["normalized_target_fit_error"] = self.objective.normalize_mode_error(self.mode_name, fit)
        return r

    def _tolerance_quantity(self, r: Realization) -> float:
        if self.cfg["search"]["tolerance_applies_to"] == "target_fit":
            return float(r.extra["normalized_target_fit_error"])
        return float(r.normalized_mode_error)

    def _cap_intervals(self, track: int, unit: UnitContext) -> List[Tuple[int, int, float]]:
        ge = self.cfg["form"]["goal_exposure"]
        if track != 0 or ge["policy"] == "free":
            return []
        out = []
        for name in ("INTRO", "OPEN"):
            if name in unit.phase_frames:
                a, b = unit.phase_frames[name]
                if b > a:
                    out.append((a, b, float(ge["intro_max"] if name == "INTRO" else ge["open_max"])))
        return out

    def _window_eval(self, unit: UnitContext, curves: List[TrackCurve], rows: np.ndarray, ref: Target,
                     rel_terms, t0: int, t1: int) -> Dict[str, float]:
        """Joint objective of a composite trajectory on the window rows against the frozen ref."""
        centers = unit.centers[rows]
        gains = np.stack([cv.values(centers) for cv in curves], axis=1)
        xi, parts = self.analyzer.composition(gains, unit.idx[rows])
        em = float(self.mode.window_error(unit, xi, rows, ref))
        fm = unit.free_mask[rows]
        fit = float(self.analyzer.dist2(xi, ref.xi_hat[rows])[fm].mean()) if fm.any() else 0.0
        ef = self.objective.e_form_rows(unit, xi, parts, rows)
        jrel = self.objective.j_relation(unit, parts, rows, rel_terms)
        step = max(1, int(round(self.fs / 50.0)))
        fr = np.arange(t0, t1, step, dtype=np.int64)
        sm = 0.0
        if len(fr) > 1:
            acc = 0.0
            for cv in curves:
                _g, v, a, _j = cv.derivatives(fr, self.fs)
                acc += float(((v / self.lim.velocity_max) ** 2 + (a / self.lim.acceleration_max) ** 2).mean())
            sm = acc / len(curves)
        rz = self.cfg["realization"]
        J = (self.objective.w_mode * self.objective.normalize_mode_error(self.mode_name, em)
             + self.objective.w_form * ef + float(rz["w_relation"]) * jrel + float(rz["w_smooth"]) * sm)
        return {"J": J, "mode_error": em, "fit": fit, "penalty": em - fit, "e_form": ef, "j_rel": jrel, "smooth": sm}

    # ------------------------------------------------------------------ local realization (audit C5)
    def _refine_window(self, unit: UnitContext, curves: List[TrackCurve], rows: np.ndarray, ref: Target,
                       rel_terms, t0: int, t1: int) -> Tuple[List[TrackCurve], Dict[str, Any]]:
        rz = self.cfg["realization"]
        fs = self.fs
        L = t1 - t0
        short = max(int(round(float(rz["bump_short_fraction"]) * L)), int(2.0 * fs))
        min_len = int(2.0 * fs)
        params: List[Tuple[int, int, int]] = []
        phases_in = set(unit.phase_names[rows].tolist())
        for i in range(len(curves)):
            if i == 0 and self.cfg["form"]["goal_exposure"]["policy"] != "free" and phases_in <= {"INTRO", "OPEN"}:
                continue  # goal is capped at 0 here: no correction possible
            if L >= min_len:
                params.append((i, t0, t1))
            if short < L and short >= min_len:
                params.append((i, t0, t0 + short))
        base = list(curves)
        amps = np.zeros(len(params))
        caps = {i: self._cap_intervals(i, unit) for i in range(len(curves))}

        def build(a: np.ndarray) -> List[TrackCurve]:
            out = list(base)
            for (i, s0, s1), amp in zip(params, a):
                if abs(amp) > 1e-12:
                    out[i] = out[i].add_bump(Bump(s0, s1, float(amp)))
            return out

        def legal(cs: List[TrackCurve], changed: int) -> bool:
            return cs[changed].check_composite(fs, self.lim, start=t0, end=t1, cap_intervals=caps[changed]) == []

        cur = self._window_eval(unit, base, rows, ref, rel_terms, t0, t1)
        best_J = cur["J"]
        best_amps = amps.copy()
        best_eval = cur
        evals = 1
        accepted = 0
        rejected_illegal = 0
        step = float(rz["initial_step"])
        eps = float(rz["improvement_epsilon"])
        max_evals = int(rz["max_evaluations_per_step"])
        for _sweep in range(int(rz["max_refinement_sweeps"])):
            improved = False
            for k, (i, _s0, _s1) in enumerate(params):
                for sign in (1.0, -1.0):
                    if evals >= max_evals:
                        break
                    trial = best_amps.copy()
                    trial[k] += sign * step
                    cs = build(trial)
                    if not legal(cs, i):
                        rejected_illegal += 1
                        continue
                    ev = self._window_eval(unit, cs, rows, ref, rel_terms, t0, t1)
                    evals += 1
                    if ev["J"] < best_J - eps:
                        best_J, best_amps, best_eval = ev["J"], trial, ev
                        accepted += 1
                        improved = True
            if not improved:
                step *= 0.5
                if step < float(rz["min_step"]):
                    break
            if evals >= max_evals:
                break
        final = build(best_amps)
        stats = {"initial": cur, "final": best_eval, "evaluations": evals, "accepted": accepted,
                 "rejected_illegal": rejected_illegal, "n_params": len(params),
                 "nonzero_bumps": int(np.sum(np.abs(best_amps) > 1e-12))}
        return final, stats

    # ------------------------------------------------------------------ unit
    def run_unit(self, unit: UnitContext) -> Realization:
        cfg = self.cfg
        s = cfg["search"]
        rz = cfg["realization"]
        budget = int(s["max_total_candidates_per_cycle"])
        bank_target = int(s["candidate_bank_target"])
        n_targets = int(s.get("targets_per_round", 4))
        tol = float(s["normalized_mode_tolerance"])
        t_unit = time.time()

        pre_mode = copy.deepcopy(self.mode)
        pre_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self.mode.extra_candidate_evaluations = 0
        self.mode.begin_unit(unit, self.history)
        hints = list(self.mode.hints(unit, self.history) or [])
        bank = Bank(unit, self.lim, cfg, self.rng, self.objective, self.history)
        cands: List[Candidate] = []
        attempts = 0
        while len(cands) < min(bank_target, budget) and attempts < 4 * bank_target:
            attempts += 1
            c = bank.random_candidate(hints if attempts % 2 == 0 else None)
            if c is not None:
                cands.append(c)
        if not cands:
            raise HardConstraintFailure(
                f"unit {unit.index}: no legal joint trajectory found in {attempts} attempts "
                f"(bank rejected {bank.n_failed} illegal plans); NO_FEASIBLE_PLAN_FOUND")
        evaluated = len(cands) + int(getattr(self.mode, "extra_candidate_evaluations", 0) or 0)
        # ---- unit-level reference R0: chosen among proposals by reachability, then frozen
        props = self.mode.propose(unit, self.history, 0, n_targets)
        if not props:
            raise HardConstraintFailure(f"unit {unit.index}: mode proposed no reference")
        best_pair = None
        for tg in props:
            for c in cands:
                rr = self._score(unit, c, tg)
                if best_pair is None or rr.total < best_pair.total:
                    best_pair = rr
        R0 = best_pair.target
        R0_hash = _hash_array(R0.xi_hat)
        warm = best_pair
        J_warm_initial = warm.total
        # bounded bank-level improvement against the frozen R0 (mutations of the best candidate)
        while evaluated < budget:
            child = bank.mutate(warm.candidate, hints)
            if child is None:
                evaluated += 1  # count the attempt toward the budget to stay bounded
                continue
            evaluated += 1
            cands.append(child)
            rr = self._score(unit, child, R0)
            if rr.total < warm.total - float(s["minimum_improvement"]):
                warm = rr
        pool = sorted([self._score(unit, c, R0) for c in cands], key=lambda x: x.total)
        alternatives = [x for x in pool if x.candidate is not warm.candidate][:3]
        rel_terms = self.mode.relation_terms(unit)

        # ---- commit steps with frozen window references (audit C5/C6/C7)
        curves: List[TrackCurve] = [TrackCurve(list(cv.segments), list(cv.bumps)) for cv in warm.candidate.curves]
        centers = unit.centers
        movable_end = unit.goal_arrival if unit.goal_arrival is not None else unit.end
        commit_frames = max(1, int(round(float(rz["commit_seconds"]) * self.fs)))
        look_frames = max(commit_frames, int(round(float(rz["lookahead_seconds"]) * self.fs)))
        n_ref = int(rz["n_reference_proposals"])
        t = unit.start
        frontier_seg = unit.start
        ref_store = unit.xi_goal.copy()          # reference rows actually used (hold rows = goal)
        ref_used = np.zeros(unit.J, dtype=bool)
        xi_committed = np.zeros((unit.J, unit.d_xi))
        c_committed = np.zeros((unit.J, unit.M))
        step_logs: List[Dict[str, Any]] = []
        n_commits = 0
        n_history_updates = 0
        total_evals = 0
        while t < movable_end:
            w_end = min(t + look_frames, movable_end)
            rows = np.where((centers >= t) & (centers < w_end))[0]
            if len(rows) == 0:
                t = w_end
                continue
            prev_rows = np.where(centers < t)[0]
            if len(prev_rows) and ref_used[prev_rows[-1]]:
                xi_current = xi_committed[prev_rows[-1]]
            else:
                g0 = np.array([cv.values(np.array([max(unit.start, t - 1)]))[0] for cv in curves])
                xi_current = unit.probe(g0)[0][0]
            props_w = self.mode.prepare_reference(unit, self.history, rows, xi_current, n_ref)
            if not props_w:
                props_w = [R0]
            # reference design by reachability of the current tail, then freeze
            ev0 = [self._window_eval(unit, curves, rows, tg, rel_terms, t, w_end) for tg in props_w]
            k_ref = int(np.argmin([e["J"] for e in ev0]))
            ref = props_w[k_ref]
            ref_hash = _hash_array(ref.xi_hat[rows])
            new_curves, rst = self._refine_window(unit, curves, rows, ref, rel_terms, t, w_end)
            total_evals += rst["evaluations"] + len(props_w)
            # commit the first commit_seconds
            c_end = min(t + commit_frames, movable_end)
            rows_c = np.where((centers >= t) & (centers < c_end))[0]
            if len(rows_c) == 0:
                rows_c = rows[:1]
                c_end = int(centers[rows_c[-1]]) + 1
            gains_c = np.stack([cv.values(centers[rows_c]) for cv in new_curves], axis=1)
            # composition on the committed rows with the preceding committed row as flux context
            ctx = np.concatenate([[rows_c[0] - 1], rows_c]) if rows_c[0] > 0 else rows_c
            gains_ctx = np.stack([cv.values(centers[ctx]) for cv in new_curves], axis=1)
            xi_ctx, parts_ctx = self.analyzer.composition(gains_ctx, unit.idx[ctx])
            off = 1 if rows_c[0] > 0 else 0
            xi_c = xi_ctx[off:]
            parts_c = {k: (v[off:] if isinstance(v, np.ndarray) and v.shape[0] == len(ctx) else v) for k, v in parts_ctx.items()}
            xi_committed[rows_c] = xi_c
            c_committed[rows_c] = parts_c["c"]
            ref_store[rows_c] = ref.xi_hat[rows_c]
            ref_used[rows_c] = True
            # completed motion events (base segments ending inside the committed block)
            events = []
            for i, cv in enumerate(new_curves):
                for seg in cv.moves():
                    if frontier_seg < seg.end <= c_end:
                        j0 = int(np.clip(np.searchsorted(centers, seg.start), 0, unit.J - 1))
                        j1 = int(np.clip(np.searchsorted(centers, seg.end) - 1, 0, unit.J - 1))
                        xi0 = xi_committed[j0] if ref_used[j0] else xi_c[0]
                        xi1 = xi_committed[j1] if ref_used[j1] else xi_c[-1]
                        events.append({"unit": unit.index, "track": i, "start_frame": int(seg.start), "end_frame": int(seg.end),
                                       "start_seconds": seg.start / self.fs, "end_seconds": seg.end / self.fs,
                                       "direction": int(seg.direction), "start_gain": float(seg.a), "end_gain": float(seg.b),
                                       "phase": str(unit.phase_names[j1]), "xi_start": xi0.copy(), "xi_end": xi1.copy(),
                                       "dxi": (xi1 - xi0).copy(), "c_end": c_committed[j1].copy()})
            frontier_seg = c_end
            dt = (c_end - t) / float(self.fs)
            hlog = self.history.observe_committed(unit, rows_c, xi_c, parts_c, dt, events)
            n_history_updates += 1
            mstat = self.mode.observe_committed(unit, self.history, rows_c, xi_c, parts_c, ref,
                                                {"refinement": rst, "reference_hash": ref_hash,
                                                 "k_ref": k_ref, "n_proposals": len(props_w),
                                                 "proposal_initial_J": [e["J"] for e in ev0]})
            n_commits += 1
            step_logs.append({
                "t_seconds": t / self.fs, "window_seconds": [t / self.fs, w_end / self.fs], "rows": int(len(rows)),
                "committed_rows": int(len(rows_c)), "reference_id": ref.id, "reference_hash": ref_hash,
                "reference_source_history_version": int(self.history.n_updates),
                "reference_candidates": len(props_w), "reference_chosen_index": k_ref,
                "reference_updated_during_realization": False,
                "initial_joint_objective": rst["initial"]["J"], "final_joint_objective": rst["final"]["J"],
                "initial_fixed_target_error": rst["initial"]["fit"], "final_fixed_target_error": rst["final"]["fit"],
                "mode_term_breakdown": {"initial": rst["initial"], "final": rst["final"]},
                "evaluations": rst["evaluations"], "accepted_changes": rst["accepted"],
                "rejected_illegal": rst["rejected_illegal"], "nonzero_bumps": rst["nonzero_bumps"],
                "history_rho": hlog.get("rho"), "events_committed": len(events), "mode_observe": mstat,
            })
            curves = new_curves
            t = c_end
        # ---- final composite candidate for the unit
        gains = np.stack([cv.values(centers) for cv in curves], axis=1)
        xi, parts = unit.composition(gains)
        final = Candidate(id=10_000 + unit.index, plan=[list(p) for p in warm.candidate.plan], curves=curves, gains=gains,
                          xi=xi, parts=parts, origin="warm_start+local_refinement", parent_ids=[warm.candidate.id])
        final.e_form = self.objective.e_form(unit, xi, parts)
        final.e_hist = self.objective.e_hist(unit, xi, self.history)
        final.e_motion = self.objective.e_motion(curves, unit)
        chosen = self._score(unit, final, R0)
        # residual on the full grid against the frozen references actually used
        fm = unit.free_mask
        full_resid = float(self.analyzer.dist2(xi, ref_store)[fm & ref_used].mean()) if (fm & ref_used).any() else 0.0
        self.unit_refs[unit.index] = ref_store
        tol_chosen = bool(self._tolerance_quantity(chosen) <= tol)
        xs = [c.xi[fm] for c in cands[:32]]
        div = 0.0
        if len(xs) > 1:
            ds = [float(unit.analyzer.dist2(xs[a], xs[b]).mean()) for a in range(len(xs)) for b in range(a + 1, len(xs))]
            div = float(np.mean(ds))
        hist_eff = self._history_effect(unit, cands, pre_mode, pre_rng_state, n_targets)
        self.mode.end_unit(unit, self.history, chosen, alternatives)
        blocks = self.objective.phase_blocks(unit, xi)
        self.history.update_unit_archive(unit, final, blocks, self.fs)
        for note in bank.stats().get("notes", []):
            self.warnings.append(f"unit {unit.index}: {note}")
        J0s = [sl["initial_joint_objective"] for sl in step_logs]
        J1s = [sl["final_joint_objective"] for sl in step_logs]
        f0s = [sl["initial_fixed_target_error"] for sl in step_logs]
        f1s = [sl["final_fixed_target_error"] for sl in step_logs]
        self.unit_reports.append({
            "unit": unit.index, "frames": [int(unit.start), int(unit.end)], "goal_arrival_frame": unit.goal_arrival,
            "grid_points": int(unit.J), "bank": bank.stats(), "candidates_evaluated": evaluated, "budget": budget,
            "unit_reference": {"id": R0.id, "hash": R0_hash, "proposals": len(props),
                               "warm_start_candidate": warm.candidate.id, "warm_start_origin": warm.candidate.origin,
                               "J_first_candidate": float(J_warm_initial), "J_warm_start": float(warm.total)},
            "steps": step_logs, "commits": n_commits, "history_updates": n_history_updates,
            "step_evaluations": total_evals,
            "realization_summary": {
                "mean_initial_joint_objective": float(np.mean(J0s)) if J0s else None,
                "mean_final_joint_objective": float(np.mean(J1s)) if J1s else None,
                "mean_initial_fixed_target_error": float(np.mean(f0s)) if f0s else None,
                "mean_final_fixed_target_error": float(np.mean(f1s)) if f1s else None,
                "steps_with_accepted_changes": int(sum(1 for sl in step_logs if sl["accepted_changes"] > 0)),
                "total_accepted_changes": int(sum(sl["accepted_changes"] for sl in step_logs)),
                "full_grid_fixed_target_residual": full_resid,
                "note": "references frozen per window (hash recorded); improvements are of the gain trajectory only",
            },
            "tolerance_met": tol_chosen, "tolerance_applies_to": cfg["search"]["tolerance_applies_to"], "tolerance": tol,
            "chosen": {"candidate_id": final.id, "origin": final.origin, "target_id": R0.id, "total": chosen.total,
                       "mode_error": chosen.mode_error, "normalized_mode_error": chosen.normalized_mode_error,
                       "target_fit_error": chosen.extra["target_fit_error"], "mode_penalty": chosen.extra["mode_penalty"],
                       "e_form": chosen.e_form, "e_hist": chosen.e_hist, "e_motion": chosen.e_motion},
            "alternatives": [{"candidate_id": x.candidate.id, "total": x.total, "target_fit_error": x.extra["target_fit_error"]}
                             for x in alternatives],
            "acoustic_diversity_mean_pair_dist2": div,
            "history_internal_effect": hist_eff["signature_diff"], "history_ideal_effect": hist_eff["ideal_diff"],
            "history_realized_effect": hist_eff["realized_diff"], "history_probe": hist_eff,
            "seconds": time.time() - t_unit,
        })
        self.unit_chosen.append((unit, chosen))
        return chosen

    def _history_effect(self, unit: UnitContext, cands: List[Candidate], pre_mode, pre_rng_state,
                        n_targets: int) -> Dict[str, Any]:
        out = {"signature_diff": 0.0, "ideal_diff": 0.0, "realized_diff": 0.0,
               "method": "same pre-unit mode state / material phase / rng state; history real vs empty; "
                         "argmin over the same bank with w_mode*nme + w_form*E_form + w_motion*E_motion"}
        if not self.history.has_history():
            out["skipped"] = "no history yet"
            return out
        try:
            def probe(hist):
                m = copy.deepcopy(pre_mode)
                m.rng = np.random.default_rng(0)
                m.rng.bit_generator.state = copy.deepcopy(pre_rng_state)
                m.extra_candidate_evaluations = 0
                m.begin_unit(unit, hist)
                sig = np.asarray(m.signature(), dtype=np.float64)
                targets = m.propose(unit, hist, 0, n_targets)
                ideal = np.mean([t.xi_hat[unit.free_mask].mean(axis=0) for t in targets], axis=0)
                best = None
                for tg in targets:
                    for c in cands:
                        em = float(m.mode_error(unit, c, tg))
                        tot = self.objective.total(self.objective.normalize_mode_error(self.mode_name, em),
                                                   c.e_form, 0.0, c.e_motion)
                        if best is None or tot < best[0]:
                            best = (tot, c)
                return sig, ideal, best[1]
            sig_a, ideal_a, best_a = probe(copy.deepcopy(self.history))
            sig_b, ideal_b, best_b = probe(History(self.M, self.analyzer.d_xi, self.cfg))
            out["signature_diff"] = float(np.linalg.norm(sig_a - sig_b)) if sig_a.shape == sig_b.shape else float("nan")
            out["ideal_diff"] = float(np.abs(ideal_a - ideal_b).mean())
            out["realized_diff"] = float(unit.mean_dist2(best_a.xi, best_b.xi))
            out["same_candidate_selected"] = bool(best_a.id == best_b.id)
        except Exception as e:  # noqa: BLE001
            self.warnings.append(f"history effect probe failed: {e}")
            out["error"] = str(e)
        return out

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        status = "BEST_EFFORT"
        error: Optional[str] = None
        y = None
        full_curves: List[TrackCurve] = []
        hard_checks: Dict[str, Any] = {}
        try:
            self.load_inputs()
            self.setup()
            unit_curves: List[List[TrackCurve]] = []
            for u in self.form.units:
                unit = self.unit_context(u)
                chosen = self.run_unit(unit)
                unit_curves.append(chosen.candidate.curves)
            full_curves = [TrackCurve([seg for uc in unit_curves for seg in uc[i].segments],
                                      [b for uc in unit_curves for b in uc[i].bumps]) for i in range(self.M)]
            hard_checks = self.hard_checks(full_curves)
            if not hard_checks["all_passed"]:
                raise HardConstraintFailure("final curve failed hard checks: " + "; ".join(hard_checks["violations"][:10]))
            y = render_equation_1(self.sources, self.base_gains, full_curves, self.total_frames,
                                  int(self.cfg["render"]["block_frames"]))
            if not np.all(np.isfinite(y)):
                raise HardConstraintFailure("rendered output contains non-finite samples")
            hard_checks.update(self.rendered_checks(y, full_curves))
            if not hard_checks["all_passed"]:
                raise HardConstraintFailure("rendered output failed goal-hold verification")
            status = "VALID_APPROXIMATION" if all(r["tolerance_met"] for r in self.unit_reports) else "BEST_EFFORT"
        except IncompleteImplementation as e:
            status, error = "INCOMPLETE_IMPLEMENTATION", str(e)
        except (FormInfeasible, HardConstraintFailure) as e:
            status, error = "HARD_CONSTRAINT_FAILURE", str(e)
        except InputError as e:
            status, error = "INPUT_ERROR", str(e)
        except Exception as e:  # noqa: BLE001
            import traceback
            status, error = "INTERNAL_ERROR", f"{e}\n{traceback.format_exc()}"
        trace = self.build_trace(status, error, full_curves, hard_checks)
        write_trace(os.path.join(self.output_dir, "state_trace.json"), trace)
        if y is not None and status in ("VALID_APPROXIMATION", "BEST_EFFORT"):
            write_result_wav(os.path.join(self.output_dir, "result.wav"), y, self.fs)
            write_gain_csv(os.path.join(self.output_dir, "gain_curves.csv"), full_curves, self.fs,
                           self.total_frames, float(self.cfg["render"]["csv_step_seconds"]), self.track_names)
        return {"status": status, "error": error, "output_dir": self.output_dir,
                "seconds": time.time() - self.t_start}

    # ------------------------------------------------------------------ checks
    def hard_checks(self, curves: List[TrackCurve]) -> Dict[str, Any]:
        viol: List[str] = []
        ge = self.cfg["form"]["goal_exposure"]
        caps0 = []
        if ge["policy"] != "free":
            for p in self.form.phases:
                if p.name in ("INTRO", "OPEN") and p.end > p.start:
                    caps0.append((p.start, p.end, float(ge["intro_max"] if p.name == "INTRO" else ge["open_max"])))
        for i, cv in enumerate(curves):
            viol += [f"track{i}:{m}" for m in cv.check(self.fs, self.lim)]          # analytic base checks
            viol += [f"track{i}:{m}" for m in cv.check_composite(self.fs, self.lim, cap_intervals=(caps0 if i == 0 else None))]
            if cv.start != 0 or cv.end != self.total_frames:
                viol.append(f"track{i}:does_not_cover_piece")
        tol = self.lim.bound_tol
        frames = np.arange(self.total_frames, dtype=np.int64)
        bounds_ok = True
        goal_ok = True
        max_g, min_g = -np.inf, np.inf
        exposure: Dict[str, Any] = {"policy": ge["policy"], "checked": ge["policy"] != "free"}
        nonzero_velocity_joints = 0
        for i, cv in enumerate(curves):
            g = cv.values(frames)
            max_g = max(max_g, float(g.max()))
            min_g = min(min_g, float(g.min()))
            if g.max() > 1.0 + tol or g.min() < -tol or not np.all(np.isfinite(g)):
                bounds_ok = False
                viol.append(f"track{i}:sampled_gain_out_of_bounds")
            for p in self.form.phases:
                if p.name == "GOAL_HOLD":
                    want = 1.0 if i == 0 else 0.0
                    seg = g[p.start:p.end]
                    if seg.size and not np.all(seg == want):
                        goal_ok = False
                        viol.append(f"track{i}:goal_hold_not_exact_in_phase@{p.start}")
                if i == 0 and ge["policy"] != "free" and p.name in ("INTRO", "OPEN") and p.end > p.start:
                    cap = float(ge["intro_max"] if p.name == "INTRO" else ge["open_max"])
                    mx = float(g[p.start:p.end].max())
                    exposure.setdefault("goal_max_per_phase", []).append({"phase": p.name, "start_frame": p.start,
                                                                          "goal_max": mx, "cap": cap})
                    if mx > cap + tol:
                        viol.append(f"track0:goal_exposure_cap_exceeded:{p.name}@{p.start}:{mx:.4f}>{cap}")
                if i == 0 and ge["policy"] != "free" and p.name == "REOPEN" and p.end > p.start and p.start > 0:
                    if g[p.start] != 1.0:
                        viol.append(f"track0:reopen_does_not_inherit_goal@{p.start}")
            # joints crossed with non-zero velocity (bump starts inside a base move)
            for b in cv.bumps:
                _gg, v, _a, _j = cv.derivatives(np.array([b.start]), self.fs)
                if abs(float(v[0])) > 1e-6:
                    nonzero_velocity_joints += 1
        return {"gain_bounds": bounds_ok, "gain_range_observed": [min_g, max_g], "continuity_and_motion_limits":
                not any(("discontinu" in v or "exceeded" in v or "below" in v or "stop" in v or "not_contiguous" in v)
                        for v in viol),
                "slow_motion_rules": not any(("slow_fraction" in v or "ramp_component" in v or "vmin" in v) for v in viol),
                "exact_goals": goal_ok, "goal_exposure": exposure, "nonzero_velocity_joints": nonzero_velocity_joints,
                "bumps_total": int(sum(len(cv.bumps) for cv in curves)),
                "violations": viol, "all_passed": len(viol) == 0}

    def rendered_checks(self, y: np.ndarray, curves: List[TrackCurve]) -> Dict[str, Any]:
        max_dev = 0.0
        ok = True
        for p in self.form.phases:
            if p.name != "GOAL_HOLD" or p.end <= p.start:
                continue
            ref = goal_hold_reference(self.sources[0], self.base_gains[0], p.start, p.end)
            dev = float(np.max(np.abs(y[p.start:p.end].astype(np.float64) - ref.astype(np.float64))))
            max_dev = max(max_dev, dev)
            if dev > 1e-6:
                ok = False
        # audit E4: windows spread over every phase of every unit plus goal boundaries; flux uses the
        # adjacent previous window on both sides; proxy / fixed reference / rendered compared per row
        checks = []
        for unit, chosen in self.unit_chosen:
            ref_store = self.unit_refs.get(unit.index)
            rows_sel: List[int] = []
            for name, (a, b) in unit.phase_frames.items():
                r = np.where((unit.centers >= a) & (unit.centers < b))[0]
                if len(r) == 0:
                    continue
                for q in (0.2, 0.5, 0.8):
                    rows_sel.append(int(r[min(len(r) - 1, int(q * len(r)))]))
            if unit.goal_arrival is not None:
                j = int(np.searchsorted(unit.centers, unit.goal_arrival))
                rows_sel += [max(1, j - 1), min(unit.J - 1, j)]
            rows_sel = sorted(set(r for r in rows_sel if r >= 1))
            d_pt, d_tt, d_tp, phi_diff, c_diff = [], [], [], [], []
            for r in rows_sel:
                idx = unit.idx[[r - 1, r]]
                xi_true, parts_true = self.analyzer.composition_from_render(y, self.sources, self.base_gains, curves, idx)
                xt = xi_true[1]
                proxy = chosen.candidate.xi[r]
                d_pt.append(float(self.analyzer.dist2(proxy[None], xt[None])[0]))
                phi_diff.append(float(np.max(np.abs(proxy[:self.analyzer.d_phi] - xt[:self.analyzer.d_phi]))))
                c_diff.append(float(np.max(np.abs(chosen.candidate.parts["c"][r] - parts_true["c"][1]))))
                if ref_store is not None and not unit.hold_mask[r]:
                    d_tt.append(float(self.analyzer.dist2(ref_store[r][None], xt[None])[0]))
                    d_tp.append(float(self.analyzer.dist2(ref_store[r][None], proxy[None])[0]))
            checks.append({"unit": unit.index, "rows": rows_sel, "seconds": [float(unit.seconds[r]) for r in rows_sel],
                           "proxy_vs_rendered_mean_dist2": float(np.mean(d_pt)) if d_pt else None,
                           "fixed_target_vs_rendered_mean_dist2": float(np.mean(d_tt)) if d_tt else None,
                           "fixed_target_vs_proxy_mean_dist2": float(np.mean(d_tp)) if d_tp else None,
                           "max_abs_phi_diff_proxy_vs_rendered": float(max(phi_diff)) if phi_diff else None,
                           "max_abs_c_diff_proxy_vs_rendered": float(max(c_diff)) if c_diff else None})
        return {"exact_goals_rendered": ok, "goal_hold_max_abs_deviation": max_dev,
                "rendered_composition_check": checks,
                "rendered_composition_note": "windows at 20/50/80% of every phase plus the goal boundary; proxy = "
                                             "window-centre gain hold Gram evaluation; fixed target = the frozen "
                                             "reference used for that row; rendered = true composition of result.wav "
                                             "(flux from the adjacent previous window)",
                "output_peak": float(np.max(np.abs(y))), "all_passed": ok}

    # ------------------------------------------------------------------ trace
    def build_trace(self, status: str, error: Optional[str], curves: List[TrackCurve],
                    hard_checks: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        tr: Dict[str, Any] = {
            "spec_version": SPEC_VERSION, "spec_document": cfg.get("spec_document"),
            "spec_document_sha256": getattr(self, "spec_document_sha256", None),
            "config_aliases_applied": cfg.get("config_aliases_applied", []),
            "mode": self.mode_name, "mode_internal_id": MODE_IDS.get(self.mode_name),
            "implementation_style": IMPLEMENTATION_STYLE, "run_status": status, "error": error,
            "calibration_status": cfg.get("calibration_status", "UNVERIFIED"),
            "perceptual_status": cfg.get("perceptual_status", "UNVERIFIED"),
            "seed": int(cfg["seed"]), "resolved_config": cfg, "config_path": self.config_path,
            "mode_override_from_cli": self.mode_override,
            "selection_policy": {"unit_reference": "frozen; chosen among proposals by best-candidate J before freezing",
                                 "window_reference": "frozen per commit window (hash recorded)",
                                 "rule": "argmin of the joint objective; only improvements accepted",
                                 "tolerance_applies_to": cfg["search"]["tolerance_applies_to"]},
            "goal_exposure_policy": cfg["form"]["goal_exposure"],
            "realization_policy": cfg["realization"],
            "motion_profile": {"profile": cfg["motion"].get("profile", "baseline"),
                               "scale_k": cfg["motion"].get("profile_scale_k", 1.0)},
            "numerical_environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                                      "machine": platform.machine()},
            "elapsed_seconds": time.time() - self.t_start,
            "warnings": list(self.warnings), "implementation_gaps": list(self.implementation_gaps),
        }
        if hasattr(self, "track_names"):
            tr["input_paths_and_hashes"] = [{"track": k, "role": "goal" if k == 0 else "material", "path": i.path,
                                             "sha256": i.sha256, "frames": i.frames, "seconds": i.frames / self.fs,
                                             "bits": i.bits, "format_tag": i.format_tag} for k, i in enumerate(self.infos)]
            tr["source_order"] = self.track_names
            tr["sample_rate"] = self.fs
            tr["channels"] = self.C
            tr["N_materials"] = self.N
            tr["base_gains"] = list(map(float, self.base_gains))
            tr["base_level"] = {"policy": "shared_static_peak_bound", "sample_peaks": self.peaks.tolist(),
                                "B": float(self.base_gains[0])}
            if hasattr(self, "source_loop_counts"):
                tr["source_loop_counts"] = self.source_loop_counts
        elif hasattr(self, "input_paths"):
            tr["input_paths"] = list(self.input_paths)
        if hasattr(self, "form"):
            tr["form"] = self.form.to_trace()
        if hasattr(self, "analyzer"):
            tr["analysis"] = self.analyzer.to_trace()
            tr["analysis"]["precompute_seconds"] = self.analysis_seconds
        if curves:
            tr["curve_segments_per_track"] = [cv.to_list() for cv in curves]
        if hasattr(self, "history"):
            tr["history"] = self.history.to_trace()
            tr["history_updates_and_selected_parent_ids"] = {
                "commit_updates": self.history.commits, "selected_parent_ids": self.history.selected_parent_ids}
        if hasattr(self, "mode"):
            tr["mode_trace"] = {self.mode_name: self.mode.trace()}
            self.warnings.extend(w for w in getattr(self.mode, "warnings", []) if w not in self.warnings)
            tr["warnings"] = list(self.warnings)
        # frozen references actually used, restorable per analysis row (audit E4)
        if self.unit_refs:
            tr["fixed_reference_rows"] = {str(k): {"grid_rows": [int(i) for i in self.unit_context_idx(k)],
                                                   "xi_hat": v.tolist()} for k, v in self.unit_refs.items()}
        tr["units"] = self.unit_reports
        tr["hard_checks"] = hard_checks
        if self.unit_reports:
            tr["soft_results"] = {
                "objective_components_per_unit": [r["chosen"] for r in self.unit_reports],
                "realization_per_unit": [r["realization_summary"] for r in self.unit_reports],
                "commits_per_unit": [r["commits"] for r in self.unit_reports],
                "history_updates_per_unit": [r["history_updates"] for r in self.unit_reports],
                "tolerance_met_per_unit": [r["tolerance_met"] for r in self.unit_reports],
                "target_fit_error_per_unit": [r["chosen"]["target_fit_error"] for r in self.unit_reports],
                "mode_penalty_per_unit": [r["chosen"]["mode_penalty"] for r in self.unit_reports],
                "acoustic_diversity_per_unit": [r["acoustic_diversity_mean_pair_dist2"] for r in self.unit_reports],
                "history_internal_effect_per_unit": [r["history_internal_effect"] for r in self.unit_reports],
                "history_ideal_effect_per_unit": [r["history_ideal_effect"] for r in self.unit_reports],
                "history_realized_effect_per_unit": [r["history_realized_effect"] for r in self.unit_reports],
                "rendered_composition_check": hard_checks.get("rendered_composition_check"),
                "best_so_far_note": "scores are best-so-far over the finite set actually examined; improvements are "
                                    "of the gain trajectory against frozen references, never of the reference",
                "warnings": list(self.warnings),
            }
        return tr

    def unit_context_idx(self, k: int) -> np.ndarray:
        for unit, _ in self.unit_chosen:
            if unit.index == k:
                return unit.idx
        return np.zeros(0, dtype=np.int64)

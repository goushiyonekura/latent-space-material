"""Job engine (spec §12.4): WAV -> legal joint gain trajectories -> WAV, one mode per job.

Audit revision (2026-09-16): final selection re-scores every examined candidate against the
final (fixed) targets and takes the argmin (or a temperature draw restricted to J <= J_min +
acceptance_margin); target-fit error, mode-specific penalties and the Gram-proxy discrepancy are
recorded separately; the history intervention probe uses identical start state, material phase,
frozen mode configuration, rng state and selection rule on both sides; the rendered PCM is
re-analysed as a full composition state on a contiguous block of the same windows.
"""
from __future__ import annotations

import copy
import hashlib
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .analysis import Analyzer
from .audio_io import read_wav
from .bank import Bank
from .config import MODE_IDS, SPEC_VERSION
from .curves import MotionLimits, TrackCurve
from .form import FormInfeasible, FormPlan
from .history import History
from .modes import IncompleteImplementation, make_mode
from .objective import Objective
from .render import goal_hold_reference, render_equation_1
from .trace import write_gain_csv, write_result_wav, write_trace
from .types import Candidate, Realization, Target, UnitContext

IMPLEMENTATION_STYLE = ("H5 structure-corresponding controllers over acoustic-composition space; "
                        "numpy only; gain is the sole actuator (eq. 1)")


class HardConstraintFailure(Exception):
    pass


class InputError(Exception):
    pass


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
        # base level: shared static peak bound (eq. 2)
        bl = cfg["base_level"]
        peaks = np.array([float(np.max(np.abs(x))) for x in self.sources])
        if bl.get("policy", "shared_static_peak_bound") != "shared_static_peak_bound":
            self.warnings.append(f"unknown base_level.policy {bl.get('policy')}; using shared_static_peak_bound")
        B = min(1.0, float(bl["sample_peak_cap"]) / (float(peaks.sum()) + float(bl.get("epsilon", 1e-9))))
        self.peaks = peaks
        self.base_gains = [B] * self.M
        self.track_names = ["goal"] + [os.path.splitext(os.path.basename(p))[0] for p in self.input_paths[1:]]
        # normative document hash (audit §7): recorded when the config names the spec file
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
        self.implementation_gaps += [
            "finite candidate search: reported scores are best-so-far over the examined legal set, not global optima",
            f"analysis holds each gain at its window-centre value inside a {w_ms:.1f} ms window; the proxy-vs-rendered "
            "composition distance is measured on the same windows and recorded (not a hard constraint)",
            "SCORED_HOLD segments are accepted by the checker when motion.nonzero_scored_holds=true but the bank never "
            "generates them",
            "motion limits and perceptual behaviour are uncalibrated on real materials (calibration_status=UNVERIFIED)",
            "mode fits are structure-corresponding controllers with bounded iterations; no trained DDPM/VAE/LLM/GAN and "
            "no claim of convergence or distribution matching",
            "the acoustic representation (energy, 8 bands, flux, contribution proxies, pair relations) is a reduction; "
            "different sounds can share a proxy score",
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

    # ------------------------------------------------------------------ scoring
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

    # ------------------------------------------------------------------ search
    def run_unit(self, unit: UnitContext) -> Realization:
        cfg = self.cfg
        s = cfg["search"]
        budget = int(s["max_total_candidates_per_cycle"])
        bank_target = int(s["candidate_bank_target"])
        n_targets = int(s.get("targets_per_round", 4))
        max_rounds = int(s["max_search_rounds"])
        tol = float(s["normalized_mode_tolerance"])
        min_impr = float(s["minimum_improvement"])
        patience = int(s["patience_rounds"])
        rule = str(s["selection"])
        margin = float(s["acceptance_margin"])
        t_unit = time.time()

        # frozen pre-unit state for the controlled history probe (same rng / same model state)
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
        all_reals: List[Realization] = []
        round_log: List[Dict[str, Any]] = []
        best_total = np.inf
        no_improve = 0
        tol_search = False
        stop_reason = "max_rounds"
        last_targets: List[Target] = []
        round_targets: List[List[Target]] = []
        for r in range(max_rounds):
            targets = self.mode.propose(unit, self.history, r, n_targets)
            if not targets:
                self.warnings.append(f"unit {unit.index}: mode proposed no targets in round {r}")
                break
            last_targets = list(targets)
            round_targets.append(last_targets)
            round_reals: List[Realization] = []
            for tg in targets:
                reals = [self._score(unit, c, tg) for c in cands]
                best = min(reals, key=lambda x: x.total)
                n_mut = max(0, min(3, budget - evaluated))
                for _ in range(n_mut):
                    child = bank.mutate(best.candidate, hints)
                    if child is None:
                        continue
                    evaluated += 1
                    cands.append(child)
                    rr = self._score(unit, child, tg)
                    if rr.total < best.total:
                        best = rr
                round_reals.append(best)
            upd = self.mode.update(unit, self.history, round_reals, r)
            all_reals.extend(round_reals)
            rb = min(x.total for x in round_reals)
            if rb < best_total - min_impr:
                best_total = rb
                no_improve = 0
            else:
                no_improve += 1
            tol_search = tol_search or any(self._tolerance_quantity(x) <= tol for x in round_reals)
            round_log.append({"round": r, "targets": len(targets), "best_total": float(rb),
                              "best_normalized_mode_error": float(min(x.normalized_mode_error for x in round_reals)),
                              "best_target_fit_error": float(min(x.extra["target_fit_error"] for x in round_reals)),
                              "evaluated_so_far": evaluated, "mode_update": upd})
            if tol_search:
                stop_reason = "tolerance_met"
                break
            if no_improve >= patience:
                stop_reason = "no_improvement"
                break
            if evaluated >= budget:
                stop_reason = "budget_exhausted"
                break
        if not last_targets:
            raise HardConstraintFailure(f"unit {unit.index}: no targets / realizations produced")
        # ---- adopted targets: the mode adapts its ideal between rounds, so one fixed target set is
        # adopted first (default: the round whose best realization scored lowest; alternatively the
        # last round), then every examined candidate is re-scored against exactly that set.
        policy = str(s.get("final_targets", "best_round"))
        if policy == "last_round" or len(round_targets) == 1:
            adopted_round = len(round_targets) - 1
        else:
            adopted_round = int(np.argmin([rl["best_total"] for rl in round_log]))
        final_targets = round_targets[adopted_round]
        pool: List[Realization] = []
        for c in cands:
            best = None
            for tg in final_targets:
                rr = self._score(unit, c, tg)
                if best is None or rr.total < best.total:
                    best = rr
            pool.append(best)
        totals = np.array([x.total for x in pool])
        k_sel, weights = self.objective.select(totals, unit.mean_openness, self.rng, rule=rule, margin=margin)
        chosen = pool[k_sel]
        J_min = float(totals.min())
        in_set = bool(chosen.total <= J_min + margin + 1e-12)
        alternatives = [x for x in pool if x is not chosen]
        tol_chosen = bool(self._tolerance_quantity(chosen) <= tol)
        # acoustic diversity of the explored candidates (eq. 24 note)
        xs = [c.xi[unit.free_mask] for c in cands[:32]]
        div = 0.0
        if len(xs) > 1:
            ds = [float(unit.analyzer.dist2(xs[a], xs[b]).mean()) for a in range(len(xs)) for b in range(a + 1, len(xs))]
            div = float(np.mean(ds))
        hist_eff = self._history_effect(unit, cands, pre_mode, pre_rng_state, n_targets)
        self.mode.end_unit(unit, self.history, chosen, alternatives)
        blocks = self.objective.phase_blocks(unit, chosen.candidate.xi)
        hist_log = self.history.update(unit, chosen.candidate, blocks, self.fs)
        for note in bank.stats().get("notes", []):
            self.warnings.append(f"unit {unit.index}: {note}")
        self.unit_reports.append({
            "unit": unit.index, "frames": [int(unit.start), int(unit.end)], "goal_arrival_frame": unit.goal_arrival,
            "grid_points": int(unit.J), "bank": bank.stats(), "candidates_evaluated": evaluated,
            "budget": budget, "rounds": round_log, "stop_reason": stop_reason,
            "tolerance_met": tol_chosen, "tolerance_reached_in_search": bool(tol_search),
            "tolerance_applies_to": cfg["search"]["tolerance_applies_to"], "tolerance": tol,
            "best_total_during_search": float(best_total),
            "best_total": J_min,
            "selection": {"rule": rule, "acceptance_margin": margin, "J_min_final_targets": J_min,
                          "chosen_total": float(chosen.total), "in_acceptance_set": in_set,
                          "final_targets_policy": policy, "adopted_target_round": adopted_round,
                          "rounds_run": len(round_targets),
                          "final_target_ids": [t.id for t in final_targets], "pool_size": len(pool),
                          "selection_weight": float(weights[k_sel]),
                          "selection_temperature": self.objective.selection_temperature(unit.mean_openness)},
            "chosen": {"candidate_id": chosen.candidate.id, "origin": chosen.candidate.origin,
                       "target_id": chosen.target.id, "total": chosen.total,
                       "mode_error": chosen.mode_error, "normalized_mode_error": chosen.normalized_mode_error,
                       "target_fit_error": chosen.extra["target_fit_error"],
                       "mode_penalty": chosen.extra["mode_penalty"],
                       "e_form": chosen.e_form, "e_hist": chosen.e_hist, "e_motion": chosen.e_motion},
            "alternatives": [{"candidate_id": x.candidate.id, "total": x.total,
                              "normalized_mode_error": x.normalized_mode_error,
                              "target_fit_error": x.extra["target_fit_error"]} for x in
                             sorted(alternatives, key=lambda z: z.total)[:6]],
            "acoustic_diversity_mean_pair_dist2": div,
            "history_internal_effect": hist_eff["signature_diff"],
            "history_ideal_effect": hist_eff["ideal_diff"],
            "history_realized_effect": hist_eff["realized_diff"],
            "history_probe": hist_eff,
            "history_update": hist_log, "seconds": time.time() - t_unit,
        })
        self.unit_chosen.append((unit, chosen))
        return chosen

    def _history_effect(self, unit: UnitContext, cands: List[Candidate], pre_mode, pre_rng_state,
                        n_targets: int) -> Dict[str, Any]:
        """Controlled intervention (audit §11.2): two computations start from the same frozen
        pre-unit mode state, the same material phase and the same rng state; only the history
        differs (real vs empty).  Both sides use the same argmin selection over the same existing
        candidate bank (no new acoustic evaluations); E_hist is excluded on both sides because it
        is itself a history term."""
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
            full_curves = [TrackCurve([seg for uc in unit_curves for seg in uc[i].segments]) for i in range(self.M)]
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
        except Exception as e:  # noqa: BLE001  (unexpected: still leave a trace, never a fake success)
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
        for i, cv in enumerate(curves):
            viol += [f"track{i}:{m}" for m in cv.check(self.fs, self.lim)]
            if cv.start != 0 or cv.end != self.total_frames:
                viol.append(f"track{i}:does_not_cover_piece")
        tol = self.lim.bound_tol
        frames = np.arange(self.total_frames, dtype=np.int64)
        bounds_ok = True
        goal_ok = True
        max_g, min_g = -np.inf, np.inf
        ge = self.cfg["form"]["goal_exposure"]
        exposure: Dict[str, Any] = {"policy": ge["policy"], "checked": ge["policy"] != "free"}
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
        return {"gain_bounds": bounds_ok, "gain_range_observed": [min_g, max_g], "continuity_and_motion_limits":
                not any(("discontinu" in v or "exceeded" in v or "below" in v or "stop" in v or "not_contiguous" in v)
                        for v in viol),
                "slow_motion_rules": not any(("slow_fraction" in v or "ramp_component" in v or "vmin" in v) for v in viol),
                "exact_goals": goal_ok, "goal_exposure": exposure, "violations": viol, "all_passed": len(viol) == 0}

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
        # full composition state of the rendered PCM on a contiguous block of the same windows:
        # proxy (window-centre gain hold) vs rendered, chosen ideal target vs rendered, target vs proxy
        checks = []
        for unit, chosen in self.unit_chosen:
            free_rows = np.where(unit.free_mask)[0]
            if len(free_rows) < 4:
                continue
            n = min(48, len(free_rows))
            mid = free_rows[len(free_rows) // 2]
            start = int(max(free_rows[0], min(mid - n // 2, free_rows[-1] - n + 1)))
            rows = np.arange(start, start + n)
            rows = rows[(rows >= 0) & (rows < unit.J)]
            idx = unit.idx[rows]
            xi_true, parts_true = self.analyzer.composition_from_render(y, self.sources, self.base_gains, curves, idx)
            keep = unit.free_mask[rows]
            keep[0] = False  # flux of the first row of a sequence is 0 by definition on both sides only if aligned
            proxy = chosen.candidate.xi[rows]
            target = chosen.target.xi_hat[rows]
            d_pt = self.analyzer.dist2(proxy, xi_true)
            d_tt = self.analyzer.dist2(target, xi_true)
            d_tp = self.analyzer.dist2(target, proxy)
            checks.append({
                "unit": unit.index, "rows": int(keep.sum()), "first_row_seconds": float(unit.seconds[rows[0]]),
                "proxy_vs_rendered_mean_dist2": float(d_pt[keep].mean()),
                "target_vs_rendered_mean_dist2": float(d_tt[keep].mean()),
                "target_vs_proxy_mean_dist2": float(d_tp[keep].mean()),
                "max_abs_phi_diff_proxy_vs_rendered": float(np.max(np.abs(proxy[keep][:, :self.analyzer.d_phi]
                                                                          - xi_true[keep][:, :self.analyzer.d_phi]))),
                "max_abs_c_diff_proxy_vs_rendered": float(np.max(np.abs(chosen.candidate.parts["c"][rows][keep]
                                                                        - parts_true["c"][keep]))),
            })
        return {"exact_goals_rendered": ok, "goal_hold_max_abs_deviation": max_dev,
                "rendered_composition_check": checks,
                "rendered_composition_note": "same analysis windows; proxy = window-centre gain hold Gram evaluation; "
                                             "target = the chosen mode target; rendered = true composition of result.wav "
                                             "with per-track components from the official curves",
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
            "selection_policy": {"rule": cfg["search"]["selection"], "acceptance_margin": cfg["search"]["acceptance_margin"],
                                 "tolerance_applies_to": cfg["search"]["tolerance_applies_to"],
                                 "final_targets": cfg["search"].get("final_targets", "best_round")},
            "goal_exposure_policy": cfg["form"]["goal_exposure"],
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
                "updates": self.history.update_log, "selected_parent_ids": self.history.selected_parent_ids}
        if hasattr(self, "mode"):
            tr["mode_trace"] = {self.mode_name: self.mode.trace()}
            self.warnings.extend(w for w in getattr(self.mode, "warnings", []) if w not in self.warnings)
            tr["warnings"] = list(self.warnings)
        tr["units"] = self.unit_reports
        tr["hard_checks"] = hard_checks
        if self.unit_reports:
            tr["soft_results"] = {
                "objective_components_per_unit": [r["chosen"] for r in self.unit_reports],
                "selection_per_unit": [r["selection"] for r in self.unit_reports],
                "best_score_per_unit": [r["best_total"] for r in self.unit_reports],
                "tolerance_met_per_unit": [r["tolerance_met"] for r in self.unit_reports],
                "tolerance_reached_in_search_per_unit": [r["tolerance_reached_in_search"] for r in self.unit_reports],
                "target_fit_error_per_unit": [r["chosen"]["target_fit_error"] for r in self.unit_reports],
                "mode_penalty_per_unit": [r["chosen"]["mode_penalty"] for r in self.unit_reports],
                "acoustic_diversity_per_unit": [r["acoustic_diversity_mean_pair_dist2"] for r in self.unit_reports],
                "history_internal_effect_per_unit": [r["history_internal_effect"] for r in self.unit_reports],
                "history_ideal_effect_per_unit": [r["history_ideal_effect"] for r in self.unit_reports],
                "history_realized_effect_per_unit": [r["history_realized_effect"] for r in self.unit_reports],
                "rendered_composition_check": hard_checks.get("rendered_composition_check"),
                "best_so_far_note": "scores are best-so-far over the finite candidate set actually examined; "
                                    "the chosen candidate is the argmin (or within acceptance_margin) against the final targets",
                "warnings": list(self.warnings),
            }
        return tr

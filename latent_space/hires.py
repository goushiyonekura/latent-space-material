"""High-resolution realization (user-authorised extension, 2026-09-16).

Authorised beyond spec v1.1: steep gain switches (short Q5 ramps exempt from the slow-motion
rules), playback-position jumps / splicing of each material within its own source, and output
dynamics.  Kept: gain per track is still the actuator (plus the position map), the goal track
keeps its continuous clock and exact goal holds, references are frozen per window, only the
gain / position choices are improved against them, and committed blocks feed the history.

Per commit step (commit_seconds):
  1. the actual material features at the *played* positions are written into the unit context
     rows of the window, then the mode prepares references; one is frozen (hash);
  2. jump candidates per material: positions of its own source whose solo band profile is closest
     to the reference mixture profile (plus one random exploration position), subject to a
     minimum clip length; a small set of combinations (no jump, one track jumps) is examined;
  3. for each combination the window Gram matrices are computed once from the actual PCM at
     those positions (exact for the summed PCM under the window-centre gain hold); the end
     levels of the materials (reached by a Q5 ramp of ramp_seconds) are improved by a
     finite-difference coordinate search on the joint objective; the best combination wins;
  4. the first commit block is committed (history, mode observation, events).
The goal track follows the exposure policy deterministically (0 in INTRO/OPEN, smooth rise over
goal_rise_seconds at the end of CONTRACT, exact 1 in GOAL_HOLD, smooth descent in REOPEN).
"""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .curves import Clip, Segment, TrackCurve
from .types import Candidate, Realization, Target, UnitContext


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


class HiresState:
    """Cross-unit state: per-track segment lists, clip maps and last jump frames."""

    def __init__(self, M: int, start: int):
        self.segments: List[List[Segment]] = [[] for _ in range(M)]
        self.clips: List[List[Clip]] = [[] for _ in range(M)]
        self.last_jump = [start] * M
        self.level = np.zeros(M)

    def curve(self, i: int) -> TrackCurve:
        return TrackCurve(list(self.segments[i]), [], list(self.clips[i]))


def run_unit_hires(job, unit: UnitContext, state: HiresState) -> Realization:
    cfg = job.cfg
    hz = cfg["hires"]
    fs = job.fs
    an = job.analyzer
    mode = job.mode
    history = job.history
    objective = job.objective
    rng = job.rng
    sources = job.sources
    M = unit.M
    t_unit = time.time()
    commit = max(1, int(round(float(hz["commit_seconds"]) * fs)))
    look = max(commit, int(round(float(hz["lookahead_seconds"]) * fs)))
    ramp = max(1, min(commit, int(round(float(hz["ramp_seconds"]) * fs))))
    rise = max(ramp, int(round(float(hz["goal_rise_seconds"]) * fs)))
    min_clip = int(round(float(hz["min_clip_seconds"]) * fs))
    n_jump = int(hz["jump_candidates"])
    level_step = float(hz["level_step"])
    max_sweeps = int(hz["max_sweeps"])
    n_ref = int(hz["n_reference_proposals"])
    w_rel = float(hz["w_relation"])
    ge = cfg["form"]["goal_exposure"]
    goal_free = ge["policy"] == "free"
    open_cap = float(ge["open_max"]) if not goal_free else 1.0
    W = an.W
    half = W // 2
    centers = unit.centers
    starts_all = centers - half

    pre_mode = copy.deepcopy(mode)
    pre_rng_state = copy.deepcopy(rng.bit_generator.state)
    mode.extra_candidate_evaluations = 0
    mode.begin_unit(unit, history)
    props0 = mode.propose(unit, history, 0, max(1, int(cfg["search"].get("targets_per_round", 4))))
    R0 = props0[0] if props0 else Target("hires:none", unit.xi_goal.copy())
    rel_terms = mode.relation_terms(unit)

    # goal track: deterministic exposure schedule for this unit (frames)
    t1 = unit.goal_arrival if unit.goal_arrival is not None else unit.end
    rise_start = max(unit.start, t1 - rise) if unit.goal_arrival is not None else None
    reopen = unit.phase_frames.get("REOPEN")
    goal_segs: List[Segment] = []
    g0 = float(state.level[0])
    cur = unit.start
    if reopen is not None and g0 >= 1.0 - 1e-12:
        desc = min(rise, reopen[1] - reopen[0])
        goal_segs.append(Segment("Q5", cur, cur + desc, 1.0, open_cap, "MOVE"))
        cur += desc
        g0 = open_cap
    if rise_start is not None:
        if rise_start > cur:
            goal_segs.append(Segment("HOLD", cur, rise_start, g0, g0, "ZERO_HOLD" if g0 <= 1e-12 else "LEVEL_HOLD"))
        goal_segs.append(Segment("Q5", rise_start, t1, g0, 1.0, "MOVE"))
        if unit.end > t1:
            goal_segs.append(Segment("HOLD", t1, unit.end, 1.0, 1.0, "GOAL_HOLD"))
    else:
        if unit.end > cur:
            goal_segs.append(Segment("HOLD", cur, unit.end, g0, g0, "ZERO_HOLD" if g0 <= 1e-12 else "LEVEL_HOLD"))
    goal_curve = TrackCurve(goal_segs)

    def gains_for(levels_end: np.ndarray, t: int, rows: np.ndarray, level_now: np.ndarray, curves_prev: List[TrackCurve]):
        """Gains at the window rows: materials ramp from level_now to levels_end over [t, t+ramp) then hold;
        the goal follows its schedule."""
        c = centers[rows]
        s = np.clip((c - t) / float(ramp), 0.0, 1.0)
        q = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
        g = level_now[None, :] + (levels_end - level_now)[None, :] * q[:, None]
        g[:, 0] = goal_curve.values(c)
        return g

    # ---- search loop
    search_end = rise_start if rise_start is not None else unit.end
    t = unit.start
    level = state.level.copy()
    level[0] = float(goal_curve.values(np.array([unit.start]))[0])
    ref_store = unit.xi_goal.copy()
    ref_used = np.zeros(unit.J, dtype=bool)
    xi_committed = np.zeros((unit.J, unit.d_xi))
    c_committed = np.zeros((unit.J, M))
    step_logs: List[Dict[str, Any]] = []
    n_commits = 0
    total_evals = 0
    jumps_made = 0
    mat = [i for i in range(1, M)]

    def positions_for(starts: np.ndarray, extra_jump: Optional[Tuple[int, int]] = None) -> np.ndarray:
        pos = np.zeros((len(starts), M), dtype=np.int64)
        for i in range(M):
            cv = state.curve(i)
            if extra_jump is not None and extra_jump[0] == i:
                cv = TrackCurve([], [], list(state.clips[i]) + [Clip(int(t), int(extra_jump[1]))])
            pos[:, i] = cv.positions(starts, sources[i].shape[0])
        return pos

    while t < search_end:
        w_end = min(t + look, search_end)
        rows = np.where((centers >= t) & (centers < w_end))[0]
        if len(rows) == 0:
            t = w_end
            continue
        st = starts_all[rows]
        # current materials at the played positions -> unit context rows (what the mode sees)
        pos_now = positions_for(st)
        f, S, chi = an.material_features_at(pos_now)
        unit.f_mat[rows] = f
        unit.S[rows] = S
        unit.chi[rows] = chi
        prev_rows = np.where(centers < t)[0]
        if len(prev_rows) and ref_used[prev_rows[-1]]:
            xi_current = xi_committed[prev_rows[-1]]
        else:
            G0c, Gbc = an.grams_at_positions(sources, st[:1], pos_now[:1])
            xi_current = an.composition_from_grams(level[None, :], G0c, Gbc, S[:1])[0][0]
        props = mode.prepare_reference(unit, history, rows, xi_current, n_ref) or [R0]
        # evaluate the current tail (levels held) against each proposal to pick the reference, then freeze
        G0n, Gbn = an.grams_at_positions(sources, st, pos_now)
        g_hold = gains_for(level.copy(), t, rows, level, [])
        xi_hold, parts_hold = an.composition_from_grams(g_hold, G0n, Gbn, S)

        def J_of(xi_w, parts_w, ref):
            em = float(mode.window_error(unit, xi_w, rows, ref))
            fm = unit.free_mask[rows]
            fit = float(an.dist2(xi_w, ref.xi_hat[rows])[fm].mean()) if fm.any() else 0.0
            ef = objective.e_form_rows(unit, xi_w, parts_w, rows)
            jr = objective.j_relation(unit, parts_w, rows, rel_terms)
            J = objective.w_mode * objective.normalize_mode_error(job.mode_name, em) + objective.w_form * ef + w_rel * jr
            return {"J": J, "mode_error": em, "fit": fit, "penalty": em - fit, "e_form": ef, "j_rel": jr}

        ev0 = [J_of(xi_hold, parts_hold, tg) for tg in props]
        k_ref = int(np.argmin([e["J"] for e in ev0]))
        ref = props[k_ref]
        ref_hash = _hash_array(ref.xi_hat[rows])
        init = ev0[k_ref]
        evals = len(props)
        # ---- combinations: no jump + one-track jumps toward the reference profile
        combos: List[Tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]] = [(None, G0n, Gbn, S)]
        if bool(hz["position_jumps"]):
            ratios_t = ref.xi_hat[rows][:, 1:1 + an.nb].mean(axis=0)     # normalized band profile of the reference
            for i in mat:
                if t - state.last_jump[i] < min_clip:
                    continue
                fbank = an.solo_f[i]
                d = ((fbank[:, 1:1 + an.nb] - ratios_t[None, :]) ** 2).sum(axis=1)
                d = d + 4.0 * (an.solo_E[i] < an.silence_energy)          # avoid silent positions
                order = np.argsort(d)
                cands = [int(an.solo_pos[i][k]) for k in order[: max(1, n_jump - 1)]]
                cands.append(int(rng.integers(0, sources[i].shape[0])))   # exploration
                for p in cands:
                    pos_j = positions_for(st, (i, p))
                    fj, Sj, _chij = an.material_features_at(pos_j)
                    G0j, Gbj = an.grams_at_positions(sources, st, pos_j)
                    combos.append(((i, p), G0j, Gbj, Sj))
        best = None
        for (jump, G0w, Gbw, Sw) in combos:
            lv = level.copy()
            g = gains_for(lv, t, rows, level, [])
            xi_w, parts_w = an.composition_from_grams(g, G0w, Gbw, Sw)
            cur_ev = J_of(xi_w, parts_w, ref)
            evals += 1
            step = level_step
            for _sweep in range(max_sweeps):
                improved = False
                for i in mat:
                    for sign in (1.0, -1.0):
                        trial = lv.copy()
                        trial[i] = float(np.clip(trial[i] + sign * step, 0.0, 1.0))
                        if abs(trial[i] - lv[i]) < 1e-9:
                            continue
                        g = gains_for(trial, t, rows, level, [])
                        xi_t, parts_t = an.composition_from_grams(g, G0w, Gbw, Sw)
                        ev = J_of(xi_t, parts_t, ref)
                        evals += 1
                        if ev["J"] < cur_ev["J"] - 1e-6:
                            cur_ev, lv, xi_w, parts_w = ev, trial, xi_t, parts_t
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 0.02:
                        break
            if best is None or cur_ev["J"] < best[0]["J"]:
                best = (cur_ev, lv, jump, xi_w, parts_w, Sw)
        fin, lv_best, jump, xi_w, parts_w, S_w = best
        total_evals += evals
        # ---- apply: clip jump, segments for [t, t+commit)
        c_end = min(t + commit, search_end)
        if jump is not None:
            i, p = jump
            state.clips[i].append(Clip(int(t), int(p)))
            state.last_jump[i] = t
            jumps_made += 1
        for i in range(M):
            if i == 0:
                continue
            a, b = float(level[i]), float(lv_best[i])
            r_end = min(t + ramp, c_end)
            if abs(b - a) > 1e-12 and r_end > t:
                state.segments[i].append(Segment("Q5", t, r_end, a, b, "SWITCH"))
                if c_end > r_end:
                    state.segments[i].append(Segment("HOLD", r_end, c_end, b, b, "LEVEL_HOLD"))
            else:
                state.segments[i].append(Segment("HOLD", t, c_end, a, a, "ZERO_HOLD" if a <= 1e-12 else "LEVEL_HOLD"))
        rows_c = rows[centers[rows] < c_end]
        if len(rows_c) == 0:
            rows_c = rows[:1]
        k_c = np.searchsorted(rows, rows_c)
        xi_c = xi_w[k_c]
        parts_c = {k: (v[k_c] if isinstance(v, np.ndarray) and v.shape[0] == len(rows) else v) for k, v in parts_w.items()}
        xi_committed[rows_c] = xi_c
        c_committed[rows_c] = parts_c["c"]
        ref_store[rows_c] = ref.xi_hat[rows_c]
        ref_used[rows_c] = True
        events = []
        for i in mat:
            if abs(lv_best[i] - level[i]) >= 0.02:
                j1 = int(rows_c[-1])
                j0 = int(prev_rows[-1]) if len(prev_rows) else int(rows_c[0])
                events.append({"unit": unit.index, "track": i, "start_frame": int(t), "end_frame": int(min(t + ramp, c_end)),
                               "start_seconds": t / fs, "end_seconds": min(t + ramp, c_end) / fs,
                               "direction": int(np.sign(lv_best[i] - level[i])), "start_gain": float(level[i]),
                               "end_gain": float(lv_best[i]), "phase": str(unit.phase_names[j1]),
                               "xi_start": xi_committed[j0].copy() if ref_used[j0] else xi_c[0].copy(),
                               "xi_end": xi_c[-1].copy(), "dxi": (xi_c[-1] - (xi_committed[j0] if ref_used[j0] else xi_c[0])).copy(),
                               "c_end": c_committed[j1].copy()})
        dt = (c_end - t) / float(fs)
        hlog = history.observe_committed(unit, rows_c, xi_c, parts_c, dt, events)
        mstat = mode.observe_committed(unit, history, rows_c, xi_c, parts_c, ref,
                                       {"refinement": {"initial": init, "final": fin}, "reference_hash": ref_hash,
                                        "k_ref": k_ref, "n_proposals": len(props), "proposal_initial_J": [e["J"] for e in ev0]})
        n_commits += 1
        step_logs.append({"t_seconds": t / fs, "window_seconds": [t / fs, w_end / fs], "rows": int(len(rows)),
                          "committed_rows": int(len(rows_c)), "reference_id": ref.id, "reference_hash": ref_hash,
                          "reference_updated_during_realization": False, "reference_candidates": len(props),
                          "reference_chosen_index": k_ref, "initial_joint_objective": init["J"], "final_joint_objective": fin["J"],
                          "initial_fixed_target_error": init["fit"], "final_fixed_target_error": fin["fit"],
                          "mode_term_breakdown": {"initial": init, "final": fin}, "evaluations": evals,
                          "accepted_changes": int(np.sum(np.abs(lv_best - level) > 1e-9)), "jump": (list(jump) if jump else None),
                          "combos": len(combos), "levels": lv_best.tolist(), "history_rho": hlog.get("rho"),
                          "events_committed": len(events), "mode_observe": mstat})
        level = lv_best.copy()
        level[0] = float(goal_curve.values(np.array([c_end - 1]))[0]) if c_end > unit.start else level[0]
        t = c_end
    # ---- deterministic tail: materials ramp to 0 over the goal rise, goal follows its schedule
    if rise_start is not None:
        for i in mat:
            a = float(level[i])
            if abs(a) > 1e-12:
                state.segments[i].append(Segment("Q5", rise_start, t1, a, 0.0, "MOVE"))
            else:
                state.segments[i].append(Segment("HOLD", rise_start, t1, 0.0, 0.0, "ZERO_HOLD"))
            if unit.end > t1:
                state.segments[i].append(Segment("HOLD", t1, unit.end, 0.0, 0.0, "GOAL_HOLD"))
            level[i] = 0.0
    else:
        for i in mat:
            a = float(level[i])
            if t < unit.end:
                state.segments[i].append(Segment("HOLD", t, unit.end, a, a, "ZERO_HOLD" if a <= 1e-12 else "LEVEL_HOLD"))
    state.segments[0].extend(goal_segs)
    state.level = level.copy()
    state.level[0] = float(goal_curve.values(np.array([unit.end - 1]))[0])
    # ---- final unit composition at the actual positions
    curves = [TrackCurve([sg for sg in state.segments[i] if sg.start >= unit.start and sg.end <= unit.end], [], list(state.clips[i]))
              for i in range(M)]
    pos_all = np.stack([curves[i].positions(starts_all, sources[i].shape[0]) for i in range(M)], axis=1)
    f_all, S_all, chi_all = an.material_features_at(pos_all)
    unit.f_mat[:] = f_all
    unit.S[:] = S_all
    unit.chi[:] = chi_all
    gains = np.stack([cv.values(centers) for cv in curves], axis=1)
    G0a, Gba = an.grams_at_positions(sources, starts_all, pos_all)
    xi, parts = an.composition_from_grams(gains, G0a, Gba, S_all)
    final = Candidate(id=20_000 + unit.index, plan=[], curves=curves, gains=gains, xi=xi, parts=parts,
                      origin="hires:level+position search", parent_ids=[])
    final.e_form = objective.e_form(unit, xi, parts)
    final.e_hist = objective.e_hist(unit, xi, history)
    final.e_motion = 0.0
    chosen = job._score(unit, final, R0)
    fm = unit.free_mask
    full_resid = float(an.dist2(xi, ref_store)[fm & ref_used].mean()) if (fm & ref_used).any() else 0.0
    job.unit_refs[unit.index] = ref_store
    hist_eff = job._history_effect(unit, [final], pre_mode, pre_rng_state, 2)
    mode.end_unit(unit, history, chosen, [])
    blocks = objective.phase_blocks(unit, xi)
    history.update_unit_archive(unit, final, blocks, fs)
    J0s = [sl["initial_joint_objective"] for sl in step_logs]
    J1s = [sl["final_joint_objective"] for sl in step_logs]
    f0s = [sl["initial_fixed_target_error"] for sl in step_logs]
    f1s = [sl["final_fixed_target_error"] for sl in step_logs]
    job.unit_reports.append({
        "unit": unit.index, "frames": [int(unit.start), int(unit.end)], "goal_arrival_frame": unit.goal_arrival,
        "grid_points": int(unit.J), "bank": {"generated": 0, "rejected_illegal": 0, "notes": [], "relations": {}},
        "candidates_evaluated": 0, "budget": 0,
        "unit_reference": {"id": R0.id, "hash": _hash_array(R0.xi_hat), "proposals": len(props0)},
        "steps": step_logs, "commits": n_commits, "history_updates": n_commits, "step_evaluations": total_evals,
        "jumps": jumps_made,
        "realization_summary": {
            "mean_initial_joint_objective": float(np.mean(J0s)) if J0s else None,
            "mean_final_joint_objective": float(np.mean(J1s)) if J1s else None,
            "mean_initial_fixed_target_error": float(np.mean(f0s)) if f0s else None,
            "mean_final_fixed_target_error": float(np.mean(f1s)) if f1s else None,
            "steps_with_accepted_changes": int(sum(1 for sl in step_logs if sl["accepted_changes"] > 0 or sl["jump"])),
            "total_accepted_changes": int(sum(sl["accepted_changes"] for sl in step_logs)),
            "full_grid_fixed_target_residual": full_resid,
            "note": "hires: end levels and playback positions improved against frozen window references",
        },
        "tolerance_met": bool(job._tolerance_quantity(chosen) <= float(cfg["search"]["normalized_mode_tolerance"])),
        "tolerance_applies_to": cfg["search"]["tolerance_applies_to"], "tolerance": float(cfg["search"]["normalized_mode_tolerance"]),
        "chosen": {"candidate_id": final.id, "origin": final.origin, "target_id": R0.id, "total": chosen.total,
                   "mode_error": chosen.mode_error, "normalized_mode_error": chosen.normalized_mode_error,
                   "target_fit_error": chosen.extra["target_fit_error"], "mode_penalty": chosen.extra["mode_penalty"],
                   "e_form": chosen.e_form, "e_hist": chosen.e_hist, "e_motion": chosen.e_motion},
        "alternatives": [], "acoustic_diversity_mean_pair_dist2": 0.0,
        "history_internal_effect": hist_eff["signature_diff"], "history_ideal_effect": hist_eff["ideal_diff"],
        "history_realized_effect": hist_eff["realized_diff"], "history_probe": hist_eff,
        "seconds": time.time() - t_unit,
    })
    job.unit_chosen.append((unit, chosen))
    return chosen

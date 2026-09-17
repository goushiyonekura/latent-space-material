"""High-resolution realization (user-authorised extension, 2026-09-16).

Authorised beyond spec v1.1: steep gain switches (short Q5 ramps exempt from the slow-motion
rules), playback-position jumps / splicing of each material within its own source, and output
dynamics.  Kept: gain per track is still the actuator (plus the position map), the goal track
keeps its continuous clock and exact goal holds, references are frozen per window, only the
gain / position choices are improved against them, and committed blocks feed the history.

Per commit step (commit_seconds):
  1. the actual material features at the *played* positions are written into the unit context
     rows of the window, then the mode prepares references; one is frozen (hash);
  2. jump candidates per material: fragment positions whose clip-averaged band profile (what
     will sound for min_clip_seconds after the jump) is closest to the reference mixture
     profile, plus one random exploration position, subject to the minimum clip length;
     fragment mode: a beam search over tracks (multi-track jumps) scored on exact mixtures of a
     row subset; v1 mode: no-jump plus single-track jumps;
  3. for each surviving combination the window Gram matrices are computed once from the actual
     PCM at those positions (exact for the summed PCM under the window-centre gain hold); the end
     levels of the materials (reached by a Q5 ramp of ramp_seconds) are improved by a
     finite-difference coordinate search on the joint objective; the best combination wins;
  4. the first commit block is committed (history, mode observation, events with positions).
The goal track follows the exposure policy deterministically (0 in INTRO/OPEN, smooth rise over
goal_rise_seconds at the end of CONTRACT, exact 1 in GOAL_HOLD, smooth descent in REOPEN).

Reference hold (opt-in, `hires.reference_hold_seconds` > commit_seconds; docs/HOLD_CONTRACT.md):
a reference that is re-anchored to the realized composition at every commit follows the sound
instead of leading it (measured: the closeness of the fragment version is largely automatic and
the 0.5 s moves of the ideal do not reach the sound).  With a hold, step 1 runs once per hold:
the mode prepares ONE ideal over the whole held span from the anchor (the mean of the last
committed block), it stays frozen for every commit of the hold, and only then is a new ideal
anchored to where the sound has arrived.  Each hold is recorded (anchor, requested move).
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
    frag = bool(hz.get("fragment_vocabulary", False))
    beam_width = max(1, int(hz.get("beam_width", 3)))
    cand_rows = max(2, int(hz.get("candidate_rows", 8)))
    hold = int(round(float(hz.get("reference_hold_seconds", 0.0)) * fs))
    hold_on = hold > commit                    # legacy: a new reference at every commit
    ref_select = str(hz.get("reference_selection", "hold_fit"))
    anchor_kind = str(hz.get("reference_anchor", "last_row")) if hold_on else "last_row"
    plan_aware = bool(hz.get("plan_aware_lookahead", True))
    explore_max = int(hz.get("explore_tracks_max", 0))
    ge = cfg["form"]["goal_exposure"]
    goal_free = ge["policy"] == "free"
    open_cap = float(ge["open_max"]) if not goal_free else 1.0
    W = an.W
    half = W // 2
    centers = unit.centers
    starts_all = centers - half

    _src = getattr(mode, "sources", None)
    mode.sources = None                        # do not deep-copy the PCM with the mode snapshot
    pre_mode = copy.deepcopy(mode)
    mode.sources = _src
    pre_mode.sources = _src
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

    def gains_for(levels_end: np.ndarray, t: int, rows: np.ndarray, level_now: np.ndarray):
        c = centers[rows]
        s = np.clip((c - t) / float(ramp), 0.0, 1.0)
        q = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
        g = level_now[None, :] + (levels_end - level_now)[None, :] * q[:, None]
        g[:, 0] = goal_curve.values(c)
        return g

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

    def positions_for(starts: np.ndarray, jumps: Optional[Dict[int, int]] = None) -> np.ndarray:
        pos = np.zeros((len(starts), M), dtype=np.int64)
        for i in range(M):
            cv = state.curve(i)
            if jumps and i in jumps:
                cv = TrackCurve([], [], list(state.clips[i]) + [Clip(int(t), int(jumps[i]))])
            pos[:, i] = cv.positions(starts, sources[i].shape[0])
        return pos

    flux_prev: Dict[str, Any] = {"row": None, "ratios": None}   # last committed row and its raw band ratios

    def prev_for(rws) -> Optional[np.ndarray]:
        """Band ratios of the committed row just before rws[0] (hold mode), else None."""
        if hold_on and flux_prev["row"] is not None and len(rws) and int(rws[0]) == int(flux_prev["row"]) + 1:
            return flux_prev["ratios"]
        return None

    def plan_rows(steps, rws: np.ndarray, level_now: np.ndarray):
        """EXACT composition rows of a realizable plan (docs/HOLD_CONTRACT.md).  `steps` is a list
        of {"frame": f, "jumps": {track: source_position}, "levels": (M,) end levels or None} with
        frames on the commit grid (ascending): at frame f the listed tracks jump (when the minimum
        clip length allows it; others are dropped and reported) and every material ramps from its
        level to `levels` over ramp_seconds - exactly what the realizer itself can do at that
        commit.  The goal track follows its deterministic schedule.  Returns (xi, parts, info)."""
        rws = np.asarray(rws, dtype=np.int64)
        c = centers[rws]
        last = list(state.last_jump)
        extra: List[List[Clip]] = [[] for _ in range(M)]
        dropped: List[Tuple[int, int]] = []
        lv = np.asarray(level_now, dtype=np.float64).copy()
        g = np.repeat(lv[None, :], len(rws), axis=0)
        for s_ in sorted(steps, key=lambda x: int(x["frame"])):
            f0 = int(s_["frame"])
            for i_, p_ in (s_.get("jumps") or {}).items():
                i_ = int(i_)
                if i_ <= 0 or i_ >= M or not bool(hz["position_jumps"]) or f0 - last[i_] < min_clip:
                    dropped.append((f0, i_))
                    continue
                extra[i_].append(Clip(f0, int(p_)))
                last[i_] = f0
            if s_.get("levels") is not None:
                end = np.clip(np.asarray(s_["levels"], dtype=np.float64), 0.0, 1.0)
                m_ = c >= f0
                sr = np.clip((c[m_] - f0) / float(ramp), 0.0, 1.0)
                q = sr * sr * sr * (10.0 + sr * (-15.0 + 6.0 * sr))
                g[m_] = lv[None, :] + (end - lv)[None, :] * q[:, None]
                lv = end
        g[:, 0] = goal_curve.values(c)
        pos = np.zeros((len(rws), M), dtype=np.int64)
        for i_ in range(M):
            cv = TrackCurve([], [], list(state.clips[i_]) + extra[i_])
            pos[:, i_] = cv.positions(starts_all[rws], sources[i_].shape[0])
        _fp, S_p, _cp = an.material_features_at(pos)
        G0p, Gbp = an.grams_at_positions(sources, starts_all[rws], pos)
        xi_p, parts_p = an.composition_from_grams(g, G0p, Gbp, S_p, prev_ratios=prev_for(rws))
        return xi_p, parts_p, {"dropped_jumps": dropped, "gains": g, "positions": pos}

    # ---- plan-aware lookahead (hold mode): a candidate is "this choice now, then the rest of the mode's
    # plan", not "this choice held for the whole window" - otherwise a plan that changes levels at every
    # commit is compared with something the realizer never intends to play (docs/HOLD_CONTRACT.md)
    def positions_plan(starts: np.ndarray, jumps: Optional[Dict[int, int]], future: List[Dict[str, Any]]) -> np.ndarray:
        if not future or not any(s_.get("jumps") for s_ in future):
            return positions_for(starts, jumps)
        pos = np.zeros((len(starts), M), dtype=np.int64)
        for i in range(M):
            clips_i = list(state.clips[i])
            last = state.last_jump[i]
            if jumps and i in jumps:
                clips_i.append(Clip(int(t), int(jumps[i])))
                last = int(t)
            for s_ in future:
                for i_, p_ in (s_.get("jumps") or {}).items():
                    if int(i_) == i and i > 0 and bool(hz["position_jumps"]) and int(s_["frame"]) - last >= min_clip:
                        clips_i.append(Clip(int(s_["frame"]), int(p_)))
                        last = int(s_["frame"])
            pos[:, i] = TrackCurve([], [], clips_i).positions(starts, sources[i].shape[0])
        return pos

    def gains_plan(levels_end: np.ndarray, t_now: int, rws: np.ndarray, level_now: np.ndarray,
                   future: List[Dict[str, Any]]) -> np.ndarray:
        g = gains_for(levels_end, t_now, rws, level_now)
        if not future:
            return g
        c = centers[rws]
        prev = np.asarray(levels_end, dtype=np.float64)
        for s_ in future:
            if s_.get("levels") is None:
                continue
            end = np.clip(np.asarray(s_["levels"], dtype=np.float64), 0.0, 1.0)
            m_ = c >= int(s_["frame"])
            if m_.any():
                sr = np.clip((c[m_] - int(s_["frame"])) / float(ramp), 0.0, 1.0)
                q = sr * sr * sr * (10.0 + sr * (-15.0 + 6.0 * sr))
                g[m_, 1:] = prev[None, 1:] + (end[1:] - prev[1:])[None, :] * q[:, None]
            prev = end
        return g

    held: Optional[Dict[str, Any]] = None      # the reference being chased (hold mode)
    hold_segments: List[Dict[str, Any]] = []
    last_rows_c: Optional[np.ndarray] = None
    while t < search_end:
        w_end = min(t + look, search_end)
        rows = np.where((centers >= t) & (centers < w_end))[0]
        if len(rows) == 0:
            t = w_end
            continue
        new_ref = (not hold_on) or held is None or (t - held["t0"] >= hold) or int(rows[-1]) > held["last_row"]
        rows_ref = rows
        if hold_on and new_ref:
            # the held ideal must cover the window of every commit of the hold
            rows_ref = np.where((centers >= t) & (centers < min(t + hold - commit + look, search_end)))[0]
        st = starts_all[rows]
        if len(rows_ref) > len(rows):
            pos_ref = positions_for(starts_all[rows_ref])
            f_r, S_r, chi_r = an.material_features_at(pos_ref)
            unit.f_mat[rows_ref] = f_r
            unit.S[rows_ref] = S_r
            unit.chi[rows_ref] = chi_r
            pos_now, f, S, chi = pos_ref[:len(rows)], f_r[:len(rows)], S_r[:len(rows)], chi_r[:len(rows)]
        else:
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
        if new_ref:
            xi_anchor = xi_current
            if anchor_kind == "block_mean" and last_rows_c is not None and len(last_rows_c):
                xi_anchor = xi_committed[last_rows_c].mean(axis=0)
            if hold_on:
                # what the realizer can do from here, for modes that build realizable ideals
                lv_now = level.copy()
                unit.realizer_state = {
                    "frame": int(t), "rows": rows_ref.copy(), "levels": lv_now,
                    "positions": pos_ref[0].copy() if len(rows_ref) > len(rows) else pos_now[0].copy(),
                    "next_jump_frame": [int(state.last_jump[i] + min_clip) for i in range(M)],
                    "jumps_enabled": bool(hz["position_jumps"]),
                    "commit_frames": int(commit), "ramp_frames": int(ramp), "hold_frames": int(hold),
                    "min_clip_frames": int(min_clip), "lookahead_frames": int(look), "search_end_frame": int(search_end),
                    "goal_gains": goal_curve.values(centers[rows_ref]),
                    "plan_rows": (lambda steps, rws=None, _lv=lv_now, _rr=rows_ref:
                                  plan_rows(steps, _rr if rws is None else rws, _lv)),
                }
            props = mode.prepare_reference(unit, history, rows_ref, xi_anchor, n_ref) or [R0]
        G0n, Gbn = an.grams_at_positions(sources, st, pos_now)
        g_hold = gains_for(level.copy(), t, rows, level)
        xi_hold, parts_hold = an.composition_from_grams(g_hold, G0n, Gbn, S, prev_ratios=prev_for(rows))

        def J_of(xi_w, parts_w, ref, rws):
            em = float(mode.window_error(unit, xi_w, rws, ref))
            fm = unit.free_mask[rws]
            fit = float(an.dist2(xi_w, ref.xi_hat[rws])[fm].mean()) if fm.any() else 0.0
            ef = objective.e_form_rows(unit, xi_w, parts_w, rws)
            jr = objective.j_relation(unit, parts_w, rws, rel_terms)
            J = objective.w_mode * objective.normalize_mode_error(job.mode_name, em) + objective.w_form * ef + w_rel * jr
            return {"J": J, "mode_error": em, "fit": fit, "penalty": em - fit, "e_form": ef, "j_rel": jr}

        if new_ref:
            ev0 = [J_of(xi_hold, parts_hold, tg, rows) for tg in props]
            if ref_select == "first":
                # no selection: choosing the proposal nearest to the unchanged sound ("hold_fit")
                # systematically shrinks the move the mode asks for
                k_ref = 0
            else:
                k_ref = int(np.argmin([e["J"] for e in ev0]))
            ref = props[k_ref]
            init = ev0[k_ref]
            evals = len(props)
            if hold_on:
                # requested move: anchor -> ideal at the end of the hold (mean of its last block)
                e_rows = rows_ref[(centers[rows_ref] >= t + hold - commit) & (centers[rows_ref] < t + hold)]
                if len(e_rows) == 0:
                    e_rows = rows_ref[-1:]
                e_free = e_rows[unit.free_mask[e_rows]]
                req = (float(an.dist2(ref.xi_hat[e_free].mean(axis=0)[None, :], np.asarray(xi_anchor)[None, :])[0])
                       if len(e_free) else None)
                # the same block if nothing were changed (the material's own drift): what the plan ADDS to it
                # is its intervention - the part of the requested move that is not achieved automatically
                stay_end, req_net = None, None
                if len(e_free):
                    k_e = np.searchsorted(rows_ref, e_free)
                    xi_stay, _ps, _is = plan_rows([], rows_ref, level)
                    stay_end = xi_stay[k_e].mean(axis=0)
                    req_net = float(an.dist2(ref.xi_hat[e_free].mean(axis=0)[None, :], stay_end[None, :])[0])
                held = {"ref": ref, "props": props, "k_ref": k_ref, "ev0": ev0, "t0": int(t),
                        "last_row": int(rows_ref[-1]), "age": 0, "pending": {},
                        "segment": {"t0_seconds": t / fs, "anchor_kind": anchor_kind,
                                    "anchor": np.asarray(xi_anchor, dtype=np.float64).tolist(),
                                    "reference_id": ref.id, "n_proposals": len(props), "k_ref": k_ref,
                                    "selection": ref_select, "reference_hash": _hash_array(ref.xi_hat[rows_ref]),
                                    "reference_rows": [int(rows_ref[0]), int(rows_ref[-1])],
                                    "committed_rows": [int(rows[0]), int(rows[0])],
                                    "openness": float(unit.o[rows[0]]), "phase": str(unit.phase_names[rows[0]]),
                                    "requested_dist2_at_hold_end": req,
                                    "stay_end": (stay_end.tolist() if stay_end is not None else None),
                                    "intervention_dist2_at_hold_end": req_net}}
                hold_segments.append(held["segment"])
        else:
            props, k_ref, ev0, ref = held["props"], held["k_ref"], held["ev0"], held["ref"]
            held["age"] += 1
            init = J_of(xi_hold, parts_hold, ref, rows)
            evals = 1
        # the rest of the mode's plan (steps after this commit), assumed to be played as planned
        future: List[Dict[str, Any]] = []
        if hold_on and plan_aware:
            future = sorted([s_ for s_ in (ref.meta.get("plan") or []) if int(s_.get("frame", -1)) > int(t)],
                            key=lambda s_: int(s_["frame"]))
        G0z, Gbz, S_z = G0n, Gbn, S                # Grams of "no jump now"
        if future:
            if any(s_.get("jumps") for s_ in future):
                pos_z = positions_plan(st, None, future)
                _fz, S_z, _cz = an.material_features_at(pos_z)
                G0z, Gbz = an.grams_at_positions(sources, st, pos_z)
            g_hold = gains_plan(level.copy(), t, rows, level, future)
            xi_hold, parts_hold = an.composition_from_grams(g_hold, G0z, Gbz, S_z, prev_ratios=prev_for(rows))
            init = J_of(xi_hold, parts_hold, ref, rows)
            evals += 1
        ref_hash = _hash_array(ref.xi_hat[rows])
        ratios_t = ref.xi_hat[rows][:, 1:1 + an.nb].mean(axis=0)
        # ---- jump candidates per material (clip-averaged fragment features in fragment mode)
        cands: Dict[int, List[int]] = {}
        explore = list(mat)
        if explore_max > 0 and hold_on:
            # many tracks (second voices, 8-12 materials): the realizer explores its own jump candidates
            # on a few tracks per commit only; the tracks of the mode's plan step are always examined below
            free_now = [i for i in mat if t - state.last_jump[i] >= min_clip]
            if len(free_now) > explore_max:
                explore = sorted(int(i) for i in rng.choice(free_now, size=explore_max, replace=False))
        if bool(hz["position_jumps"]):
            for i in explore:
                if t - state.last_jump[i] < min_clip:
                    continue
                cur_pos = int(pos_now[0, i])
                if frag:
                    lst = an.fragment_candidates(i, ratios_t, max(1, n_jump - 1), exclude_near=cur_pos,
                                                 exclude_frames=int(2 * min_clip))
                else:
                    fbank = an.solo_f[i]
                    d = ((fbank[:, 1:1 + an.nb] - ratios_t[None, :]) ** 2).sum(axis=1) + 4.0 * (an.solo_E[i] < an.silence_energy)
                    lst = [int(an.solo_pos[i][k]) for k in np.argsort(d)[: max(1, n_jump - 1)]]
                lst.append(int(rng.integers(0, sources[i].shape[0])))
                cands[i] = lst
        # ---- plan hints of a held reference (docs/HOLD_CONTRACT.md): the positions / end levels the
        # mode built its ideal from join the candidates; the objective against the frozen ideal decides
        hint_jumps: Dict[int, int] = {}
        hint_levels: Optional[np.ndarray] = None
        if hold_on:
            for s_ in (ref.meta.get("plan") or []):
                if int(s_.get("frame", -1)) != int(t):
                    continue
                for i_, p_ in (s_.get("jumps") or {}).items():
                    held["pending"][int(i_)] = (int(t), int(p_))
                if s_.get("levels") is not None:
                    hint_levels = np.clip(np.asarray(s_["levels"], dtype=np.float64), 0.0, 1.0)
            if bool(hz["position_jumps"]):
                for i_, (f0_, p0_) in list(held["pending"].items()):
                    if i_ in mat and t - state.last_jump[i_] >= min_clip:
                        # a hinted jump that could not happen at its frame stays aligned in time
                        hint_jumps[i_] = int((p0_ + (t - f0_)) % sources[i_].shape[0])
                        cands.setdefault(i_, [])
                        if hint_jumps[i_] not in cands[i_]:
                            cands[i_].insert(0, hint_jumps[i_])
        # ---- combinations
        combos: List[Dict[int, int]] = [{}]
        if frag and cands:
            sub = rows[np.linspace(0, len(rows) - 1, min(cand_rows, len(rows))).astype(int)]
            st_sub = starts_all[sub]

            def score(jumps: Dict[int, int]) -> float:
                pos_s = positions_plan(st_sub, jumps, future)
                _f, S_s, _c = an.material_features_at(pos_s)
                G0s, Gbs = an.grams_at_positions(sources, st_sub, pos_s)
                g = gains_plan(level.copy(), t, sub, level, future)
                xi_s, parts_s = an.composition_from_grams(g, G0s, Gbs, S_s, prev_ratios=prev_for(sub))
                return J_of(xi_s, parts_s, ref, sub)["J"]

            beam: List[Tuple[Dict[int, int], float]] = [({}, score({}))]
            evals += 1
            for i in mat:
                if i not in cands:
                    continue
                new_beam = list(beam)
                for (jumps, _J) in beam:
                    for p in cands[i]:
                        j2 = dict(jumps)
                        j2[i] = p
                        new_beam.append((j2, score(j2)))
                        evals += 1
                new_beam.sort(key=lambda x: x[1])
                beam = new_beam[:beam_width]
            combos = [b[0] for b in beam]
        elif cands:
            for i, lst in cands.items():
                for p in lst:
                    combos.append({i: p})
        if hint_jumps and hint_jumps not in combos:
            combos.append(dict(hint_jumps))        # the mode's whole plan step is always examined
        # ---- level search on each surviving combination (full window rows)
        best = None
        for jumps in combos:
            if jumps:
                pos_j = positions_plan(st, jumps, future)
                _fj, S_w, _cj = an.material_features_at(pos_j)
                G0w, Gbw = an.grams_at_positions(sources, st, pos_j)
            else:
                G0w, Gbw, S_w = G0z, Gbz, S_z
            lv = level.copy()
            g = gains_plan(lv, t, rows, level, future)
            xi_w, parts_w = an.composition_from_grams(g, G0w, Gbw, S_w, prev_ratios=prev_for(rows))
            cur_ev = J_of(xi_w, parts_w, ref, rows)
            evals += 1
            if hint_levels is not None:
                trial = lv.copy()
                trial[1:] = hint_levels[1:]
                g = gains_plan(trial, t, rows, level, future)
                xi_t, parts_t = an.composition_from_grams(g, G0w, Gbw, S_w, prev_ratios=prev_for(rows))
                ev = J_of(xi_t, parts_t, ref, rows)
                evals += 1
                if ev["J"] < cur_ev["J"] - 1e-6:       # the level search then starts from the hinted levels
                    cur_ev, lv, xi_w, parts_w = ev, trial, xi_t, parts_t
            step = level_step
            for _sweep in range(max_sweeps):
                improved = False
                for i in mat:
                    for sign in (1.0, -1.0):
                        trial = lv.copy()
                        trial[i] = float(np.clip(trial[i] + sign * step, 0.0, 1.0))
                        if abs(trial[i] - lv[i]) < 1e-9:
                            continue
                        g = gains_plan(trial, t, rows, level, future)
                        xi_t, parts_t = an.composition_from_grams(g, G0w, Gbw, S_w, prev_ratios=prev_for(rows))
                        ev = J_of(xi_t, parts_t, ref, rows)
                        evals += 1
                        if ev["J"] < cur_ev["J"] - 1e-6:
                            cur_ev, lv, xi_w, parts_w = ev, trial, xi_t, parts_t
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 0.02:
                        break
            if best is None or cur_ev["J"] < best[0]["J"]:
                best = (cur_ev, lv, jumps, xi_w, parts_w)
        fin, lv_best, jumps, xi_w, parts_w = best
        total_evals += evals
        # ---- apply: clip jumps, segments for [t, t+commit)
        c_end = min(t + commit, search_end)
        for i, p in jumps.items():
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
        flux_prev["row"] = int(rows_c[-1])
        flux_prev["ratios"] = np.asarray(parts_c["phi_raw"])[-1, 1:1 + an.nb].copy()
        ref_store[rows_c] = ref.xi_hat[rows_c]
        ref_used[rows_c] = True
        pos_end = positions_for(np.array([c_end - 1]), jumps)[0]
        events = []
        for i in mat:
            if abs(lv_best[i] - level[i]) >= 0.02 or i in jumps:
                j1 = int(rows_c[-1])
                j0 = int(prev_rows[-1]) if len(prev_rows) else int(rows_c[0])
                xi0 = xi_committed[j0] if ref_used[j0] else xi_c[0]
                events.append({"unit": unit.index, "track": i, "start_frame": int(t), "end_frame": int(min(t + ramp, c_end)),
                               "start_seconds": t / fs, "end_seconds": min(t + ramp, c_end) / fs,
                               "direction": int(np.sign(lv_best[i] - level[i])), "start_gain": float(level[i]),
                               "end_gain": float(lv_best[i]), "phase": str(unit.phase_names[j1]),
                               "xi_start": xi0.copy(), "xi_end": xi_c[-1].copy(), "dxi": (xi_c[-1] - xi0).copy(),
                               "c_end": c_committed[j1].copy(), "jump": bool(i in jumps),
                               "src_position": int(pos_end[i])})
        dt = (c_end - t) / float(fs)
        hlog = history.observe_committed(unit, rows_c, xi_c, parts_c, dt, events)
        mstats = {"refinement": {"initial": init, "final": fin}, "reference_hash": ref_hash,
                  "k_ref": k_ref, "n_proposals": len(props), "proposal_initial_J": [e["J"] for e in ev0],
                  "positions": pos_end.tolist(), "jumps": {int(k): int(v) for k, v in jumps.items()},
                  "levels": lv_best.tolist(), "commit_end_frame": int(c_end)}
        if hold_on:
            mstats.update({"reference_hold_seconds": hold / float(fs), "reference_age_steps": int(held["age"]),
                           "reference_t0_frame": int(held["t0"]), "reference_is_new": bool(new_ref),
                           "plan_hint_jumps": {int(k): int(v) for k, v in hint_jumps.items()},
                           "plan_hint_jumps_taken": {int(i_): int(p_) for i_, p_ in jumps.items() if hint_jumps.get(i_) == p_},
                           "plan_hint_levels": (hint_levels.tolist() if hint_levels is not None else None)})
            held["segment"]["committed_rows"][1] = int(rows_c[-1])
        mstat = mode.observe_committed(unit, history, rows_c, xi_c, parts_c, ref, mstats)
        last_rows_c = rows_c
        n_commits += 1
        step_logs.append({"t_seconds": t / fs, "window_seconds": [t / fs, w_end / fs], "rows": int(len(rows)),
                          "committed_rows": int(len(rows_c)), "reference_id": ref.id, "reference_hash": ref_hash,
                          "reference_updated_during_realization": False, "reference_candidates": len(props),
                          "reference_chosen_index": k_ref, "initial_joint_objective": init["J"], "final_joint_objective": fin["J"],
                          "initial_fixed_target_error": init["fit"], "final_fixed_target_error": fin["fit"],
                          "mode_term_breakdown": {"initial": init, "final": fin}, "evaluations": evals,
                          "accepted_changes": int(np.sum(np.abs(lv_best - level) > 1e-9)),
                          "jumps": {int(k): int(v) for k, v in jumps.items()}, "combos": len(combos),
                          "levels": lv_best.tolist(), "history_rho": hlog.get("rho"), "events_committed": len(events),
                          "mode_observe": mstat})
        if hold_on:
            taken = {int(i_): int(p_) for i_, p_ in jumps.items() if hint_jumps.get(i_) == p_}
            step_logs[-1].update({"reference_is_new": bool(new_ref), "reference_age_steps": int(held["age"]),
                                  "plan_future_steps_assumed": int(len(future)),
                                  "plan_hint": {"jumps": {int(k): int(v) for k, v in hint_jumps.items()},
                                                "jumps_taken": taken, "levels_hinted": hint_levels is not None,
                                                "levels_distance_to_hint": (float(np.abs(lv_best[1:] - hint_levels[1:]).max())
                                                                            if hint_levels is not None else None)}})
            for i_ in jumps:
                held["pending"].pop(int(i_), None)
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
                      origin=("hires:fragment beam + level search" if frag else "hires:level+position search"), parent_ids=[])
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
    hold_report: Dict[str, Any] = {}
    if hold_on:
        # how the requested moves compare with the sound's own fast motion (docs/HOLD_CONTRACT.md):
        # flutter = variance (d2, Bessel-corrected) of the committed rows of a commit block around
        # their mean, on the rows AFTER the gain ramp - the realizer's own switches are not flutter
        blk = ((centers - unit.start) // commit).astype(np.int64)
        um = fm & ref_used & (((centers - unit.start) % commit) >= ramp)
        fl, fl_open = [], []
        for b in np.unique(blk[um]):
            r_ = um & (blk == b)
            n_ = int(r_.sum())
            if n_ >= 2:
                fl.append(float(an.dist2(xi[r_], xi[r_].mean(axis=0)[None, :]).mean()) * n_ / (n_ - 1.0))
                if float(unit.o[r_].min()) >= 0.8:
                    fl_open.append(fl[-1])
        flutter = float(np.mean(fl)) if fl else None
        flutter_open = float(np.mean(fl_open)) if fl_open else None
        reqs = [h["requested_dist2_at_hold_end"] for h in hold_segments if h["requested_dist2_at_hold_end"] is not None]
        reqs_open = [h["requested_dist2_at_hold_end"] for h in hold_segments
                     if h["requested_dist2_at_hold_end"] is not None and h["openness"] >= 0.8]
        hints = [sl["plan_hint"] for sl in step_logs if "plan_hint" in sl]
        unit.realizer_state = None
        hold_report = {
            "reference_hold_seconds": hold / float(fs), "reference_selection": ref_select, "reference_anchor": anchor_kind,
            "hold_segments": hold_segments,
            "hold_summary": {
                "holds": len(hold_segments), "realized_flutter_dist2": flutter,
                "realized_flutter_dist2_open": flutter_open,
                "mean_requested_dist2_at_hold_end": float(np.mean(reqs)) if reqs else None,
                "mean_requested_dist2_at_hold_end_open": float(np.mean(reqs_open)) if reqs_open else None,
                "requested_over_flutter": (float(np.mean(reqs) / flutter) if reqs and flutter else None),
                "requested_over_flutter_open": (float(np.mean(reqs_open) / flutter_open) if reqs_open and flutter_open else None),
                "mean_intervention_dist2_at_hold_end": (float(np.mean([h["intervention_dist2_at_hold_end"] for h in hold_segments
                                                                        if h.get("intervention_dist2_at_hold_end") is not None]))
                                                         if any(h.get("intervention_dist2_at_hold_end") is not None for h in hold_segments) else None),
                "plan_hints": {"steps_with_jump_hints": int(sum(1 for h in hints if h["jumps"])),
                               "hinted_jumps_offered": int(sum(len(h["jumps"]) for h in hints)),
                               "hinted_jumps_taken": int(sum(len(h["jumps_taken"]) for h in hints)),
                               "steps_with_level_hints": int(sum(1 for h in hints if h["levels_hinted"]))},
                "note": "requested = d2(anchor, held ideal averaged over the last commit block of the hold); "
                        "open = holds that start at openness >= 0.8"}}
    job.unit_reports.append({
        **hold_report,
        "unit": unit.index, "frames": [int(unit.start), int(unit.end)], "goal_arrival_frame": unit.goal_arrival,
        "grid_points": int(unit.J), "bank": {"generated": 0, "rejected_illegal": 0, "notes": [], "relations": {}},
        "candidates_evaluated": 0, "budget": 0,
        "unit_reference": {"id": R0.id, "hash": _hash_array(R0.xi_hat), "proposals": len(props0)},
        "steps": step_logs, "commits": n_commits, "history_updates": n_commits, "step_evaluations": total_evals,
        "jumps": jumps_made, "fragment_vocabulary": frag,
        "realization_summary": {
            "mean_initial_joint_objective": float(np.mean(J0s)) if J0s else None,
            "mean_final_joint_objective": float(np.mean(J1s)) if J1s else None,
            "mean_initial_fixed_target_error": float(np.mean(f0s)) if f0s else None,
            "mean_final_fixed_target_error": float(np.mean(f1s)) if f1s else None,
            "steps_with_accepted_changes": int(sum(1 for sl in step_logs if sl["accepted_changes"] > 0 or sl["jumps"])),
            "total_accepted_changes": int(sum(sl["accepted_changes"] for sl in step_logs)),
            "full_grid_fixed_target_residual": full_resid,
            "note": ("fragment mode: multi-track jump beam on exact mixtures + level search against frozen window references"
                     if frag else "hires: end levels and playback positions improved against frozen window references"),
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

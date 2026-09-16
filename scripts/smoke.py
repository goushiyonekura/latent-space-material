"""Checks A-C of spec §16 in one small script.

  python3 scripts/smoke.py            # all available modes
  python3 scripts/smoke.py vae gan    # subset

A: audio path + legal curves (read/write, Q5 analytic vs numeric derivatives, modulo playback,
   gain-only rendering, exact goal holds, re-render from the trace's analytic curves)
B: four modes: two contractions + final re-opening; history-only and one-material-only changes
   alter the mode's internal state / ideal distribution (realized differences recorded separately);
   variable N (2 and 7) dimension/CLI/candidate generation
C: artifacts (WAV/CSV/trace) with readable hard/soft results; a poor-approximation fixture still
   yields BEST_EFFORT within a bounded time.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latent_space import curves as C  # noqa: E402
from latent_space.audio_io import read_wav, write_wav_float32  # noqa: E402
from latent_space.config import DEFAULTS, deep_merge, load_config  # noqa: E402
from latent_space.curves import MotionLimits, Segment, TrackCurve  # noqa: E402
from latent_space.engine import Job  # noqa: E402
from latent_space.fixtures import write_fixture_config, write_fixture_set, synth_material  # noqa: E402
from latent_space.render import render_equation_1  # noqa: E402

DEV = os.path.join(ROOT, "dev", "smoke")
os.makedirs(DEV, exist_ok=True)
REPORT = {"A": {}, "B": {}, "C": {}, "failures": []}


def ok(name: str, cond: bool, detail=None):
    print(("  PASS " if cond else "  FAIL ") + name + ("" if detail is None else f"  ({detail})"))
    if not cond:
        REPORT["failures"].append(name)
    return bool(cond)


# ----------------------------------------------------------------------------- A
def check_A():
    print("[A] audio path and legal curves")
    fs = 22050
    lim = MotionLimits.from_config(DEFAULTS["motion"], DEFAULTS["numerics"])
    rng = np.random.default_rng(1)
    # WAV round trip
    x = (rng.standard_normal((1000, 2)) * 0.3).astype(np.float32)
    p = os.path.join(DEV, "rt.wav")
    write_wav_float32(p, x, fs)
    y, info = read_wav(p)
    ok("wav_float32_round_trip_exact", np.array_equal(x, y) and info.sample_rate == fs and info.channels == 2)
    # Q5: analytic peaks vs numeric derivatives at min feasible duration
    a, b = 0.0, 0.8
    T_lo, T_hi = C.move_bounds(a, b, lim)
    n = int(round(T_lo * fs))
    seg = Segment("Q5", 0, n, a, b, "MOVE")
    frames = np.arange(n + 1)
    g = TrackCurve([seg]).values(np.minimum(frames, n - 1))
    g[-1] = b
    d1 = np.diff(g) * fs
    # acceleration / jerk on a coarser grid (per-sample third differences are dominated by rounding)
    step = max(1, n // 2000)
    gc = g[::step]
    hc = step / fs
    d2 = np.diff(gc, 2) / hc ** 2
    d3 = np.diff(gc, 3) / hc ** 3
    vmax, amax, jmax = C.q5_peaks(abs(b - a), n / fs)
    ok("q5_numeric_velocity_matches_analytic", abs(np.max(np.abs(d1)) - vmax) < 1e-3 * vmax, f"{np.max(np.abs(d1)):.5f} vs {vmax:.5f}")
    ok("q5_numeric_acceleration_matches_analytic", abs(np.max(np.abs(d2)) - amax) < 1e-2 * amax)
    ok("q5_numeric_jerk_matches_analytic", abs(np.max(np.abs(d3)) - jmax) < 5e-2 * jmax)
    ok("q5_within_motion_limits", vmax <= lim.velocity_max * (1 + 1e-6) and amax <= lim.acceleration_max * (1 + 1e-6)
       and jmax <= lim.jerk_max * (1 + 1e-6), f"T_lo={T_lo:.3f}s T_hi={T_hi:.3f}s")
    ok("q5_check_segment_passes", C.check_segment(seg, fs, lim) == [])
    # numeric slow zone vs eq. (10)
    s_minus = C.q5_slow_zone(abs(b - a), n / fs, lim.velocity_min_bulk)
    slow_numeric = np.mean(np.abs(d1) < lim.velocity_min_bulk)
    ok("q5_slow_fraction_matches_eq10", abs(slow_numeric - 2 * s_minus) < 0.01, f"{slow_numeric:.4f} vs {2 * s_minus:.4f}")
    # numeric relative dB rate vs eq. (11) bound
    lam = 20 * np.log10((g + lim.gain_log_epsilon) / (1 + lim.gain_log_epsilon))
    dl = np.max(np.abs(np.diff(lam) * fs))
    bound = C.q5_db_rate_bound(a, b, n / fs, lim.gain_log_epsilon)
    ok("q5_db_rate_numeric_below_bound", dl <= bound * (1 + 1e-3) and bound <= lim.regularized_db_rate_max * (1 + 1e-6),
       f"numeric {dl:.3f} <= bound {bound:.3f} <= {lim.regularized_db_rate_max}")
    # a too-fast ramp is rejected; a same-direction stop is rejected; nonzero hold is rejected
    bad = Segment("Q5", 0, int(0.5 * T_lo * fs), a, b, "MOVE")
    ok("too_fast_ramp_rejected", any("exceeded" in v for v in C.check_segment(bad, fs, lim)))
    tc = TrackCurve([Segment("Q5", 0, n, 0.0, 0.4, "MOVE"), Segment("Q5", n, 2 * n, 0.4, 0.8, "MOVE")])
    ok("same_direction_stop_rejected", any("same_direction_stop" in v for v in tc.check(fs, lim)))
    tc2 = TrackCurve([Segment("HOLD", 0, n, 0.5, 0.5, "SCORED_HOLD")])
    ok("nonzero_hold_rejected_by_default", any("nonzero_hold" in v for v in tc2.check(fs, lim)))
    # rendering: modulo playback, gain-only, exact goal hold
    srcs = [synth_material(rng, fs, 1.5, "chord_pad"), synth_material(rng, fs, 2.2, "harmonic_drift"),
            synth_material(rng, fs, 1.1, "noise_texture")]
    total = int(4.0 * fs)
    n1 = int(1.5 * fs)
    cv0 = TrackCurve([Segment("Q5", 0, n1, 0.0, 1.0, "MOVE"), Segment("HOLD", n1, total, 1.0, 1.0, "GOAL_HOLD")])
    h1 = int(0.7 * fs)
    cv1 = TrackCurve([Segment("Q5", 0, h1, 0.0, 0.6, "MOVE"), Segment("Q5", h1, n1, 0.6, 0.0, "MOVE"),
                      Segment("HOLD", n1, total, 0.0, 0.0, "GOAL_HOLD")])
    cv2 = TrackCurve([Segment("HOLD", 0, int(0.3 * fs), 0.0, 0.0, "ZERO_HOLD"), Segment("Q5", int(0.3 * fs), h1, 0.0, 0.3, "MOVE"),
                      Segment("Q5", h1, n1, 0.3, 0.0, "MOVE"), Segment("HOLD", n1, total, 0.0, 0.0, "GOAL_HOLD")])
    bg = [0.3, 0.3, 0.3]
    y = render_equation_1(srcs, bg, [cv0, cv1, cv2], total, 4096)
    fr = np.arange(total)
    manual = sum((bg[i] * c.values(fr))[:, None] * s[fr % s.shape[0]].astype(np.float64)
                 for i, (s, c) in enumerate(zip(srcs, [cv0, cv1, cv2]))).astype(np.float32)
    ok("render_equals_equation_1_modulo_playback", np.array_equal(y, manual))
    ref = (bg[0] * srcs[0][np.arange(n1, total) % srcs[0].shape[0]].astype(np.float64)).astype(np.float32)
    ok("goal_hold_equals_scaled_goal_pcm_exactly", np.array_equal(y[n1:], ref))
    ok("goal_hold_curve_is_exact_constant", np.all(cv0.values(np.arange(n1, total)) == 1.0)
       and np.all(cv1.values(np.arange(n1, total)) == 0.0) and np.all(cv2.values(np.arange(n1, total)) == 0.0))
    # audit-2 F1: sync derivation (audit B3 example) and local counter relation
    from latent_space.bank import Bank
    from latent_space.types import UnitContext
    from latent_space.curves import Bump
    t1 = 20 * fs
    tend = 22 * fs
    class _A:  # analyzer stand-in: plan derivation needs no acoustics
        pass
    u = UnitContext(index=0, start=0, end=tend, goal_arrival=t1, starts_from_silence=True, idx=np.arange(10), centers=np.arange(10),
                    seconds=np.arange(10) / fs, o=np.ones(10), phase_names=np.array(["OPEN"] * 10), hold_mask=np.zeros(10, bool),
                    xi_goal=np.zeros((10, 3)), f_mat=None, S=None, chi=None, start_gains=np.zeros(3), fs=fs, M=3, d_xi=3, analyzer=_A(),
                    phase_frames={"INTRO": (0, 0), "OPEN": (0, 10 * fs), "CONTRACT": (10 * fs, t1), "GOAL_HOLD": (t1, tend)})
    base_pts = [(0, 0.0, "MOVE"), (10 * fs, 0.5, "MOVE"), (t1, 0.0, "GOAL_HOLD"), (tend, 0.0, "")]
    n_ok = 0
    for sd in range(5):
        bk = Bank(u, lim, DEFAULTS, np.random.default_rng(sd), None, None)
        out = bk.derived_track_plan(2, base_pts, "sync")
        n_ok += int(out is not None and C.curve_from_waypoints(out).check(fs, lim) == [])
    ok("sync derivation of the audit B3 example is legal (5 seeds)", n_ok == 5, f"{n_ok}/5")
    t1b, tendb = 40 * fs, 42 * fs
    u.goal_arrival, u.end = t1b, tendb
    u.phase_frames = {"INTRO": (0, 0), "OPEN": (0, 20 * fs), "CONTRACT": (20 * fs, t1b), "GOAL_HOLD": (t1b, tendb)}
    base2 = [(0, 0.0, "MOVE"), (10 * fs, 0.6, "MOVE"), (18 * fs, 0.2, "MOVE"), (26 * fs, 0.7, "MOVE"), (t1b, 0.0, "GOAL_HOLD"), (tendb, 0.0, "")]
    n_ok = 0
    for sd in range(5):
        bk = Bank(u, lim, DEFAULTS, np.random.default_rng(sd), None, None)
        out = bk.derived_track_plan(2, base2, "counter")
        n_ok += int(out is not None and C.curve_from_waypoints(out).check(fs, lim) == [])
    ok("counter relation realized on an interior interval (5 seeds)", n_ok == 5, f"{n_ok}/5")
    # composite curve: bump derivatives and checker
    cvq = TrackCurve([Segment("Q5", 0, 10 * fs, 0.0, 0.5, "MOVE")])
    ok("state_at returns the real velocity at the Q5 midpoint", abs(cvq.state_at(5 * fs, fs)[1] - 0.09375) < 1e-9)
    ok("composite checker rejects a too-strong bump", cvq.add_bump(Bump(2 * fs, 7 * fs, 0.5)).check_composite(fs, lim) != [])
    ok("composite checker rejects a sub-minimum reversal", TrackCurve([Segment("HOLD", 0, 10 * fs, 0.3, 0.3, "SCORED_HOLD")])
       .add_bump(Bump(2 * fs, 7 * fs, -0.01)).check_composite(fs, lim) != [])
    REPORT["A"] = {"T_lo_seconds_0_to_0.8": T_lo, "T_hi": T_hi, "db_rate_bound": bound}


# ----------------------------------------------------------------------------- fixtures
def make_fixtures():
    fx = write_fixture_set(os.path.join(DEV, "fx"), fs=22050, N=4, seed=7)
    # alternative set: only material 1 differs (different texture), everything else identical
    alt_dir = os.path.join(DEV, "fx_alt")
    os.makedirs(alt_dir, exist_ok=True)
    alt = {"materials": [], "goal": fx["goal"], "fs": 22050}
    for k, p in enumerate(fx["materials"]):
        q = os.path.join(alt_dir, os.path.basename(p))
        if k == 0:
            x = synth_material(np.random.default_rng(123), 22050, 24.0, "pulse_density")
            write_wav_float32(q, x, 22050)
        else:
            x, _ = read_wav(p)
            write_wav_float32(q, x, 22050)
        alt["materials"].append(q)
    hard = write_fixture_set(os.path.join(DEV, "fx_hard"), fs=22050, N=4, seed=7, hard=True)
    n2 = write_fixture_set(os.path.join(DEV, "fx_n2"), fs=22050, N=2, seed=11)
    n7 = write_fixture_set(os.path.join(DEV, "fx_n7"), fs=22050, N=7, seed=12)
    return fx, alt, hard, n2, n7


def job_for(fixture, mode, name, seed=48291, extra=None):
    cfg_path = write_fixture_config(os.path.join(DEV, f"{name}_{mode}.json"), fixture, mode=mode, seed=seed, extra=extra)
    cfg = load_config(cfg_path, mode)
    return Job(cfg, os.path.join(DEV, "out", f"{name}_{mode}"), config_path=cfg_path, mode_override=mode), cfg


# ----------------------------------------------------------------------------- B
def check_B(modes, fx, alt, n2, n7):
    print("[B] four modes: contractions, re-opening, history and acoustic dependence")
    for mode in modes:
        print(f" mode={mode}")
        rep = {}
        job, cfg = job_for(fx, mode, "full")
        t0 = time.time()
        res = job.run()
        rep["run_seconds"] = time.time() - t0
        rep["status"] = res["status"]
        tr = json.load(open(os.path.join(job.output_dir, "state_trace.json")))
        holds = [p for p in tr.get("form", {}).get("phase_intervals_in_integer_frames", []) if p["name"] == "GOAL_HOLD"]
        reopen_last = tr.get("form", {}).get("phase_intervals_in_integer_frames", [])[-1:]
        ok(f"{mode}: job succeeds", res["status"] in ("VALID_APPROXIMATION", "BEST_EFFORT"), res.get("error"))
        ok(f"{mode}: two contractions and a final re-opening",
           len(holds) >= 2 and reopen_last and reopen_last[0]["name"] == "REOPEN")
        ok(f"{mode}: hard checks all passed", tr.get("hard_checks", {}).get("all_passed", False))
        eff = tr.get("soft_results", {}).get("history_internal_effect_per_unit", [])
        ok(f"{mode}: engine-side history effect > 0 in later units", len(eff) >= 2 and all(e > 0 for e in eff[1:]), eff)
        rep["history_internal_effect_per_unit"] = eff
        rep["history_realized_effect_per_unit"] = tr.get("soft_results", {}).get("history_realized_effect_per_unit")
        rep["normalized_mode_error_per_unit"] = [u["chosen"]["normalized_mode_error"] for u in tr.get("units", [])]
        # --- audit-revision checks: goal exposure, motion distribution, selection, history order
        fs_ = tr["sample_rate"]
        curves_ = [TrackCurve.from_list(lst) for lst in tr["curve_segments_per_track"]]
        ge = tr["resolved_config"]["form"]["goal_exposure"]
        g0 = curves_[0]
        caps_ok = True
        inherit_ok = True
        for p in tr["form"]["phase_intervals_in_integer_frames"]:
            fr = np.arange(p["start_frame"], p["end_frame"])
            if p["name"] in ("INTRO", "OPEN") and len(fr):
                cap = ge["intro_max"] if p["name"] == "INTRO" else ge["open_max"]
                caps_ok &= bool(g0.values(fr).max() <= cap + 1e-9)
            if p["name"] == "REOPEN" and p["start_frame"] > 0:
                v = g0.values(fr[: int(2 * fs_)])
                inherit_ok &= bool(v[0] == 1.0 and np.all(np.diff(v) <= 1e-12))
        ok(f"{mode}: goal stays within intro/open caps and rises only in CONTRACT", caps_ok)
        ok(f"{mode}: REOPEN inherits goal=1 and descends continuously", inherit_ok)
        durs = np.array([(sg["end_frame"] - sg["start_frame"]) / fs_ for lst in tr["curve_segments_per_track"][1:]
                         for sg in lst if sg["type"] == "Q5"])
        ok(f"{mode}: material moves mix short and long durations", len(durs) > 0 and durs.min() < 6.0
           and float(np.mean(durs < 8.0)) >= 0.2, f"n={len(durs)} min={durs.min():.1f}s median={np.median(durs):.1f}s frac<8s={np.mean(durs < 8.0):.2f}")
        # audit-2 F2/F3: fixed-reference improvement (hash unchanged), commits, non-zero-velocity joints
        steps = [st for u in tr["units"] for st in u["steps"]]
        ok(f"{mode}: references frozen during realization (hash recorded, never updated)",
           all(st["reference_hash"] and st["reference_updated_during_realization"] is False for st in steps))
        ok(f"{mode}: joint objective never worsens and improves in some steps against frozen references",
           all(st["final_joint_objective"] <= st["initial_joint_objective"] + 1e-12 for st in steps)
           and sum(1 for st in steps if st["accepted_changes"] > 0) >= max(1, len(steps) // 4),
           f"{sum(1 for st in steps if st['accepted_changes'] > 0)}/{len(steps)} steps improved")
        ok(f"{mode}: commits update the history each step",
           all(u["commits"] == u["history_updates"] and u["commits"] >= 3 for u in tr["units"]), [u["commits"] for u in tr["units"]])
        ok(f"{mode}: at least one internal joint crossed with non-zero velocity",
           tr["hard_checks"].get("nonzero_velocity_joints", 0) >= 1, tr["hard_checks"].get("nonzero_velocity_joints"))
        rep["realization"] = [u["realization_summary"] for u in tr["units"]]
        ev = tr["history"]["events"]
        ok(f"{mode}: history events ordered by time", all(ev[i]["end_frame"] <= ev[i + 1]["end_frame"] for i in range(len(ev) - 1)))
        rc = tr["hard_checks"]["rendered_composition_check"]
        ok(f"{mode}: proxy composition matches rendered PCM (spread windows)", all(c["proxy_vs_rendered_mean_dist2"] < 1e-3 for c in rc),
           [round(c["proxy_vs_rendered_mean_dist2"], 6) for c in rc])
        rep["rendered_composition_check"] = rc
        if mode == "gan":
            dp = tr["mode_trace"]["gan"]["discriminator_parameters"]
            ok("gan: discriminator input standardisation frozen after the first unit",
               all(d.get("feature_scale_source") == "frozen_from_first_unit" for d in dp[1:]))
            upd = [st["mode_observe"] for u in tr["units"] for st in u["steps"] if isinstance(st.get("mode_observe"), dict)]
            ok("gan: adversarial D/G updates recorded at commit observations",
               any(("updates_D" in m or "D_loss_after" in m) for m in upd), f"{len(upd)} observations")
        # --- controlled comparisons on unit 1: same current audio, same phase, same rng
        base_job, _ = job_for(fx, mode, "cmp")
        base_job.load_inputs()
        base_job.setup()
        u0 = base_job.unit_context(base_job.form.units[0])
        chosen0 = base_job.run_unit(u0)          # builds a real history from unit 0
        hist_full = copy.deepcopy(base_job.history)
        hist_empty = type(base_job.history)(base_job.M, base_job.analyzer.d_xi, cfg)
        u1 = base_job.unit_context(base_job.form.units[1])
        rng_state = copy.deepcopy(base_job.rng.bit_generator.state)

        def ideal_stats(job_, unit_, hist_):
            m = copy.deepcopy(job_.mode)
            m.rng = np.random.default_rng(0)
            m.rng.bit_generator.state = copy.deepcopy(rng_state)
            m.extra_candidate_evaluations = 0
            m.begin_unit(unit_, hist_)
            sig = np.asarray(m.signature(), dtype=np.float64)
            tg = m.propose(unit_, hist_, 0, 3)
            mean_xi = np.mean([t.xi_hat[unit_.free_mask].mean(axis=0) for t in tg], axis=0)
            return sig, mean_xi

        sig_h, xi_h = ideal_stats(base_job, u1, hist_full)
        sig_e, xi_e = ideal_stats(base_job, u1, hist_empty)
        d_sig = float(np.linalg.norm(sig_h - sig_e)) if sig_h.shape == sig_e.shape else float("nan")
        d_xi = float(np.abs(xi_h - xi_e).mean())
        ok(f"{mode}: history-only change alters internal state", d_sig > 0, f"|dsig|={d_sig:.4g}, |d ideal xi|={d_xi:.4g}")
        rep["history_only"] = {"signature_diff": d_sig, "ideal_xi_diff": d_xi}
        # one material's current acoustics changed (material 1 replaced), same history, same rng
        alt_job, _ = job_for(alt, mode, "cmpalt")
        alt_job.load_inputs()
        alt_job.setup()
        u1a = alt_job.unit_context(alt_job.form.units[1])
        sig_a, xi_a = ideal_stats(alt_job, u1a, hist_full)
        d_sig_a = float(np.linalg.norm(sig_h - sig_a)) if sig_h.shape == sig_a.shape else float("nan")
        d_xi_a = float(np.abs(xi_h - xi_a).mean())
        ok(f"{mode}: one-material acoustic change alters ideal composition", d_xi_a > 0 or d_sig_a > 0,
           f"|dsig|={d_sig_a:.4g}, |d ideal xi|={d_xi_a:.4g}")
        rep["material_only"] = {"signature_diff": d_sig_a, "ideal_xi_diff": d_xi_a}
        # variable N: dimension construction + CLI load + short candidate generation
        for nm, fxn in (("N2", n2), ("N7", n7)):
            jn, cfgn = job_for(fxn, mode, nm)
            try:
                jn.load_inputs()
                jn.setup()
                un = jn.unit_context(jn.form.units[0])
                jn.mode.extra_candidate_evaluations = 0
                jn.mode.begin_unit(un, jn.history)
                from latent_space.bank import Bank
                bk = Bank(un, jn.lim, cfgn, jn.rng, jn.objective, jn.history)
                cand = bk.random_candidate(jn.mode.hints(un, jn.history))
                tg = jn.mode.propose(un, jn.history, 0, 2)
                good = cand is not None and all(t.xi_hat.shape == (un.J, un.d_xi) for t in tg)
                ok(f"{mode}: {nm} dimensions/candidates/proposals", good, f"M={jn.M} d_xi={un.d_xi} J={un.J}")
            except Exception as e:  # noqa: BLE001
                ok(f"{mode}: {nm} dimensions/candidates/proposals", False, repr(e))
        REPORT["B"][mode] = rep


# ----------------------------------------------------------------------------- C
def check_C(modes, fx, hard):
    print("[C] artifacts and transparency")
    for mode in modes:
        out = os.path.join(DEV, "out", f"full_{mode}")
        files = ["result.wav", "gain_curves.csv", "state_trace.json"]
        ok(f"{mode}: three artifacts exist", all(os.path.exists(os.path.join(out, f)) for f in files))
        tr = json.load(open(os.path.join(out, "state_trace.json")))
        need = ["run_status", "hard_checks", "soft_results", "curve_segments_per_track", "mode_trace", "history",
                "implementation_gaps", "calibration_status", "resolved_config", "base_gains", "input_paths_and_hashes"]
        ok(f"{mode}: trace has required sections", all(k in tr for k in need), [k for k in need if k not in tr])
        # re-render from the analytic curves in the trace and compare with result.wav bit-exactly
        job, cfg = job_for(fx, mode, "rerender")
        job.load_inputs()
        curves = [TrackCurve.from_list(lst) for lst in tr["curve_segments_per_track"]]
        y2 = render_equation_1(job.sources, tr["base_gains"], curves, tr["form"]["total_frames"], 8192)
        y1, _ = read_wav(os.path.join(out, "result.wav"))
        ok(f"{mode}: re-render from trace curves is bit-exact", np.array_equal(y1, y2))
        kinds = {s["type"] for lst in tr["curve_segments_per_track"] for s in lst}
        ok(f"{mode}: only Q5/HOLD segments plus local bumps", kinds <= {"Q5", "HOLD", "BUMPS"}, kinds)
        REPORT["C"][mode] = {"run_status": tr["run_status"], "warnings": tr.get("warnings", [])}
    # configuration aliases are applied and unknown keys are rejected (audit §7)
    alias_cfg = os.path.join(DEV, "alias.json")
    base = json.load(open(os.path.join(DEV, f"full_{modes[0]}.json")))
    base.pop("form", None)
    base["schedule"] = {"cycles": 2, "open_seconds": 9}
    base["motion"] = {"relative_rate_max": 11.0}
    base["search"] = {"max_rounds": 2, "candidate_count": 6, "acceptance_margin": 0.0}
    json.dump(base, open(alias_cfg, "w"))
    cfg_a = load_config(alias_cfg, modes[0])
    ok("config aliases map to canonical keys", cfg_a["form"]["open_seconds"] == 9 and cfg_a["motion"]["regularized_db_rate_max"] == 11.0
       and cfg_a["search"]["max_search_rounds"] == 2 and cfg_a["search"]["candidate_bank_target"] == 6, cfg_a["config_aliases_applied"])
    bad_cfg = os.path.join(DEV, "bad.json")
    base2 = json.load(open(os.path.join(DEV, f"full_{modes[0]}.json")))
    base2["search"] = {"candidate_pool": 5}
    json.dump(base2, open(bad_cfg, "w"))
    try:
        load_config(bad_cfg, modes[0])
        ok("unknown config key is rejected", False)
    except ValueError as e:
        ok("unknown config key is rejected", "candidate_pool" in str(e))
    # hires extension (user-authorised): switches, position jumps, master chain; exact goal on the raw render
    hcfg = json.load(open(os.path.join(DEV, f"full_{modes[0]}.json")))
    hcfg["hires"] = {"enabled": True}
    hcfg["render"] = {"master": {"enabled": True}}
    hpath = os.path.join(DEV, "hires.json")
    json.dump(hcfg, open(hpath, "w"))
    jh = Job(load_config(hpath, modes[0]), os.path.join(DEV, "out", f"hires_{modes[0]}"), config_path=hpath, mode_override=modes[0])
    t0 = time.time()
    rh = jh.run()
    th = json.load(open(os.path.join(jh.output_dir, "state_trace.json")))
    hc = th.get("hard_checks", {})
    ok(f"{modes[0]} hires: legal output with switches and position jumps",
       rh["status"] in ("BEST_EFFORT", "VALID_APPROXIMATION") and hc.get("all_passed") and hc.get("position_jumps", 0) > 0
       and hc.get("switches", 0) > 0, f"{rh['status']} jumps={hc.get('position_jumps')} switches={hc.get('switches')} {time.time() - t0:.0f}s")
    ok(f"{modes[0]} hires: exact goal on the raw render before the master chain",
       hc.get("exact_goals_rendered") and hc.get("goal_hold_max_abs_deviation") == 0.0 and "master_chain" in hc)
    curves_h = [TrackCurve.from_list(lst) for lst in th["curve_segments_per_track"]]
    ok(f"{modes[0]} hires: position maps restored from the trace", any(len(cv.clips) > 0 for cv in curves_h[1:]))
    REPORT["C"]["hires"] = {"status": rh["status"], "jumps": hc.get("position_jumps"), "switches": hc.get("switches")}
    # poor-approximation fixture: must still output BEST_EFFORT/VALID within bounded time
    mode = modes[0]
    job, cfg = job_for(hard, mode, "hard")
    t0 = time.time()
    res = job.run()
    dt = time.time() - t0
    ok(f"{mode}: hard fixture yields legal best-effort output in bounded time", res["status"] in ("BEST_EFFORT", "VALID_APPROXIMATION") and dt < 120,
       f"{res['status']} in {dt:.1f}s")
    REPORT["C"]["hard_fixture"] = {"mode": mode, "status": res["status"], "seconds": dt}


def main():
    modes = sys.argv[1:] or ["diffusion", "vae", "transformer", "gan"]
    avail = []
    for m in modes:
        try:
            __import__(f"latent_space.modes.{m}")
            avail.append(m)
        except Exception as e:  # noqa: BLE001
            print(f"  mode {m} not importable: {e}")
    check_A()
    fx, alt, hard, n2, n7 = make_fixtures()
    if avail:
        check_B(avail, fx, alt, n2, n7)
        check_C(avail, fx, hard)
    REPORT["modes_checked"] = avail
    REPORT["modes_missing"] = [m for m in modes if m not in avail]
    with open(os.path.join(DEV, "smoke_report.json"), "w") as f:
        json.dump(REPORT, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("failures:", REPORT["failures"] or "none", "| missing modes:", REPORT["modes_missing"] or "none")
    return 0 if not REPORT["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())

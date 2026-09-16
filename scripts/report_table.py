"""Print the completion-report table (audit-2 §F) from state_trace.json files.
  python3 scripts/report_table.py output            # one directory per mode
"""
import json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "output"
for mode in ("diffusion", "vae", "transformer", "gan"):
    p = os.path.join(root, mode, "state_trace.json")
    if not os.path.exists(p):
        print(f"== {mode}: (no trace)")
        continue
    t = json.load(open(p))
    fs = t.get("sample_rate")
    print(f"== {mode}: {t['run_status']}  hard_checks.all_passed={t.get('hard_checks', {}).get('all_passed')}  elapsed={t['elapsed_seconds']:.0f}s")
    if t.get("error"):
        print("   error:", t["error"][:200])
        continue
    hc = t["hard_checks"]
    for u in t["units"]:
        rs = u["realization_summary"]
        steps = u["steps"]
        hashes_ok = all(st["reference_updated_during_realization"] is False for st in steps)
        pen = np.mean([st["mode_term_breakdown"]["final"]["penalty"] for st in steps]) if steps else 0.0
        print(f"   unit{u['unit']}: J {rs['mean_initial_joint_objective']:.4f}->{rs['mean_final_joint_objective']:.4f} | fit {rs['mean_initial_fixed_target_error']:.4f}->{rs['mean_final_fixed_target_error']:.4f} | mode penalty {pen:.4f} | full-grid fixed residual {rs['full_grid_fixed_target_residual']:.4f} | commits {u['commits']} history updates {u['history_updates']} | steps improved {rs['steps_with_accepted_changes']}/{u['commits']} | ref frozen {hashes_ok} | hist internal {u['history_internal_effect']:.3f} ideal {u['history_ideal_effect']:.4f} realized {u['history_realized_effect']:.4f}")
    rc = hc.get("rendered_composition_check", [])
    print("   rendered residual (proxy|target vs rendered):", [(c["unit"], round(c["proxy_vs_rendered_mean_dist2"], 5), round(c["fixed_target_vs_rendered_mean_dist2"], 4) if c["fixed_target_vs_rendered_mean_dist2"] is not None else None) for c in rc])
    print(f"   non-zero-velocity joints {hc.get('nonzero_velocity_joints')} | bumps {hc.get('bumps_total')} | violations {len(hc.get('violations', []))} | goal-hold deviation {hc.get('goal_hold_max_abs_deviation')}")
    rows = np.loadtxt(os.path.join(root, mode, "gain_curves.csv"), delimiter=",", skiprows=1)
    tt, g0 = rows[:, 0], rows[:, 1]
    rise = tt[g0 > 1.0][0] if (g0 > 1.0).any() else None
    full = tt[g0 >= 100.0][0] if (g0 >= 100.0).any() else None
    holds = [(ph["start_frame"] / fs, ph["end_frame"] / fs) for ph in t["form"]["phase_intervals_in_integer_frames"] if ph["name"] == "GOAL_HOLD"]
    print(f"   goal rise starts {rise}s | first full goal {full}s | goal holds {[(round(a,1), round(b,1)) for a,b in holds]}")
    segs = t["curve_segments_per_track"]
    durs = np.array([(s["end_frame"] - s["start_frame"]) / fs for tr in segs[1:] for s in tr if s.get("type") == "Q5"])
    amps = np.array([abs(s["end_gain"] - s["start_gain"]) * 100 for tr in segs[1:] for s in tr if s.get("type") == "Q5"])
    bumps = [b["amplitude"] * 100 for tr in segs for s in tr if s.get("type") == "BUMPS" for b in s["bumps"]]
    print(f"   material base moves: n={len(durs)} duration median {np.median(durs):.1f}s (min {durs.min():.1f}, max {durs.max():.1f}, <8s {np.mean(durs<8):.2f}) amplitude median {np.median(amps):.1f} pts | bump amplitudes: n={len(bumps)} median |a| {np.median(np.abs(bumps)):.1f} pts max {np.max(np.abs(bumps)) if bumps else 0:.1f}")
    print("   source loop counts:", {k: round(v, 2) for k, v in t.get("source_loop_counts", {}).items()})
    print("   warnings:", [w[:100] for w in t["warnings"] if "length" not in w])

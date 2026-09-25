"""Seed comparison of 1-cycle runs (presence version): who sounds, goal entry, loudness, usage, cast changes.

usage: python3 dev/seed_compare.py [--series OUT.json] RUN_DIR [RUN_DIR ...]
Reads gain_curves.csv, result.wav and state_trace.json of every run directory. Times are piece times in seconds
(the slow form: INTRO 0-12, CONTRACT 12-192, GOAL_HOLD 192-197, REOPEN 197-227).
"""
import csv, json, sys
import numpy as np
sys.path.insert(0, ".")
from latent_space.audio_io import read_wav

BINS = [(20, 60), (60, 90), (90, 120), (120, 140), (140, 150), (150, 160), (160, 180), (180, 192)]
EPS = 1e-6


def mmss(t):
    return "—" if t is None else f"{int(t // 60)}:{t % 60:04.1f}"


def group(n):
    for g in ("lumin", "papillon", "prism"):
        if n.startswith(g):
            return {"lumin": "luminasity"}.get(g, g)
    return "other"


def first_cross(t, x, thr, t0=0.0, below=False):
    ok = (t >= t0) & ((x < thr) if below else (x >= thr))
    return float(t[np.argmax(ok)]) if ok.any() else None


def running_mean(x, n):
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def analyse(d):
    rows = list(csv.reader(open(d + "/gain_curves.csv")))
    names = rows[0][2:]
    g = np.array([[float(v) for v in r] for r in rows[1:]])
    t, goal, lv = g[:, 0], g[:, 1] / 100.0, g[:, 2:]          # goal 0..1, material levels in percent
    dt = t[1] - t[0]
    on = lv > EPS
    cnt = on.sum(axis=1).astype(float)
    top = np.sort(lv, axis=1)[:, ::-1][:, :3]

    wav, info = read_wav(d + "/result.wav")
    fs = int(getattr(info, "sample_rate", 44100))
    p = (wav.astype(np.float64) ** 2).mean(axis=1)

    def rms_db(a, b):
        s = p[int(a * fs):int(b * fs)]
        return float(10 * np.log10(max(s.mean(), 1e-12)))

    tr = json.load(open(d + "/state_trace.json"))
    hc = tr["hard_checks"]
    out = {"dir": d, "seed": tr["seed"], "status": tr["run_status"], "elapsed_min": tr["elapsed_seconds"] / 60.0,
           "hard_all_passed": hc.get("all_passed"), "violations": len(hc.get("violations") or []),
           "goal_hold_max_abs_deviation": hc.get("goal_hold_max_abs_deviation"),
           "exact_goals_rendered": hc.get("exact_goals_rendered")}
    pc = hc.get("polyphony_cap") or {}
    out["cap"] = {k: pc.get(k) for k in ("max_sounding_outside_ramps", "shortest_sounding_seconds", "passed")
                  if k in pc}

    # interval table
    tab = []
    for a, b in BINS:
        m = (t >= a) & (t < b)
        tab.append({"bin": f"{mmss(a)}-{mmss(b)}", "sounding": float(cnt[m].mean()),
                    "top3": [float(x) for x in top[m].mean(axis=0)],
                    "goal_min": float(goal[m].min()), "goal_max": float(goal[m].max()), "rms_db": rms_db(a, b)})
    out["bins"] = tab

    # goal entry (law) and final state
    out["goal_cross"] = {str(th): first_cross(t, goal, th, 12.0) for th in (0.01, 0.1, 0.5, 0.9)}
    out["goal_at_179"] = float(goal[int(round(179.0 / dt))])
    out["goal_at_180"] = float(goal[int(round(180.0 / dt))])
    # goal level reached before the designed entry window (2:20) - an early "leak" of the goal
    out["goal_max_before_140"] = float(goal[t < 140.0].max())

    # quiet holes: longest run of 1-s RMS below -40 dBFS inside 0:20-3:12
    best, cur, start, n40 = (0, None), 0, None, 0
    for a in range(20, 192):
        q = rms_db(a, a + 1.0) < -40.0
        n40 += int(q)
        cur = cur + 1 if q else 0
        if q and cur == 1:
            start = a
        if cur > best[0]:
            best = (cur, start)
    out["quiet_longest_below_40"] = {"seconds": best[0], "from": best[1], "total_seconds": n40}

    # materials leaving (5 s running mean of the sounding count inside CONTRACT)
    rc = running_mean(cnt, int(round(5.0 / dt)))
    inside = (t >= 20) & (t < 192)
    rc_in = np.where(inside, rc, np.nan)
    out["count_below"] = {str(th): first_cross(t, np.nan_to_num(rc_in, nan=9.0), th, 20.0, below=True)
                          for th in (2.5, 1.5, 0.5)}
    # the last material(s) before the goal takes over
    idx = np.where((t < 192) & on.any(axis=1))[0]
    last_i = int(idx[-1]) if len(idx) else None
    out["last_material_silent_at"] = float(t[last_i + 1]) if last_i is not None else None
    if last_i is not None:
        seg = (t > t[last_i] - 20) & (t <= t[last_i])
        secs = on[seg].sum(axis=0) * dt
        out["last_20s_materials"] = {names[k]: float(secs[k]) for k in np.argsort(-secs) if secs[k] > 0}
    g01 = out["goal_cross"]["0.01"]
    out["sounding_at_goal_entry"] = ([names[k] for k in np.where(on[int(round(g01 / dt))])[0]]
                                     if g01 is not None else [])

    # usage in goal-free CONTRACT time
    free = (goal < 0.01) & (t >= 12) & (t < 192)
    secs = on[free].sum(axis=0) * dt
    tot = secs.sum()
    grp = {}
    for k, n in enumerate(names):
        grp[group(n)] = grp.get(group(n), 0.0) + secs[k]
    out["usage_group_share"] = {k: float(v / tot) for k, v in grp.items()}
    srt = np.sort(secs)[::-1]
    out["usage_top3_share"] = float(srt[:3].sum() / tot)
    out["usage_unused"] = int((secs == 0).sum())
    out["usage_top3"] = [(names[k], float(secs[k])) for k in np.argsort(-secs)[:3]]

    # cast changes (same rule as scripts/cast_changes.py: 0.5 s commit ends, goal-free time)
    step = max(1, int(round(0.5 / dt)))
    ii = np.arange(step - 1, len(t), step)
    fr = goal[ii] < 1e-4
    sets = [frozenset(np.where(on[i])[0]) for i in ii]
    ch = sum(1 for a, b, f in zip(sets[:-1], sets[1:], fr[1:]) if f and a != b)
    minutes = fr.sum() * 0.5 / 60.0
    out["cast_changes_per_min"] = ch / max(minutes, 1e-9)

    # diffusion law summaries
    md = tr["mode_trace"]["diffusion"]
    hp = md["hold_plan_chain_per_unit"][0]
    out["law"] = {k: hp.get(k) for k in ("swaps_per_minute", "mean_sounding_materials", "energy_fidelity",
                                          "lower_than_stay_rate")}
    ratios = []
    for s in tr["units"][0]["steps"]:
        fe = (s.get("mode_observe") or {}).get("field_energy_vs_random_fragments") or {}
        if fe.get("available") and 12 <= s["t_seconds"] < 120:
            ratios.append(fe["ratio_committed_over_random"])
    out["law"]["field_ratio_12_120"] = float(np.mean(ratios)) if ratios else None

    # series for plotting (0.5 s)
    ser_t = t[ii]
    rms_1s = [rms_db(a, a + 1.0) for a in np.arange(0, int(t[-1]))]
    out["series"] = {"t": [float(x) for x in ser_t], "count5": [float(x) for x in rc[ii]],
                     "count": [float(x) for x in cnt[ii]], "goal": [float(x) for x in goal[ii]],
                     "rms_t": [float(a) + 0.5 for a in np.arange(0, int(t[-1]))], "rms_db": rms_1s}
    return out


def main(argv):
    series_path = None
    if argv and argv[0] == "--series":
        series_path, argv = argv[1], argv[2:]
    res = [analyse(d.rstrip("/")) for d in argv]
    for r in res:
        print(f"\n=== {r['dir']}  seed {r['seed']}  {r['status']}  hard {r['hard_all_passed']} "
              f"(violations {r['violations']})  goal-hold dev {r['goal_hold_max_abs_deviation']}  "
              f"{r['elapsed_min']:.1f} min  cap {r['cap']}")
        print(f"{'interval':16s} {'sounding':>8s} {'top3 levels':>18s} {'goal':>13s} {'RMS dBFS':>9s}")
        for b in r["bins"]:
            print(f"{b['bin']:16s} {b['sounding']:8.2f} {'/'.join(f'{x:.0f}' for x in b['top3']):>18s} "
                  f"{b['goal_min']:6.3f}-{b['goal_max']:5.3f} {b['rms_db']:9.1f}")
        gc = r["goal_cross"]
        print("goal >= 0.01/0.1/0.5/0.9 at", " / ".join(mmss(gc[k]) for k in ("0.01", "0.1", "0.5", "0.9")),
              f"| at 2:59 {r['goal_at_179']:.3f} | max before 2:20 {r['goal_max_before_140']:.3f}")
        ql = r["quiet_longest_below_40"]
        print(f"quiet: longest run < -40 dBFS {ql['seconds']} s from {mmss(ql['from'])}, total {ql['total_seconds']} s")
        cb = r["count_below"]
        print("count (5 s mean) < 2.5/1.5/0.5 at", " / ".join(mmss(cb[k]) for k in ("2.5", "1.5", "0.5")),
              f"| last material silent at {mmss(r['last_material_silent_at'])}")
        print("sounding at goal entry:", r["sounding_at_goal_entry"])
        print("last 20 s of materials:", {k[:22]: round(v, 1) for k, v in (r.get("last_20s_materials") or {}).items()})
        print("usage shares:", {k: round(v, 2) for k, v in r["usage_group_share"].items()},
              f"top3 {r['usage_top3_share']:.2f} unused {r['usage_unused']}",
              [(n[:18], round(s, 1)) for n, s in r["usage_top3"]])
        print(f"cast changes/min {r['cast_changes_per_min']:.1f} | law {r['law']}")
    if series_path:
        json.dump([{k: r[k] for k in ("dir", "seed", "series", "goal_cross", "count_below")} for r in res],
                  open(series_path, "w"))
        print("\nseries ->", series_path)


if __name__ == "__main__":
    main(sys.argv[1:])

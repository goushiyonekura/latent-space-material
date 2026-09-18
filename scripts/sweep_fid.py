"""Hold-length and period sweep for the hold / fidelity versions (2026-09-17).

The old period sweep picked the period with the lowest residual; with realizable held ideals the
residual is ~0 for every setting, so each mode is judged by its OWN law -> sound figure:

  diffusion    committed field energy / field energy of random fragment compositions   (lower is better)
  vae          cosine between the requested latent move of a hold and the committed one   (higher)
  transformer  attention -> contribution correlation (law 3) or attention alignment       (higher)
  gan          D(real recordings) - D(published plan)                                     (lower)

next to the common figures (net achieved, intervention vs flutter, distinct 2-s cells per jump).

  python3 scripts/sweep_fid.py vae gan --holds=1,2,4                 # hold lengths at the config's period
  python3 scripts/sweep_fid.py vae gan --opens=90,150,210 --holds=2   # OPEN lengths (+30 s CONTRACT = the cycle)
  options: --base=project.fid (uses <base>.<mode>.json)  --out=dev/sweep_fid  --jobs=4  --keep-wav
"""
import json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import closeness  # noqa: E402

argv = sys.argv[1:]
opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
flags = {a[2:] for a in argv if a.startswith("--") and "=" not in a}
modes = [a for a in argv if not a.startswith("--")] or ["diffusion", "vae", "transformer", "gan"]
BASE = opts.get("base", "project.fid")
OUT = opts.get("out", "dev/sweep_fid")
JOBS = int(opts.get("jobs", 4))
holds = [float(x) for x in opts["holds"].split(",")] if "holds" in opts else [None]
opens = [float(x) for x in opts["opens"].split(",")] if "opens" in opts else [None]
os.makedirs(OUT, exist_ok=True)


def tag_of(mode, hold, op, cfg):
    h = cfg["hires"]["reference_hold_seconds"] if hold is None else hold
    o = cfg["form"]["open_seconds"] if op is None else op
    return f"{mode}_h{h:g}_o{o:g}"


def wmean(vals, weights):
    v = [(x, w) for x, w in zip(vals, weights) if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(sum(x * w for x, w in v) / sum(w for _, w in v)) if v else float("nan")


def mode_figure(mode, t):
    """(name, value, better) of the mode's own law -> sound figure, holds-weighted over the units."""
    mt = t["mode_trace"][mode]
    w = [max(1, u.get("commits", 1)) for u in t["units"]]
    try:
        if mode == "diffusion":
            fs_ = mt["fragment_statistics_per_unit"]
            return ("committed/random field energy", wmean([f["committed_field_energy_mean"] / f["random_fragment_field_energy_mean"] for f in fs_], w), "lower")
        if mode == "vae":
            return ("latent tracking cos", wmean([u["latent_tracking"]["cosine_requested_vs_committed_dz_mean"] for u in mt["unit_statistics"]], w), "higher")
        if mode == "transformer":
            hm = [u["statistics"]["hold_mode"] for u in mt["units"]]
            key = next((k for k in ("attention_contribution_corr", "mean_attention_contribution_corr") if k in hm[0]), None)
            if key is not None and not isinstance(hm[0][key], dict):
                return ("attention->contribution corr", wmean([h[key] for h in hm], w), "higher")
            return ("attention alignment", wmean([h["mean_attention_alignment_of_chosen_step"] for h in hm], w), "higher")
        if mode == "gan":
            hs = [s["hold"] for s in mt["statistics"]]
            return ("D(real) - D(published plan)", wmean([h["D_real_windows_mean"] - h["D_published_plan_mean"] for h in hs], w), "lower")
    except Exception as e:  # noqa: BLE001
        return (f"n/a ({e!r})", float("nan"), "higher")
    return ("n/a", float("nan"), "higher")


def distinct_ratio(t):
    fs_ = t["sample_rate"]
    tot = dis = 0
    for tr in t["curve_segments_per_track"][1:]:
        pos = [c["src_start"] / fs_ for s in tr if s.get("type") == "CLIPS" for c in s["clips"] if c["out_start"] > 0]
        tot += len(pos)
        dis += len({int(round(p / 2.0)) for p in pos})
    return dis / tot if tot else float("nan")


runs = []
for mode in modes:
    base = json.load(open(f"{BASE}.{mode}.json"))
    for hold in holds:
        for op in opens:
            cfg = json.loads(json.dumps(base))
            if hold is not None:
                cfg["hires"]["reference_hold_seconds"] = hold
            if op is not None:
                cfg["form"]["open_seconds"] = op
            tag = tag_of(mode, hold, op, cfg)
            cpath = os.path.join(OUT, f"config_{tag}.json")
            json.dump(cfg, open(cpath, "w"), indent=1)
            runs.append({"mode": mode, "tag": tag, "config": cpath, "dir": os.path.join(OUT, tag),
                         "hold": cfg["hires"]["reference_hold_seconds"], "open": cfg["form"]["open_seconds"]})

pending = [r for r in runs if not os.path.exists(os.path.join(r["dir"], r["mode"], "state_trace.json"))]
active = []
while pending or active:
    while pending and len(active) < JOBS:
        r = pending.pop(0)
        os.makedirs(r["dir"], exist_ok=True)
        p = subprocess.Popen([sys.executable, "-m", "latent_space", "generate", "--config", r["config"], "--mode", r["mode"],
                              "--output", os.path.join(r["dir"], r["mode"])], stdout=open(os.path.join(r["dir"], "run.log"), "w"),
                             stderr=subprocess.STDOUT)
        active.append((r, p))
    for r, p in list(active):
        if p.poll() is not None:
            active.remove((r, p))
            wav = os.path.join(r["dir"], r["mode"], "result.wav")
            if "keep-wav" not in flags and os.path.exists(wav):
                os.remove(wav)
            print("finished", r["tag"], flush=True)
    if active:
        try:
            active[0][1].wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

table = []
for r in runs:
    tp = os.path.join(r["dir"], r["mode"], "state_trace.json")
    if not os.path.exists(tp):
        table.append({**r, "status": "no trace"})
        continue
    t = json.load(open(tp))
    name, val, better = mode_figure(r["mode"], t)
    res = closeness.diag(r["dir"], r["mode"], n_random=60)
    n = sum(x["rows"] for x in res) or 1
    avg = lambda k: float(sum(x[k] * x["rows"] for x in res if not np.isnan(x[k])) / max(1, sum(x["rows"] for x in res if not np.isnan(x[k]))))  # noqa: E731
    hc = t["hard_checks"]
    table.append({**r, "status": t["run_status"], "hard": bool(hc.get("all_passed")), "exact_goal": bool(hc.get("exact_goals_rendered")),
                  "elapsed": t["elapsed_seconds"], "total_seconds": t["form"]["total_seconds"], "figure": name, "value": val, "better": better,
                  "achieved_net": avg("achieved_net"), "intervention": avg("intervention"), "flutter": avg("flutter"),
                  "gauge": 100.0 * float(np.sqrt(avg("now") / avg("random"))), "jumps": hc.get("position_jumps"),
                  "switches": hc.get("switches"), "distinct_cells_per_jump": distinct_ratio(t)})
json.dump(table, open(os.path.join(OUT, f"table_{'_'.join(modes)}_{opts.get('holds', 'h')}_{opts.get('opens', 'o')}.json".replace(",", "-")), "w"), indent=1)
print(f"\n{'mode':11s} hold  open | {'mode figure':34s} value  | net-achieved intervention/flutter  gauge | distinct/jump jumps switches  elapsed")
for r in table:
    if r.get("status") == "no trace":
        print(f"{r['mode']:11s} {r['hold']:4g} {r['open']:5g} | no trace")
        continue
    print(f"{r['mode']:11s} {r['hold']:4g} {r['open']:5g} | {r['figure'][:34]:34s} {r['value']:+.3f} | {r['achieved_net']:+.2f}        "
          f"{r['intervention']:.3f}/{r['flutter']:.3f} = {r['intervention'] / r['flutter']:4.1f}   {r['gauge']:5.1f} | "
          f"{r['distinct_cells_per_jump']:.2f}          {r['jumps']:5d} {r['switches']:6d}   {r['elapsed']:5.0f}s  {'' if r['hard'] and r['exact_goal'] else 'CHECK FAILED'}")

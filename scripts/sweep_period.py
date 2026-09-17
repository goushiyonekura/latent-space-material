"""Period sweep for the hires extension (user request 2026-09-16): for each mode run the
OPEN+CONTRACT cycle at 120 / 180 / 240 s (CONTRACT fixed at 30 s) and adopt the length with the
lowest mean full-grid fixed-reference residual, i.e. the period at which the realized composition
follows that mode's ideal best.  Writes project.hires.<mode>.json and copies the adopted run to
output_hires/<mode>/.

  python3 scripts/sweep_period.py [mode ...]          # default: all four, one process per mode
"""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
modes = args or ["diffusion", "vae", "transformer", "gan"]
OPENS = [90, 150, 210]          # + contract 30 -> cycle 120 / 180 / 240 s
BASE = opts.get("base", "project.hires.json")
OUT = opts.get("out", "output_hires")
TAG = opts.get("tag", "hires")
base = json.load(open(BASE))
base["materials"] = [os.path.abspath(m) for m in base["materials"]]
base["goal"] = os.path.abspath(base["goal"])
report = {}
for mode in modes:
    results = []
    for op in OPENS:
        cfg = json.loads(json.dumps(base))
        cfg["mode"] = mode
        cfg["form"]["open_seconds"] = op
        cpath = f"dev/sweep_{TAG}/config_{mode}_{op}.json"
        os.makedirs(f"dev/sweep_{TAG}", exist_ok=True)
        json.dump(cfg, open(cpath, "w"), indent=1)
        out = f"dev/sweep_{TAG}/{mode}_{op}"
        subprocess.run([sys.executable, "-m", "latent_space", "generate", "--config", cpath, "--mode", mode, "--output", out],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tp = os.path.join(out, "state_trace.json")
        if not os.path.exists(tp):
            results.append({"open": op, "status": "no trace"})
            continue
        t = json.load(open(tp))
        us = t.get("units", [])
        resid = [u["realization_summary"]["full_grid_fixed_target_residual"] for u in us]
        fit1 = [u["realization_summary"]["mean_final_fixed_target_error"] for u in us]
        results.append({"open": op, "cycle": op + 30, "status": t["run_status"],
                        "mean_full_grid_residual": sum(resid) / len(resid) if resid else None,
                        "mean_final_fit": sum(fit1) / len(fit1) if fit1 else None,
                        "elapsed": t["elapsed_seconds"], "jumps": t["hard_checks"].get("position_jumps"),
                        "switches": t["hard_checks"].get("switches")})
        print(mode, results[-1], flush=True)
    ok = [r for r in results if r.get("mean_full_grid_residual") is not None]
    if ok:
        best = min(ok, key=lambda r: r["mean_full_grid_residual"])
        cfg = json.loads(json.dumps(base))
        cfg["mode"] = mode
        cfg["form"]["open_seconds"] = best["open"]
        json.dump(cfg, open(f"project.{TAG}.{mode}.json", "w"), indent=2, ensure_ascii=False)
        dst = f"{OUT}/{mode}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(f"dev/sweep_{TAG}/{mode}_{best['open']}", dst)
        report[mode] = {"adopted_open_seconds": best["open"], "adopted_cycle_seconds": best["cycle"], "sweep": results}
    else:
        report[mode] = {"adopted_open_seconds": None, "sweep": results}
    json.dump(report, open(f"dev/sweep_{TAG}/report_{'_'.join(modes)}.json", "w"), indent=1)
print(json.dumps(report, indent=1))

"""Compact 1-s series (0:20-3:15) for the seed-comparison chart.

usage: python3 dev/seed_widget_data.py NEW_SERIES.json [REF_SERIES.json]
NEW runs are drawn as lines; REF runs (optional) as a min-max envelope.
"""
import json, sys
import numpy as np
T0, T1 = 20, 196
TT = list(range(T0, T1))


def per_second(r):
    s = r["series"]
    t = np.array(s["t"]); c5 = np.array(s["count5"]); g = np.array(s["goal"])
    rt = np.array(s["rms_t"]); rd = np.array(s["rms_db"])
    rs = 10 * np.log10(np.maximum(np.convolve(10 ** (rd / 10.0), np.ones(5) / 5, mode="same"), 1e-12))   # 5 s power mean
    idx = [int(np.argmin(np.abs(t - (x + 0.5)))) for x in TT]
    ridx = [int(np.argmin(np.abs(rt - (x + 0.5)))) for x in TT]
    return {"c": c5[idx], "g": g[idx], "r": rs[ridx]}


out = {"t": TT, "new": {}}
for r in json.load(open(sys.argv[1])):
    p = per_second(r)
    out["new"][str(r["seed"])] = {"c": [round(float(x), 1) for x in p["c"]], "g": [round(float(x), 2) for x in p["g"]],
                                  "r": [int(round(float(x))) for x in p["r"]]}
if len(sys.argv) > 2:
    ps = [per_second(r) for r in json.load(open(sys.argv[2]))]
    out["ref"] = {}
    for k, nd in (("c", 1), ("g", 2), ("r", 0)):
        a = np.stack([p[k] for p in ps])
        lo, hi = a.min(axis=0), a.max(axis=0)
        f = (lambda x: int(round(float(x)))) if nd == 0 else (lambda x, n=nd: round(float(x), n))
        out["ref"][k] = [[f(x) for x in lo], [f(x) for x in hi]]
print(json.dumps(out, separators=(",", ":")))

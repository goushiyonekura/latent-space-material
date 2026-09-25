"""Per material/part: fragments in time order, their cheapest option (output extent) and what the placement chose."""
import json, os, pickle, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, ".")
from scoremap import build, extract, layout, mxl

out_dir = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
trace = json.load(open(os.path.join(out_dir, "state_trace.json")))
names = list(trace["source_order"])
scores, cps, acts, aligns = {}, {}, {}, {}
frags = defaultdict(list)
for (lane, t0, t1, s0, s1) in build.map_bars(trace):
    mat = names[lane]
    if only and not mat.startswith(only):
        continue
    if lane not in scores:
        scores[lane] = mxl.parse(os.path.join("materials/scores-diffusion", mat + ".musicxml"))
        cps[lane] = {p.pid: extract.clean_points(scores[lane], p.pid) for p in scores[lane].parts}
        acts[lane] = {p.pid: extract.Activity(scores[lane], p.pid) for p in scores[lane].parts}
    sec = build.section_of(mat)
    if sec not in aligns:
        aligns[sec] = pickle.load(open(os.path.join("dev/out/align", sec + ".pkl"), "rb"))
    opts, drops, ref_q = extract.options(scores[lane], cps[lane], acts[lane], aligns[sec], s0, s1)
    if opts:
        frags[lane].append(extract.Fragment(lane, mat, t0, t1, s0, s1, [], ref_q, opts, drops))
placed, dropped = layout.place(frags, scores)
ch = {(id(p.frag), p.piece.pid): p for p in placed if p.first}
dr = {(id(f), pid) for f, pid in dropped}
for lane in sorted(frags):
    frs = sorted(frags[lane], key=lambda f: f.t0)
    for pid in sorted({pid for f in frs for pid in f.options}):
        print(f"== {names[lane]} {pid}")
        for f in frs:
            if pid not in f.options:
                print(f"   t={f.t0:7.2f}-{f.t1:7.2f} (no options)"); continue
            o = f.options[pid][0]; w = layout.snap(f.t0)
            best = f"best [{float(w+o.start):7.2f},{float(w+o.end):7.2f}] c={o.cost:.2f} n={len(f.options[pid])}"
            if (id(f), pid) in dr:
                dec = f"DROP (drop cost {f.drop_cost[pid]:.2f})"
            else:
                p = ch[(id(f), pid)]
                dec = f"[{float(p.x):7.2f},{float(p.end):7.2f}] {p.piece.a}-{p.piece.b}" + (" CUT" if p.cut_from is not None else "") + (" LATE" if p.head_from is not None else "")
            print(f"   t={f.t0:7.2f}-{f.t1:7.2f} ({f.t1-f.t0:5.2f}s) src {f.s0:6.2f}-{f.s1:6.2f} | {best} | {dec}")

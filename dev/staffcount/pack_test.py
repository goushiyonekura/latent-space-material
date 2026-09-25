import json, os, pickle, sys, time
from collections import defaultdict
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material")
from scoremap import build, extract, layout, merge, mxl
out_dir = "output_diff003/diffusion_1cycle_slow_law_T2_presence_keep_s3"
trace = json.load(open(os.path.join(out_dir, "state_trace.json"))); names = list(trace["source_order"])
scores, cps, acts, aligns = {}, {}, {}, {}
frags = defaultdict(list)
for (lane, t0, t1, s0, s1) in build.merge_continuous(build.map_bars(trace)):
    mat = names[lane]
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
cap = int(sys.argv[1]) if len(sys.argv) > 1 else 3
t = time.time()
specs, stats = merge.pack(placed, names, cap=cap)
print("pack time %.1fs" % (time.time() - t), "cap", cap)
print(stats)
print("total staves", sum(sp.staves for sp in specs), "parts", len(specs))
for sp in specs:
    print(f"   {sp.kind:7s} part {sp.index} staves {sp.staves} units {len(sp.units)} lum {sum(1 for u in sp.units if u[0].frag.material.startswith('lumin'))}")

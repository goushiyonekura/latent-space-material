"""Run the placement of build.main and the merge plan; print the plan."""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material")
from scoremap import build, extract, layout, merge, mxl
out_dir = "output_diff003/diffusion_1cycle_slow_law_T2_presence_keep_s3"
score_dir, align_dir = "materials/scores-diffusion", "dev/out/align"
trace = json.load(open(os.path.join(out_dir, "state_trace.json")))
names = list(trace["source_order"])
scores, cps, acts, aligns = {}, {}, {}, {}
frags = defaultdict(list)
for (lane, t0, t1, s0, s1) in build.merge_continuous(build.map_bars(trace)):
    mat = names[lane]
    if lane not in scores:
        scores[lane] = mxl.parse(os.path.join(score_dir, mat + ".musicxml"))
        cps[lane] = {p.pid: extract.clean_points(scores[lane], p.pid) for p in scores[lane].parts}
        acts[lane] = {p.pid: extract.Activity(scores[lane], p.pid) for p in scores[lane].parts}
    sec = build.section_of(mat)
    if sec not in aligns:
        aligns[sec] = pickle.load(open(os.path.join(align_dir, sec + ".pkl"), "rb"))
    opts, drops, ref_q = extract.options(scores[lane], cps[lane], acts[lane], aligns[sec], s0, s1)
    if not opts:
        continue
    frags[lane].append(extract.Fragment(lane, mat, t0, t1, s0, s1, [], ref_q, opts, drops))
placed, dropped = layout.place(frags, scores)
t = time.time()
pl = merge.plan(placed, names)
print("plan time %.1fs" % (time.time() - t))
print("removed", len(pl.removed), [s.name for s in pl.removed])
print("stay", [s.name for s in pl.stay])
print("left_out", [(names[p.frag.lane], p.piece.pid, float(p.x), float(p.end)) for p in pl.left_out])
print("moves", len(pl.moves), "cost (touch, switches)", pl.cost)
touch = [(names[p.frag.lane][:18], p.piece.pid, float(p.x), float(p.end), pl.moves[id(p)].name, [(float(a), float(b)) for a, b in pl.touched[id(p)]])
         for s in pl.removed for p in s.pieces if pl.touched.get(id(p))]
print("touching", len(touch))
for x in touch: print("   ", x)
for s in pl.removed:
    seq = [pl.moves[id(u[0])].name if id(u[0]) in pl.moves else "-" for u in s.units]
    print(f"{s.name:28s} n={len(seq):2d} targets={len(set(seq))} seq={[t.replace('luminasity_','L').replace('luminusity_','L').replace('Violin','Vn').replace('Violoncello','Vc').replace('Viola','Va') for t in seq]}")
bt = Counter(pl.moves[id(p)].name for s in pl.removed for p in s.pieces if id(p) in pl.moves)
print("per target", dict(sorted(bt.items())))
pickle.dump({"moves": {(names[p.frag.lane], p.piece.pid, str(p.x)): pl.moves[id(p)].name for s in pl.removed for p in s.pieces if id(p) in pl.moves}}, open(os.environ.get("PLAN_PKL", "/dev/null"), "wb")) if os.environ.get("PLAN_PKL") else None

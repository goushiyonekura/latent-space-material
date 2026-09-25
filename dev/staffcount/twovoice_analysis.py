"""How many staves are needed when every staff may carry two voices (two passages at once)?

Units = fragment-parts (as in scoremap.merge).  Per unit: kind, written extent, note span, voice demand on its
normal staff over time (number of source voices with a written element), clef timeline on that staff, and for
luminasity whether its harmonic (0-line) staff has notes.  Per kind: peak demands, lower bounds."""
import json, math, os, pickle, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material")
from scoremap import build, extract, layout, merge, mxl
from scoremap.emit import _clef_key
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


def short(m):
    return m.split("_0")[0]


units = []
by = defaultdict(list)
for p in placed:
    by[(id(p.frag), p.piece.pid)].append(p)
for key, pls in by.items():
    pls.sort(key=lambda p: p.x)
    p0 = pls[0]; mat = names[p0.frag.lane]; part = p0.score.part(p0.piece.pid)
    lum = mat.startswith("lumin"); kind = merge.kind_of(part.name)
    nstaff = 2 if lum else 1
    x0 = min(p.x for p in pls); x1 = max(p.end for p in pls)
    vocc = defaultdict(list); clefs = []; hnotes = False; nspan = [None, None]
    for p in pls:
        a, b = p.piece.a, p.piece.b
        for e in part.elems:
            if e.kind == "note" and a <= e.pos < b and not e.chord and not e.grace and e.dur > 0:
                if e.staff == nstaff:
                    vocc[e.voice].append((p.out(e.pos), p.out(min(e.end, b))))
                if not e.rest:
                    s_, e_ = p.out(e.pos), p.out(min(e.end, b))
                    nspan[0] = s_ if nspan[0] is None else min(nspan[0], s_)
                    nspan[1] = e_ if nspan[1] is None else max(nspan[1], e_)
                    if e.staff == 1 and lum:
                        hnotes = True
        cl = [(e.pos, e.idx, c) for e in part.elems if e.kind == "attributes" for c in e.el.findall("clef")
              if int(c.get("number", "1")) == nstaff]
        cl.sort(key=lambda t: (t[0], t[1]))
        cur = None
        for (q, i, c) in cl:
            if q <= a:
                cur = c
        t_prev = p.x; k_prev = _clef_key(cur) if cur is not None else None
        for (q, i, c) in cl:
            if a < q < b:
                clefs.append((t_prev, p.out(q), k_prev)); t_prev = p.out(q); k_prev = _clef_key(c)
        clefs.append((t_prev, p.end, k_prev))
    pts = sorted({t for ivs in vocc.values() for iv in ivs for t in iv} | {x0, x1})
    demand = []
    for t0, t1 in zip(pts[:-1], pts[1:]):
        n = sum(1 for ivs in vocc.values() if any(s <= t0 and t1 <= e for s, e in ivs))
        demand.append((t0, t1, n))
    units.append({"name": f"{short(mat)} {part.name}", "mat": short(mat), "part": part.name, "kind": kind, "lum": lum,
                  "x0": x0, "x1": x1, "n0": nspan[0], "n1": nspan[1], "demand": demand,
                  "maxv": max([d[2] for d in demand] + [0]), "clefs": clefs, "hnotes": hnotes})
print("units", len(units), "luminasity", sum(u["lum"] for u in units), "others", sum(not u["lum"] for u in units))
print("units by max simultaneous written voices:", {k: sum(1 for u in units if u["maxv"] == k) for k in range(0, 4)})
for u in units:
    if u["maxv"] >= 3:
        print("   3-voice moments:", u["name"], float(u["x0"]), float(u["x1"]),
              [(float(a), float(b), n) for a, b, n in u["demand"] if n >= 3][:3])


def clef_at(u, t):
    for (a, b, k) in u["clefs"]:
        if a <= t < b:
            return k
    return u["clefs"][-1][2] if u["clefs"] else None


def dem_at(u, t, mode):
    if mode == "notes" and (u["n0"] is None or not (u["n0"] <= t < u["n1"])):
        return 0
    for (a, b, n) in u["demand"]:
        if a <= t < b:
            return n
    return 0


def active(u, t, mode):
    if mode == "written":
        return u["x0"] <= t < u["x1"]
    return u["n0"] is not None and u["n0"] <= t < u["n1"]


for mode in ("written", "notes"):
    print(f"\n===== occupancy = {mode} extent")
    for kind in ("violin", "viola", "cello"):
        us = [u for u in units if u["kind"] == kind]
        pts = sorted({t for u in us for t in (u["x0"], u["x1"], u["n0"], u["n1"]) if t is not None}
                     | {t for u in us for d in u["demand"] for t in d[:2]} | {t for u in us for c in u["clefs"] for t in c[:2]})
        peak = {k: (0, None, []) for k in ("lum", "lumH", "other_units", "voices", "bound", "units")}
        for t0, t1 in zip(pts[:-1], pts[1:]):
            t = t0
            act = [u for u in us if active(u, t, mode)]
            L = [u for u in act if u["lum"]]; O = [u for u in act if not u["lum"]]
            LH = [u for u in L if u["hnotes"]]
            V = sum(dem_at(u, t, mode) for u in act)
            byclef = defaultdict(int)
            for u in act:
                byclef[clef_at(u, t)] += dem_at(u, t, mode)
            bound = sum(math.ceil(v / 2) for v in byclef.values())
            for k, val in (("lum", len(L)), ("lumH", len(LH)), ("other_units", len(O)), ("voices", V), ("bound", bound), ("units", len(act))):
                if val > peak[k][0]:
                    peak[k] = (val, float(t), [u["name"] + (f"({clef_at(u, t)[0]}{clef_at(u, t)[1]})" if clef_at(u, t) else "") for u in act])
        print(f"-- {kind}: luminasity passages at once {peak['lum'][0]} (t={peak['lum'][1]}), of which with harmonic notes {peak['lumH'][0]}; "
              f"other passages at once {peak['other_units'][0]} (t={peak['other_units'][1]}); total written voices {peak['voices'][0]} (t={peak['voices'][1]}); "
              f"all passages at once {peak['units'][0]}")
        print(f"   lower bound, normal staves (2 voices, clef classes kept apart): {peak['bound'][0]} at t={peak['bound'][1]}")
        print(f"      {peak['bound'][2]}")
        print(f"   lower bound, harmonic staves (2 voices): {math.ceil(peak['lumH'][0] / 2)} (if every luminasity passage needs one: {math.ceil(peak['lum'][0] / 2)})")
pickle.dump(units, open(os.environ["UNITS_PKL"], "wb"))

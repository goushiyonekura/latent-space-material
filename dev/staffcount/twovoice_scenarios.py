"""Staff counts under several rule variants (two voices per staff unless stated), per instrument kind:
lower bounds from the peak demand, plus a greedy construction for the main variant.

Variants of the occupancy of a passage:
  written   = its written extent (current policy: every fragment notated in full)
  notes     = first note .. end of last note (leading / trailing rests may go)
  truncate  = written, but a passage of one part ends where the next passage of the same part begins
  sounding  = the map bar (the time the material is audible), snapped to the grid
"""
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
    snd0 = layout.snap(p0.frag.t0); snd1 = Fr(p0.frag.t1).limit_denominator(1 << 8)
    units.append({"name": f"{short(mat)} {part.name}", "mat": short(mat), "part": (p0.frag.lane, p0.piece.pid), "kind": kind, "lum": lum,
                  "x0": x0, "x1": x1, "n0": nspan[0], "n1": nspan[1], "s0": snd0, "s1": max(snd1, snd0 + Fr(1, 8)),
                  "demand": demand, "clefs": clefs, "hnotes": hnotes, "frag": id(p0.frag)})
# truncation at the next passage of the same part
byp = defaultdict(list)
for u in units:
    byp[u["part"]].append(u)
for us in byp.values():
    us.sort(key=lambda u: u["x0"])
    for a, b in zip(us, us[1:]):
        a["tr1"] = min(a["x1"], b["x0"]) if b["x0"] > a["x0"] else a["x1"]
    us[-1]["tr1"] = us[-1]["x1"]


def extent(u, mode):
    if mode == "written":
        return (u["x0"], u["x1"])
    if mode == "notes":
        return (u["n0"], u["n1"]) if u["n0"] is not None else None
    if mode == "truncate":
        return (u["x0"], u["tr1"]) if u["tr1"] > u["x0"] else None
    if mode == "sounding":
        return (u["s0"], u["s1"])
    raise KeyError(mode)


def clef_at(u, t):
    for (a, b, k) in u["clefs"]:
        if a <= t < b:
            return k
    return u["clefs"][-1][2] if u["clefs"] else None


def dem_at(u, t):
    for (a, b, n) in u["demand"]:
        if a <= t < b:
            return n
    return 1                                                  # outside the written range (sounding variant): one voice


def bounds(us, mode, cap, clef_apart=True, harmonic="notes"):
    """Peak-based lower bounds: (normal staves, harmonic staves, peak time, peak names)."""
    ex = {id(u): extent(u, mode) for u in us}
    pts = sorted({t for e in ex.values() if e for t in e} | {t for u in us for d in u["demand"] for t in d[:2]}
                 | {t for u in us for c in u["clefs"] for t in c[:2]})
    bestN = (0, None, []); bestH = (0, None)
    for t in pts:
        act = [u for u in us if ex[id(u)] and ex[id(u)][0] <= t < ex[id(u)][1]]
        byclef = defaultdict(int)
        for u in act:
            byclef[clef_at(u, t) if clef_apart else "any"] += dem_at(u, t)
        n = sum(math.ceil(v / cap) for v in byclef.values())
        if n > bestN[0]:
            bestN = (n, float(t), [u["name"] for u in act])
        h = sum(1 for u in act if u["lum"] and (u["hnotes"] or harmonic == "always"))
        hh = math.ceil(h / cap)
        if hh > bestH[0]:
            bestH = (hh, float(t))
    return bestN, bestH


def greedy(us, mode, cap, clef_apart=True):
    """First-fit / best-fit assignment of whole passages to staves (capacity `cap` voices at every instant, one
    clef at a time per staff).  Returns the number of staves used (an upper bound on the optimum)."""
    items = [(extent(u, mode), u) for u in us]
    items = [(e, u) for e, u in items if e]
    items.sort(key=lambda t: (t[0][0], t[0][1]))
    staves = []                                               # each: list of (e, u)

    def fits(st, e, u):
        pts = sorted({e[0], e[1]} | {t for (e2, u2) in st for t in e2 if e[0] <= t < e[1]}
                     | {t for (e2, u2) in st for c in u2["clefs"] for t in c[:2] if e[0] <= t < e[1]}
                     | {t for c in u["clefs"] for t in c[:2] if e[0] <= t < e[1]}
                     | {t for d in u["demand"] for t in d[:2] if e[0] <= t < e[1]}
                     | {t for (e2, u2) in st for d in u2["demand"] for t in d[:2] if e[0] <= t < e[1]})
        for t in pts:
            if not (e[0] <= t < e[1]):
                continue
            act = [(e2, u2) for (e2, u2) in st if e2[0] <= t < e2[1]]
            v = dem_at(u, t) + sum(dem_at(u2, t) for e2, u2 in act)
            if v > cap:
                return False
            if clef_apart and any(clef_at(u2, t) != clef_at(u, t) for e2, u2 in act):
                return False
        return True
    for e, u in items:
        cands = [k for k, st in enumerate(staves) if fits(st, e, u)]
        if cands:
            # best fit: the staff whose last passage ended most recently (keeps staves compact)
            k = max(cands, key=lambda k: max(e2[1] for e2, u2 in staves[k]))
            staves[k].append((e, u))
        else:
            staves.append([(e, u)])
    return len(staves)


print("units", len(units))
# simultaneous materials (sounding, from the map bars) and written
for label, mode in (("sounding", "sounding"), ("written", "written")):
    pts = sorted({t for u in units for t in extent(u, mode)})
    peak = (0, None, [])
    for t in pts:
        mats = {u["mat"] for u in units if extent(u, mode)[0] <= t < extent(u, mode)[1]}
        lm = {m for m in mats if m.startswith("lumin")}
        if len(lm) > peak[0]:
            peak = (len(lm), float(t), sorted(mats))
    print(f"max luminasity materials at once ({label}): {peak[0]} at t={peak[1]}  all materials then: {peak[2]}")

print("\nscenario table (normal staves N / harmonic staves H per kind; lower bounds from the peak; greedy = constructed count of normal staves)")
rows = [("written", 2, True, "notes"), ("notes", 2, True, "notes"), ("truncate", 2, True, "notes"), ("sounding", 2, True, "notes"),
        ("written", 2, False, "notes"), ("written", 4, True, "notes"), ("truncate", 4, True, "notes"), ("sounding", 4, True, "notes"),
        ("written", 1, True, "always")]
for mode, cap, clef_apart, harm in rows:
    tot = 0; parts = []
    for kind in ("violin", "viola", "cello"):
        us = [u for u in units if u["kind"] == kind]
        (n, tn, who), (h, th) = bounds(us, mode, cap, clef_apart, harm)
        g = greedy(us, mode, cap, clef_apart)
        parts.append(f"{kind[:2]}: N>={n} (greedy {g}) H>={h}")
        tot += n + h
    print(f"  {mode:9s} cap={cap} clef_apart={str(clef_apart):5s} harmonic={harm:6s}: total lower bound {tot:3d}   " + " | ".join(parts))

# who makes the violin peak in the written variant
us = [u for u in units if u["kind"] == "violin"]
(n, tn, who), _ = bounds(us, "written", 2)
print(f"\nviolin peak ({n} staves) at t={tn}: {who}")
us = [u for u in units if u["kind"] == "viola"]
(n, tn, who), _ = bounds(us, "written", 2)
print(f"viola peak ({n} staves) at t={tn}: {who}")
us = [u for u in units if u["kind"] == "cello"]
(n, tn, who), _ = bounds(us, "written", 2)
print(f"cello peak ({n} staves) at t={tn}: {who}")
# time profile of the total lower bound (written, cap 2): how often is it above 8, 10, 16?
prof = defaultdict(list)
pts = sorted({t for u in units for t in (u["x0"], u["x1"])})
tot_by_t = []
for t in pts:
    tot = 0
    for kind in ("violin", "viola", "cello"):
        us = [u for u in units if u["kind"] == kind]
        act = [u for u in us if u["x0"] <= t < u["x1"]]
        byclef = defaultdict(int)
        for u in act:
            byclef[clef_at(u, t)] += dem_at(u, t)
        tot += sum(math.ceil(v / 2) for v in byclef.values())
        h = sum(1 for u in act if u["lum"] and u["hnotes"])
        tot += math.ceil(h / 2)
    tot_by_t.append((t, tot))
import bisect
dur = defaultdict(float)
for (t, v), (t2, v2) in zip(tot_by_t, tot_by_t[1:]):
    dur[v] += float(t2 - t)
print("\nseconds of the piece by the momentary staff need (written, 2 voices):", {k: round(v, 1) for k, v in sorted(dur.items())})
pickle.dump(units, open(os.environ.get("UNITS2_PKL", "/dev/null"), "wb"))

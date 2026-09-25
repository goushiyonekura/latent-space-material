"""Two voices per staff + 'notate only while the material sounds': how many staves, when the cuts can only be made
at clean positions (no note / beam / tuplet / tremolo / glissando cut)?

Cut rules per fragment-part (options come from extract.options, i.e. every clean start / end near the heard span):
  current  = the cheapest option (v12/v13 policy: cheap tails, whole phrases)
  nearest  = start and end at the clean positions nearest to the heard start / end (least mismatch)
  cover    = the shortest option that contains the whole heard span (nothing heard is left out)
  inner    = the longest option inside the heard span (nothing unheard is written; heard edges may be lost)
  ideal    = occupancy = the heard span itself (cuts anywhere; theoretical floor)
"""
import json, math, os, pickle, random, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material")
from scoremap import build, extract, layout, merge, mxl
from scoremap.emit import _clef_key
from scoremap.layout import Placed, snap, assign_layers
import bisect
NOTECUT = "--notecut" in sys.argv


def clean_points_notecut(score, pid):
    """Clean positions when beams may be divided: only notes, tuplets, two-note tremolos and glissandi block a cut."""
    length = score.length
    cand = {Fr(0), length}
    spans = []
    for part in score.parts:
        if part.pid != pid:
            continue
        for m in part.measures:
            cand.add(m.start)
        for e in part.elems:
            if e.kind == "note" and not e.grace:
                cand.add(e.pos); cand.add(e.end)
                if e.dur > 0 and not e.chord:
                    spans.append((e.pos, e.end))
    for g in score.groups:
        if g.hard and g.part == pid and g.kind != "beam":
            spans.append((g.start, g.end))
    spans.sort()
    blocked = set()
    cands = sorted(cand)
    for (s_, e_) in spans:
        i = bisect.bisect_right(cands, s_)
        while i < len(cands) and cands[i] < e_:
            blocked.add(cands[i]); i += 1
    pos = [p for p in cands if p not in blocked and Fr(0) <= p <= length and (p * 64).denominator == 1]
    pen = {p: 0.0 for p in pos}
    for g in score.groups:
        if g.part != pid or (g.hard and g.kind != "beam"):
            continue
        w = extract.SOFT_WEIGHT.get(g.kind, 0.2)
        i = bisect.bisect_right(pos, g.start)
        while i < len(pos) and pos[i] < g.end:
            pen[pos[i]] += w; i += 1
    return extract.CleanPoints(pos, pen)

out_dir = "output_diff003/diffusion_1cycle_slow_law_T2_presence_keep_s3"
trace = json.load(open(os.path.join(out_dir, "state_trace.json"))); names = list(trace["source_order"])
scores, cps, acts, aligns = {}, {}, {}, {}
frags = defaultdict(list)
heard = {}                                                    # id(frag) -> (h0, h1) heard span in output offsets
for (lane, t0, t1, s0, s1) in build.merge_continuous(build.map_bars(trace)):
    mat = names[lane]
    if lane not in scores:
        scores[lane] = mxl.parse(os.path.join("materials/scores-diffusion", mat + ".musicxml"))
        cps[lane] = {p.pid: (clean_points_notecut(scores[lane], p.pid) if NOTECUT else extract.clean_points(scores[lane], p.pid)) for p in scores[lane].parts}
        acts[lane] = {p.pid: extract.Activity(scores[lane], p.pid) for p in scores[lane].parts}
    sec = build.section_of(mat)
    if sec not in aligns:
        aligns[sec] = pickle.load(open(os.path.join("dev/out/align", sec + ".pkl"), "rb"))
    al = aligns[sec]
    opts, drops, ref_q = extract.options(scores[lane], cps[lane], acts[lane], al, s0, s1)
    if not opts:
        continue
    f = extract.Fragment(lane, mat, t0, t1, s0, s1, [], ref_q, opts, drops)
    spans = al.spans(s0, s1)
    q0 = Fr(spans[0][0]).limit_denominator(1 << 10); qL = Fr(spans[-1][1]).limit_denominator(1 << 10)
    multi = len(spans) > 1
    heard[id(f)] = (q0 - ref_q, qL - ref_q, multi)
    frags[lane].append(f)


def choose(f, pid, rule):
    opts = f.options[pid]
    h0, h1, multi = heard[id(f)]
    if rule == "current" or multi:
        return opts[0]
    eps = Fr(1, 64)
    if rule == "nearest":
        return min(opts, key=lambda o: (abs(o.start - h0) + abs(o.end - h1), o.cost))
    if rule == "cover":
        c = [o for o in opts if o.start <= h0 + eps and o.end >= h1 - eps]
        if c:
            return min(c, key=lambda o: (o.end - o.start, o.cost))
        return min(opts, key=lambda o: (max(Fr(0), o.start - h0) + max(Fr(0), h1 - o.end), o.end - o.start))
    if rule == "inner":
        c = [o for o in opts if o.start >= h0 - eps and o.end <= h1 + eps]
        if c:
            return max(c, key=lambda o: (o.end - o.start, -o.cost))
        return min(opts, key=lambda o: (abs(o.start - h0) + abs(o.end - h1), o.cost))
    raise KeyError(rule)


def place_with(rule):
    placed = []
    for lane, fs in frags.items():
        for f in sorted(fs, key=lambda f: f.t0):
            w = snap(f.t0)
            for pid in f.options:
                o = choose(f, pid, rule)
                first = True
                for pc in o.pieces:
                    placed.append(Placed(f, pc, w + pc.ref, scores[lane], first, 0.0))
                    first = False
    assign_layers(placed)
    return placed


def short(m):
    return m.split("_0")[0]


def make_units(placed):
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
        vocc = defaultdict(list); clefs = []; hnotes = False
        for p in pls:
            a, b = p.piece.a, p.piece.b
            for e in part.elems:
                if e.kind == "note" and a <= e.pos < b and not e.chord and not e.grace and e.dur > 0:
                    if e.staff == nstaff:
                        vocc[e.voice].append((p.out(e.pos), p.out(min(e.end, b))))
                    if not e.rest and e.staff == 1 and lum:
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
        h0, h1, multi = heard[id(p0.frag)]
        w = snap(p0.frag.t0)
        units.append({"name": f"{short(mat)} {part.name}", "mat": short(mat), "part": (p0.frag.lane, p0.piece.pid), "kind": kind,
                      "lum": lum, "x0": x0, "x1": x1, "s0": w, "s1": w + Fr(p0.frag.t1 - p0.frag.t0).limit_denominator(1 << 8),
                      "h0": w + h0, "h1": w + h1, "demand": demand, "clefs": clefs, "hnotes": hnotes})
    return units


def clef_at(u, t):
    for (a, b, k) in u["clefs"]:
        if a <= t < b:
            return k
    return u["clefs"][-1][2] if u["clefs"] else None


def dem_at(u, t):
    for (a, b, n) in u["demand"]:
        if a <= t < b:
            return n
    return 1


def ext(u, mode):
    return (u["s0"], u["s1"]) if mode == "ideal" else (u["x0"], u["x1"])


def bounds(us, mode, cap):
    ex = {id(u): ext(u, mode) for u in us}
    pts = sorted({t for e in ex.values() for t in e} | {t for u in us for d in u["demand"] for t in d[:2]} | {t for u in us for c in u["clefs"] for t in c[:2]})
    bestN = (0, None, []); bestH = 0
    for t in pts:
        act = [u for u in us if ex[id(u)][0] <= t < ex[id(u)][1]]
        byclef = defaultdict(int)
        for u in act:
            byclef[clef_at(u, t)] += dem_at(u, t)
        n = sum(math.ceil(v / cap) for v in byclef.values())
        if n > bestN[0]:
            bestN = (n, float(t), [u["name"] for u in act])
        bestH = max(bestH, math.ceil(sum(1 for u in act if u["lum"] and u["hnotes"]) / cap))
    return bestN, bestH


def construct(us, mode, cap, rng, rule="best", jitter=True):
    items = sorted(((ext(u, mode), u) for u in us), key=lambda t: (t[0][0] + (Fr(rng.random()).limit_denominator(64) * Fr(1, 4) if jitter else 0), t[0][1]))
    staves = []

    def fits(st, e, u):
        pts = {e[0]} | {t for (e2, u2) in st for t in e2 if e[0] <= t < e[1]} | {t for c in u["clefs"] for t in c[:2] if e[0] <= t < e[1]} \
            | {t for d in u["demand"] for t in d[:2] if e[0] <= t < e[1]} | {t for (e2, u2) in st for d in u2["demand"] for t in d[:2] if e[0] <= t < e[1]} \
            | {t for (e2, u2) in st for c in u2["clefs"] for t in c[:2] if e[0] <= t < e[1]}
        for t in pts:
            act = [(e2, u2) for (e2, u2) in st if e2[0] <= t < e2[1]]
            if dem_at(u, t) + sum(dem_at(u2, t) for e2, u2 in act) > cap:
                return False
            if any(clef_at(u2, t) != clef_at(u, t) for e2, u2 in act):
                return False
        return True
    for e, u in items:
        cands = [k for k, st in enumerate(staves) if fits(st, e, u)]
        if cands:
            k = max(cands, key=lambda k: max(e2[1] for e2, u2 in staves[k])) if rule == "best" else (cands[0] if rule == "first" else rng.choice(cands))
            staves[k].append((e, u))
        else:
            staves.append([(e, u)])
    return len(staves)


page_starts = [Fr(0), Fr(9), Fr(43, 2), Fr(79, 2), Fr(111, 2), Fr(351, 4), Fr(185, 2), Fr(1057, 8), Fr(333, 2), Fr(1541, 8), Fr(231)]
for rule in [a for a in sys.argv[1:] if not a.startswith("--")] or ("current", "nearest", "cover", "inner"):
    placed = place_with(rule)
    units = make_units(placed)
    wr = sum(float(u["x1"] - u["x0"]) for u in units); so = sum(float(u["s1"] - u["s0"]) for u in units)
    lost = sum(float(max(Fr(0), u["x0"] - u["h0"]) + max(Fr(0), u["h1"] - u["x1"])) for u in units)
    extra = sum(float(max(Fr(0), u["h0"] - u["x0"]) + max(Fr(0), u["x1"] - u["h1"])) for u in units)
    n_extra_units = sum(1 for u in units if u["x1"] - u["x0"] > (u["h1"] - u["h0"]) + Fr(1, 2))
    modes = ["written"] + (["ideal"] if rule == "current" else [])
    for mode in modes:
        tot_b = 0; tot_c = 0; det = []
        for kind in ("violin", "viola", "cello"):
            us = [u for u in units if u["kind"] == kind]
            (n, tn, who), h = bounds(us, mode, 2)
            best = None
            for seed in range(10):
                rng = random.Random(seed)
                for r in ("best", "first", "random"):
                    c = construct(us, mode, 2, rng, rule=r, jitter=seed > 0)
                    best = c if best is None else min(best, c)
            det.append(f"{kind[:2]} N>={n}/{best} H={h}")
            if kind == "violin" and mode == "written" and "--peak" in sys.argv:
                print(f"   violin peak at t={tn}:")
                for u in sorted((u for u in us if ext(u, mode)[0] <= Fr(tn).limit_denominator(1 << 10) < ext(u, mode)[1]), key=lambda u: u["x0"]):
                    print(f"      {u['name']:26s} written {float(u['x0']):7.2f}-{float(u['x1']):7.2f} heard(score q) {float(u['h0']):7.2f}-{float(u['h1']):7.2f} heard(s) {float(u['s0']):7.2f}-{float(u['s1']):7.2f} voices {max(d[2] for d in u['demand'])}")
            tot_b += n + h; tot_c += best + h
        # per page need
        per_page = []
        for k in range(10):
            pts = sorted({t for u in units for t in ext(u, mode) if page_starts[k] <= t < page_starts[k + 1]} | {page_starts[k]})
            m = 0
            for t in pts:
                tot = 0
                for kind in ("violin", "viola", "cello"):
                    act = [u for u in units if u["kind"] == kind and ext(u, mode)[0] <= t < ext(u, mode)[1]]
                    byclef = defaultdict(int)
                    for u in act:
                        byclef[clef_at(u, t)] += dem_at(u, t)
                    tot += sum(math.ceil(v / 2) for v in byclef.values()) + math.ceil(sum(1 for u in act if u["lum"] and u["hnotes"]) / 2)
                m = max(m, tot)
            per_page.append(m)
        label = rule if mode == "written" else "ideal (cuts anywhere)"
        print(f"{label:22s}: staves lower bound {tot_b:2d}, constructed {tot_c:2d}   {' | '.join(det)}   per page max {max(per_page)} {per_page}")
    print(f"{'':22s}  written {wr:.0f} s vs heard {so:.0f} s (x{wr / so:.2f}); heard music left out {lost:.0f} s, unheard music written {extra:.0f} s; passages written >0.5 s beyond heard span: {n_extra_units}/{len(units)}")

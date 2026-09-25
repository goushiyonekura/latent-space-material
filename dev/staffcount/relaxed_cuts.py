"""Relaxed cutting rules (user proposal 2026-09-25): match the sounding seconds as closely as possible; a phrase that
spills into unheard time is cut mid-phrase (ties may be cut); beams may be divided (approximate values), tuplets
dissolved.  Variants of where a cut may fall:
  A1  between notes; beams / ties cut; tuplets, two-note tremolos, glissandi kept whole
  A2  between notes, also inside tuplets (dissolved); tremolos / glissandi whole
  B1  anywhere on a 1/8 grid, also inside a note (the note is shortened); tuplets / tremolos / glissandi whole
  B2  anywhere, also inside tuplets
Two voices per staff.  Occupancy of a passage = its written extent, extended to the heard seconds when the notation
would otherwise be shorter than the sound (a sustained note heard longer than its written remainder)."""
import bisect, json, math, os, pickle, random, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material")
from scoremap import build, extract, layout, merge, mxl
from scoremap.emit import _clef_key
from scoremap.layout import Placed, snap, assign_layers

out_dir = "output_diff003/diffusion_1cycle_slow_law_T2_presence_keep_s3"
trace = json.load(open(os.path.join(out_dir, "state_trace.json"))); names = list(trace["source_order"])
VARIANT = sys.argv[1]
RULES = sys.argv[2:] or ["nearest", "inner"]


def cps_variant(score, pid, variant):
    length = score.length
    part = score.part(pid)
    cand = {Fr(0), length}
    for m in part.measures:
        cand.add(m.start)
    spans = []
    for e in part.elems:
        if e.kind == "note" and not e.grace:
            cand.add(e.pos); cand.add(e.end)
            if e.dur > 0 and not e.chord and variant.startswith("A"):
                spans.append((e.pos, e.end))                  # notes stay whole in A
    if variant.startswith("B"):
        q = Fr(0)
        while q < length:
            cand.add(q); q += Fr(1, 8)
    keep_whole = {"A1": ("tuplet", "tremolo", "gliss"), "A2": ("tremolo", "gliss"),
                  "B1": ("tuplet", "tremolo", "gliss"), "B2": ("tremolo", "gliss")}[variant]
    for g in score.groups:
        if g.part == pid and g.hard and g.kind in keep_whole:
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
    return extract.CleanPoints(pos, pen)


scores, cps, acts, aligns = {}, {}, {}, {}
frags = defaultdict(list)
heard = {}
for (lane, t0, t1, s0, s1) in build.merge_continuous(build.map_bars(trace)):
    mat = names[lane]
    if lane not in scores:
        scores[lane] = mxl.parse(os.path.join("materials/scores-diffusion", mat + ".musicxml"))
        cps[lane] = {p.pid: cps_variant(scores[lane], p.pid, VARIANT) for p in scores[lane].parts}
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
    heard[id(f)] = (q0 - ref_q, qL - ref_q, len(spans) > 1)
    frags[lane].append(f)


def choose(f, pid, rule):
    opts = f.options[pid]
    h0, h1, multi = heard[id(f)]
    if rule == "current" or multi:
        return opts[0]
    eps = Fr(1, 64)
    if rule == "nearest":
        return min(opts, key=lambda o: (abs(o.start - h0) + abs(o.end - h1), o.cost))
    if rule == "inner":
        c = [o for o in opts if o.start >= h0 - eps and o.end <= h1 + eps]
        if c:
            return max(c, key=lambda o: (o.end - o.start, -o.cost))
        return min(opts, key=lambda o: (abs(o.start - h0) + abs(o.end - h1), o.cost))
    raise KeyError(rule)


def cut_kinds(score, pid, q):
    """What a cut at source position q breaks: inside a note / a beam / a tuplet / a tie."""
    part = score.part(pid)
    out = set()
    for e in part.elems:
        if e.kind == "note" and not e.rest and not e.grace and not e.chord and e.dur > 0 and e.pos < q < e.end:
            out.add("note")
    for g in score.groups:
        if g.part == pid and g.start < q < g.end:
            if g.kind in ("beam", "tuplet", "tie"):
                out.add(g.kind)
    return out


def place_with(rule):
    placed = []
    cuts = defaultdict(int)
    for lane, fs in frags.items():
        for f in sorted(fs, key=lambda f: f.t0):
            w = snap(f.t0)
            for pid in f.options:
                o = choose(f, pid, rule)
                for q in (o.pieces[0].a, o.pieces[-1].b):
                    for k in cut_kinds(scores[lane], pid, q):
                        cuts[k] += 1
                first = True
                for pc in o.pieces:
                    placed.append(Placed(f, pc, w + pc.ref, scores[lane], first, 0.0))
                    first = False
    assign_layers(placed)
    return placed, cuts


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
                if e.kind == "note" and not e.chord and not e.grace and e.dur > 0 and e.pos < b and e.end > a:
                    if e.staff == nstaff:
                        vocc[e.voice].append((p.out(max(e.pos, a)), p.out(min(e.end, b))))
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
        w = snap(p0.frag.t0)
        s0, s1 = w, w + Fr(p0.frag.t1 - p0.frag.t0).limit_denominator(1 << 8)
        # occupancy: the written extent, extended to the heard seconds where the notation is shorter than the sound
        o0, o1 = min(x0, s0), max(x1, s1)
        if x1 - x0 >= s1 - s0:
            o0, o1 = x0, x1
        pts = sorted({t for ivs in vocc.values() for iv in ivs for t in iv} | {o0, o1})
        demand = []
        for t0, t1 in zip(pts[:-1], pts[1:]):
            n = sum(1 for ivs in vocc.values() if any(s <= t0 and t1 <= e for s, e in ivs))
            demand.append((t0, t1, max(1, n)))
        if clefs and clefs[-1][1] < o1:
            clefs.append((clefs[-1][1], o1, clefs[-1][2]))
        units.append({"name": f"{short(mat)} {part.name}", "kind": kind, "lum": lum, "x0": x0, "x1": x1, "o0": o0, "o1": o1,
                      "s0": s0, "s1": s1, "demand": demand, "clefs": clefs, "hnotes": hnotes})
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


def ext(u):
    return (u["o0"], u["o1"])


def bounds(us, cap):
    ex = {id(u): ext(u) for u in us}
    pts = sorted({t for e in ex.values() for t in e} | {t for u in us for d in u["demand"] for t in d[:2]} | {t for u in us for c in u["clefs"] for t in c[:2]})
    bestN = (0, None); bestH = 0
    for t in pts:
        act = [u for u in us if ex[id(u)][0] <= t < ex[id(u)][1]]
        byclef = defaultdict(int)
        for u in act:
            byclef[clef_at(u, t)] += dem_at(u, t)
        n = sum(math.ceil(v / cap) for v in byclef.values())
        if n > bestN[0]:
            bestN = (n, float(t))
        bestH = max(bestH, math.ceil(sum(1 for u in act if u["lum"] and u["hnotes"]) / cap))
    return bestN, bestH


def construct(us, cap, rng, rule="best", jitter=True):
    items = sorted(((ext(u), u) for u in us), key=lambda t: (t[0][0] + (Fr(rng.random()).limit_denominator(64) * Fr(1, 4) if jitter else 0), t[0][1]))
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
for rule in RULES:
    placed, cuts = place_with(rule)
    units = make_units(placed)
    wr = sum(float(u["o1"] - u["o0"]) for u in units); so = sum(float(u["s1"] - u["s0"]) for u in units)
    tot_b = 0; tot_c = 0; det = []
    for kind in ("violin", "viola", "cello"):
        us = [u for u in units if u["kind"] == kind]
        (n, tn), h = bounds(us, 2)
        best = None
        for seed in range(10):
            rng = random.Random(seed)
            for r in ("best", "first", "random"):
                c = construct(us, 2, rng, rule=r, jitter=seed > 0)
                best = c if best is None else min(best, c)
        det.append(f"{kind[:2]} N>={n}/{best} H={h}")
        tot_b += n + h; tot_c += best + h
    per_page = []
    for k in range(10):
        pts = sorted({t for u in units for t in ext(u) if page_starts[k] <= t < page_starts[k + 1]} | {page_starts[k]})
        m = 0
        for t in pts:
            tot = 0
            for kind in ("violin", "viola", "cello"):
                act = [u for u in units if u["kind"] == kind and ext(u)[0] <= t < ext(u)[1]]
                byclef = defaultdict(int)
                for u in act:
                    byclef[clef_at(u, t)] += dem_at(u, t)
                tot += sum(math.ceil(v / 2) for v in byclef.values()) + math.ceil(sum(1 for u in act if u["lum"] and u["hnotes"]) / 2)
            m = max(m, tot)
        per_page.append(m)
    print(f"{VARIANT} {rule:8s}: staves lower bound {tot_b:2d}, constructed {tot_c:2d}   {' | '.join(det)}   per page max {max(per_page)} {per_page}")
    print(f"{'':12s} paper {wr:.0f} s vs heard {so:.0f} s (x{wr / so:.2f}); cuts inside: note {cuts['note']}, beam {cuts['beam']}, tuplet {cuts['tuplet']}, tie {cuts['tie']}  (of {2 * len(units)} passage ends)")

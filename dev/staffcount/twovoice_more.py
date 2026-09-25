"""(1) staves shown per page (hide empty staves) for v13 and for the 2-voice designs, (2) a better construction for
the 2-voice / written scenario (randomized restarts), (3) written vs sounding time, self-overlaps of one part."""
import json, math, os, pickle, random, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material")
S = "/private/tmp/claude-501/-Users-goushiyonekura-Claude-latent-space-material/fdc8c174-22fb-4b75-b3c2-64d7c34b0581/scratchpad"
units = pickle.load(open(S + "/units2.pkl", "rb"))
rep = json.load(open("dev/out/scoremap/keep_s3_v13.json"))
bars = [Fr(0)]
for L in rep["bar_lengths"]:
    bars.append(bars[-1] + Fr(L).limit_denominator(1 << 16))
page_starts = [Fr(0), Fr(9), Fr(43, 2), Fr(79, 2), Fr(111, 2), Fr(351, 4), Fr(185, 2), Fr(1057, 8), Fr(333, 2), Fr(1541, 8), Fr(231)]


def extent(u, mode):
    if mode == "written":
        return (u["x0"], u["x1"])
    if mode == "notes":
        return (u["n0"], u["n1"]) if u["n0"] is not None else None
    if mode == "truncate":
        return (u["x0"], u["tr1"]) if u["tr1"] > u["x0"] else None
    if mode == "sounding":
        return (u["s0"], u["s1"])


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


# ---- (1) v13: staves with content per page
out_staves = defaultdict(set)                                 # page -> (out_part, staff)
for pl in rep["placed"]:
    x, e = Fr(pl["x"]).limit_denominator(1 << 16), Fr(pl["end"]).limit_denominator(1 << 16)
    sh = pl.get("staff_shift", 0)
    nst = 2 if pl["material"].startswith("lumin") else 1
    for k in range(len(page_starts) - 1):
        if x < page_starts[k + 1] and page_starts[k] < e:
            for st in range(1, nst + 1):
                out_staves[k].add((pl["out_part"], st + sh))
print("v13 (52 staves): staves with any passage per page:", [len(out_staves[k]) for k in range(10)])

# momentary need per page for the 2-voice designs (max over the page of the per-kind bound sum)
def need_at(t, mode, cap, clef_apart=True):
    tot = 0
    for kind in ("violin", "viola", "cello"):
        act = [u for u in units if u["kind"] == kind and extent(u, mode) and extent(u, mode)[0] <= t < extent(u, mode)[1]]
        byclef = defaultdict(int)
        for u in act:
            byclef[clef_at(u, t) if clef_apart else "any"] += dem_at(u, t)
        tot += sum(math.ceil(v / cap) for v in byclef.values())
        h = sum(1 for u in act if u["lum"] and u["hnotes"])
        tot += math.ceil(h / cap)
    return tot
for mode, cap in (("written", 2), ("sounding", 2), ("written", 4), ("sounding", 4)):
    pts = sorted({t for u in units if extent(u, mode) for t in extent(u, mode)} | {t for u in units for d in u["demand"] for t in d[:2]})
    per_page = []
    for k in range(10):
        ts = [t for t in pts if page_starts[k] <= t < page_starts[k + 1]] + [page_starts[k]]
        per_page.append(max(need_at(t, mode, cap) for t in ts))
    print(f"2-voice design, {mode:8s} cap={cap}: staves needed per page (if staves may differ per page): {per_page}  max {max(per_page)}")

# ---- (2) better construction for written / cap 2 / clef apart: randomized greedy
def construct(us, mode, cap, rng, clef_apart=True, rule="best", jitter=True):
    items = [(extent(u, mode), u) for u in us]
    items = [(e, u) for e, u in items if e]
    items.sort(key=lambda t: (t[0][0] + (Fr(rng.random()).limit_denominator(64) * Fr(1, 4) if jitter else 0), t[0][1]))
    staves = []

    def fits(st, e, u):
        pts = {e[0]} | {t for (e2, u2) in st for t in e2 if e[0] <= t < e[1]} | {t for c in u["clefs"] for t in c[:2] if e[0] <= t < e[1]} \
            | {t for d in u["demand"] for t in d[:2] if e[0] <= t < e[1]} | {t for (e2, u2) in st for d in u2["demand"] for t in d[:2] if e[0] <= t < e[1]} \
            | {t for (e2, u2) in st for c in u2["clefs"] for t in c[:2] if e[0] <= t < e[1]}
        for t in pts:
            act = [(e2, u2) for (e2, u2) in st if e2[0] <= t < e2[1]]
            if dem_at(u, t) + sum(dem_at(u2, t) for e2, u2 in act) > cap:
                return False
            if clef_apart and any(clef_at(u2, t) != clef_at(u, t) for e2, u2 in act):
                return False
        return True
    for e, u in items:
        cands = [k for k, st in enumerate(staves) if fits(st, e, u)]
        if cands:
            if rule == "best":
                k = max(cands, key=lambda k: max(e2[1] for e2, u2 in staves[k]))
            elif rule == "first":
                k = cands[0]
            else:
                k = rng.choice(cands)
            staves[k].append((e, u))
        else:
            staves.append([(e, u)])
    return staves


results = {}
for mode, cap in (("written", 2), ("notes", 2), ("truncate", 2), ("sounding", 2), ("written", 4)):
    tot = 0; detail = []
    for kind in ("violin", "viola", "cello"):
        us = [u for u in units if u["kind"] == kind]
        best = None
        for seed in range(40):
            rng = random.Random(seed)
            for rule in ("best", "first", "random"):
                st = construct(us, mode, cap, rng, rule=rule, jitter=seed > 0)
                if best is None or len(st) < len(best):
                    best = st
        h = 0
        pts = sorted({t for u in us if extent(u, mode) for t in extent(u, mode)})
        for t in pts:
            hh = sum(1 for u in us if u["lum"] and u["hnotes"] and extent(u, mode)[0] <= t < extent(u, mode)[1])
            h = max(h, math.ceil(hh / cap))
        detail.append(f"{kind[:2]} N={len(best)} H={h}")
        tot += len(best) + h
    results[(mode, cap)] = tot
    print(f"constructed ({mode}, cap {cap}): total {tot}   {' | '.join(detail)}")

# ---- (3) written vs sounding time; self-overlaps
wr = sum(float(u["x1"] - u["x0"]) for u in units); so = sum(float(u["s1"] - u["s0"]) for u in units)
print(f"\nsum of written passage lengths {wr:.0f} s vs sum of sounding (map bar) lengths {so:.0f} s -> written/sounding = {wr / so:.2f}")
byp = defaultdict(list)
for u in units:
    byp[u["part"]].append(u)
self_ov = 0; self_ov_units = 0
for us in byp.values():
    us.sort(key=lambda u: u["x0"])
    for a, b in zip(us, us[1:]):
        if b["x0"] < a["x1"]:
            self_ov += 1
print(f"consecutive passages of one part that overlap in the written score: {self_ov} of {len(units) - len(byp)} consecutive pairs")
lum_only = [u for u in units if u["lum"]]
print("luminasity units", len(lum_only), "with harmonic notes", sum(u["hnotes"] for u in lum_only))

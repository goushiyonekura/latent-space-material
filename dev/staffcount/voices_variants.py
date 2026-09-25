"""Voices per staff: 2 / 3 / 4, with or without a limit on luminasity voices per staff (e.g. 4 voices but at most
one luminasity passage per staff, the other 3 for papillon / prism).  Written extents (the current notation rules,
nothing cut).  Per kind: lower bound, greedy construction, and how dense the constructed staves really are
(staff-seconds by number of simultaneous voices)."""
import math, pickle, random, sys
from collections import defaultdict
from fractions import Fraction as Fr
S = "/private/tmp/claude-501/-Users-goushiyonekura-Claude-latent-space-material/fdc8c174-22fb-4b75-b3c2-64d7c34b0581/scratchpad"
units = pickle.load(open(S + "/units2.pkl", "rb"))
MODE = "written"


def ext(u):
    return (u["x0"], u["x1"])


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


def bound(us, cap, lum_cap):
    """Per instant: sum over clef classes of max(ceil(voices/cap), ceil(luminasity passages/lum_cap))."""
    pts = sorted({t for u in us for t in ext(u)} | {t for u in us for d in u["demand"] for t in d[:2]} | {t for u in us for c in u["clefs"] for t in c[:2]})
    bestN = 0; bestH = 0
    for t in pts:
        act = [u for u in us if ext(u)[0] <= t < ext(u)[1]]
        byclef = defaultdict(lambda: [0, 0])
        for u in act:
            byclef[clef_at(u, t)][0] += dem_at(u, t)
            if u["lum"]:
                byclef[clef_at(u, t)][1] += 1
        n = sum(max(math.ceil(v / cap), math.ceil(l / lum_cap)) for v, l in byclef.values())
        bestN = max(bestN, n)
        h = sum(1 for u in act if u["lum"] and u["hnotes"])
        bestH = max(bestH, math.ceil(h / min(cap, lum_cap)))
    return bestN, bestH


def construct(us, cap, lum_cap, rng, rule="best", jitter=True):
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
            if (1 if u["lum"] else 0) + sum(1 for e2, u2 in act if u2["lum"]) > lum_cap:
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
    return staves


def density(staves):
    """staff-seconds by number of simultaneous voices, over the constructed staves."""
    hist = defaultdict(float)
    for st in staves:
        pts = sorted({t for (e, u) in st for t in e} | {t for (e, u) in st for d in u["demand"] for t in d[:2]})
        for t0, t1 in zip(pts[:-1], pts[1:]):
            v = sum(dem_at(u, t0) for (e, u) in st if e[0] <= t0 < e[1])
            if v > 0:
                hist[v] += float(t1 - t0)
    return hist


page_starts = [Fr(0), Fr(9), Fr(43, 2), Fr(79, 2), Fr(111, 2), Fr(351, 4), Fr(185, 2), Fr(1057, 8), Fr(333, 2), Fr(1541, 8), Fr(231)]
combos = [(2, 2), (3, 3), (3, 1), (3, 2), (4, 4), (4, 1), (4, 2)]
for cap, lum_cap in combos:
    tot_b = 0; tot_c = 0; det = []; hist_all = defaultdict(float); n_staves_used = 0
    for kind in ("violin", "viola", "cello"):
        us = [u for u in units if u["kind"] == kind]
        n, h = bound(us, cap, lum_cap)
        best = None
        for seed in range(8):
            rng = random.Random(seed)
            for r in ("best", "first", "random"):
                st = construct(us, cap, lum_cap, rng, rule=r, jitter=seed > 0)
                if best is None or len(st) < len(best):
                    best = st
        hs = density(best)
        for k, v in hs.items():
            hist_all[k] += v
        det.append(f"{kind[:2]} N>={n}/{len(best)} H={h}")
        tot_b += n + h; tot_c += len(best) + h
        n_staves_used += len(best)
    # per-page need (bound formula)
    per_page = []
    for k in range(10):
        pts = sorted({t for u in units for t in ext(u) if page_starts[k] <= t < page_starts[k + 1]} | {page_starts[k]})
        m = 0
        for t in pts:
            tot = 0
            for kind in ("violin", "viola", "cello"):
                act = [u for u in units if u["kind"] == kind and ext(u)[0] <= t < ext(u)[1]]
                byclef = defaultdict(lambda: [0, 0])
                for u in act:
                    byclef[clef_at(u, t)][0] += dem_at(u, t)
                    if u["lum"]:
                        byclef[clef_at(u, t)][1] += 1
                tot += sum(max(math.ceil(v / cap), math.ceil(l / lum_cap)) for v, l in byclef.values())
                tot += math.ceil(sum(1 for u in act if u["lum"] and u["hnotes"]) / min(cap, lum_cap))
            m = max(m, tot)
        per_page.append(m)
    total_active = sum(hist_all.values())
    dens = " ".join(f"{k}v:{100 * hist_all[k] / total_active:4.1f}%" for k in sorted(hist_all))
    print(f"cap={cap} luminasity<={lum_cap}: staves lower bound {tot_b:2d}, constructed {tot_c:2d}  [{' | '.join(det)}]  per page max {max(per_page)} {per_page}")
    print(f"      density of the constructed normal staves (share of staff-seconds with music, by simultaneous voices): {dens}   (active staff-seconds {total_active:.0f})")

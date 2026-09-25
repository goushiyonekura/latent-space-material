"""How many staves disappear if the non-luminasity passages (papillon, prism) are moved into the lower staves of the
luminasity instruments of the same kind (violin / viola / cello), where those are free?

A passage (one piece = one fragment of one part) moves as a whole to one target staff.  A source staff disappears
when every passage on it has found a place.  Free = no luminasity note sounds there (lenient: rests inside a
luminasity passage count as free, as do the moved passage's own rests) / no luminasity passage at all (strict: the
whole written extent of both sides must not overlap).

usage: python3 dev/merge_analysis.py REPORT.json"""
import itertools, json, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, ".")
from scoremap import mxl

rep = json.load(open(sys.argv[1]))
scores = {}
KIND = {"Violin": "violin", "Violin I": "violin", "Violin II": "violin", "Viola": "viola", "Viola I": "viola",
        "Viola II": "viola", "Violoncello": "cello", "Violoncello I": "cello", "Violoncello II": "cello"}


def sc_of(mat):
    if mat not in scores:
        scores[mat] = mxl.parse(f"materials/scores-diffusion/{mat}.musicxml")
    return scores[mat]


def note_ivs(sc, pid, a, b, x, staff=None):
    out = []
    for e in sc.part(pid).elems:
        if e.kind != "note" or e.rest or not (a <= e.pos < b) or (staff is not None and e.staff != staff):
            continue
        s = x + (e.pos - a)
        if e.grace:
            out.append((s - Fr(1, 8), s))                    # a grace note needs a little room before its note
        elif not e.chord:
            out.append((s, x + (min(e.end, b) - a)))
    return out


# targets: lower staff (2) of every luminasity instrument; sources: every other part
targets = {}          # (material, pid) -> {"kind", "notes": [...], "extent": [...]}
sources = defaultdict(lambda: {"kind": None, "pieces": []})
for p in rep["placed"]:
    mat, pid = p["material"], p["part"]
    sc = sc_of(mat)
    part = sc.part(pid)
    a, b = Fr(p["a"]), Fr(p["b"]); x = Fr(p["x"]).limit_denominator(1 << 16)
    kind = KIND[part.name]
    if mat.startswith("lumin"):
        t = targets.setdefault((mat, pid), {"kind": kind, "notes": [], "extent": []})
        t["notes"] += note_ivs(sc, pid, a, b, x, staff=2)
        t["extent"].append((x, x + (b - a)))
    else:
        s = sources[(mat, pid)]
        s["kind"] = kind
        nv = note_ivs(sc, pid, a, b, x)
        s["pieces"].append({"extent": (x, x + (b - a)), "notes": nv})
# luminasity parts that never got a passage are empty targets
for lane_mat in {p["material"] for p in rep["placed"] if p["material"].startswith("lumin")}:
    for part in sc_of(lane_mat).parts:
        targets.setdefault((lane_mat, part.pid), {"kind": KIND[part.name], "notes": [], "extent": []})


def overlaps(ivs1, ivs2):
    for s1, e1 in ivs1:
        for s2, e2 in ivs2:
            if s1 < e2 and s2 < e1:
                return True
    return False


def feasible(src_keys, mode, tlist):
    """Backtracking assignment of all pieces of the given source staves to targets (mode lenient/strict)."""
    pieces = []
    for k in src_keys:
        for pc in sources[k]["pieces"]:
            occ = pc["notes"] if mode == "lenient" else [pc["extent"]]
            if occ:
                pieces.append(occ)
    pieces.sort(key=lambda o: min(s for s, e in o))
    busy = [list(t["notes"] if mode == "lenient" else t["extent"]) for t in tlist]
    # which targets each piece can go to at all (against luminasity only)
    options = [[j for j in range(len(tlist)) if not overlaps(occ, busy[j])] for occ in pieces]
    if any(not o for o in options):
        return None
    order = sorted(range(len(pieces)), key=lambda i: (len(options[i]), min(s for s, e in pieces[i])))
    assign = {}
    placed = [[] for _ in tlist]
    steps = [0]

    def dfs(k):
        steps[0] += 1
        if steps[0] > 200000:
            raise TimeoutError
        if k == len(order):
            return True
        i = order[k]
        for j in options[i]:
            if not overlaps(pieces[i], placed[j]):
                placed[j].append(pieces[i][0] if len(pieces[i]) == 1 else None)
                placed[j].pop()
                placed[j].extend(pieces[i])
                assign[i] = j
                if dfs(k + 1):
                    return True
                for _ in pieces[i]:
                    placed[j].pop()
        return False
    try:
        return assign if dfs(0) else None
    except TimeoutError:
        return "timeout"


results = {}
for mode in ("strict", "lenient"):
    total = 0
    detail = {}
    for kind in ("violin", "viola", "cello"):
        tlist = [t for k, t in sorted(targets.items()) if t["kind"] == kind]
        srcs = sorted(k for k, s in sources.items() if s["kind"] == kind)
        best = []
        for r in range(len(srcs), 0, -1):
            found = None
            for combo in itertools.combinations(srcs, r):
                res = feasible(combo, mode, tlist)
                if res not in (None, "timeout"):
                    found = combo
                    break
            if found:
                best = list(found)
                break
        detail[kind] = (len(srcs), len(tlist), best)
        total += len(best)
    results[mode] = (total, detail)
    print(f"== {mode}: {total} staves can disappear")
    for kind, (ns, nt, best) in detail.items():
        print(f"   {kind}: {len(best)} of {ns} source staves into {nt} luminasity lower staves: {[f'{m[:20]} {p}' for m, p in best]}")
# per source staff: can it move alone? (and how many of its pieces are blocked)
for mode in ("strict", "lenient"):
    print(f"-- {mode}: each source staff on its own")
    for k in sorted(sources):
        kind = sources[k]["kind"]
        tlist = [t for kk, t in sorted(targets.items()) if t["kind"] == kind]
        blocked = 0
        for pc in sources[k]["pieces"]:
            occ = pc["notes"] if mode == "lenient" else [pc["extent"]]
            if occ and all(overlaps(occ, (t["notes"] if mode == "lenient" else t["extent"])) for t in tlist):
                blocked += 1
        ok = feasible([k], mode, tlist)
        print(f"   {k[0][:24]:24s} {k[1]} ({kind}): {len(sources[k]['pieces'])} passages, {blocked} fit nowhere -> {'movable' if ok not in (None, 'timeout') else ('timeout' if ok == 'timeout' else 'not movable')}")

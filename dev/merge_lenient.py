"""Lenient staff-merging analysis, solved exactly: a moved passage only needs its NOTES to fall where the target
luminasity lower staff has no note (its rests count as free), and passages moved onto one staff must not have
overlapping notes.  Search over the passages in time order with forward checking and memoisation of the state
(which passages on each target still reach beyond the current moment).

usage: python3 dev/merge_lenient.py REPORT.json [--nokey]"""
import itertools, sys
from functools import lru_cache
sys.argv, ARGS = sys.argv[:2], sys.argv[2:]
src = open("dev/merge_analysis.py").read()
ns = {}
exec(compile(src[:src.index("results = {}")], "merge_analysis", "exec"), ns)
sources, targets, overlaps = ns["sources"], ns["targets"], ns["overlaps"]


def solve(src_keys, tlist):
    pieces = [pc["notes"] for k in src_keys for pc in sources[k]["pieces"] if pc["notes"]]
    pieces.sort(key=lambda nv: min(s for s, e in nv))
    n, m = len(pieces), len(tlist)
    first = [min(s for s, e in nv) for nv in pieces]
    last = [max(e for s, e in nv) for nv in pieces]
    allowed = [tuple(j for j in range(m) if not overlaps(nv, tlist[j]["notes"])) for nv in pieces]
    if any(not a for a in allowed):
        return None
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for k in range(i + 1, n):
            if first[k] < last[i] and first[i] < last[k] and overlaps(pieces[i], pieces[k]):
                conflict[i][k] = conflict[k][i] = True

    @lru_cache(maxsize=None)
    def go(i, active):                   # active: per target, the assigned passages still reaching beyond first[i]
        if i == n:
            return ()
        for j in allowed[i]:
            if any(conflict[i][p] for p in active[j]):
                continue
            nxt = first[i + 1] if i + 1 < n else None
            na = tuple(tuple(p for p in (act + ((i,) if jj == j else ())) if nxt is not None and last[p] > nxt)
                       for jj, act in enumerate(active))
            res = go(i + 1, na)
            if res is not None:
                return (j,) + res
        return None
    res = go(0, tuple(() for _ in range(m)))
    go.cache_clear()
    return res


nokey = "--nokey" in ARGS
keyed = lambda k: not (k[0].startswith("papillon_violin") or k[0].startswith("prism_viola") or k[0].startswith("prism_cello"))
total = 0
for kind in ("violin", "viola", "cello"):
    tlist = [t for k, t in sorted(targets.items()) if t["kind"] == kind]
    srcs = sorted(k for k, s in sources.items() if s["kind"] == kind and (not nokey or keyed(k)))
    best = None
    for r in range(len(srcs), 0, -1):
        for combo in itertools.combinations(srcs, r):
            if solve(combo, tlist) is not None:
                best = combo
                break
        if best:
            break
    got = len(best) if best else 0
    total += got
    left = [f"{k[0].split('_0')[0]} {k[1]}" for k in srcs if not best or k not in best]
    print(f"{kind}: {got} of {len(srcs)}; stay: {left}")
print("TOTAL staves that disappear:", total)

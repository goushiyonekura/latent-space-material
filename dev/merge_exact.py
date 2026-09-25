"""Exact version of the strict staff-merging analysis (dev/merge_analysis.py): every moved passage occupies its whole
written extent; a target (luminasity lower staff) is available where no luminasity passage is written on it.
Assignment of passages to targets is solved exactly by a memoised search over the passages in time order (state:
for each target, when it becomes free again).

usage: python3 dev/merge_exact.py REPORT.json [--nokey]"""
import itertools, json, sys
from functools import lru_cache
sys.argv, ARGS = sys.argv[:2], sys.argv[2:]
src = open("dev/merge_analysis.py").read()
ns = {}
exec(compile(src[:src.index("results = {}")], "merge_analysis", "exec"), ns)
sources, targets, overlaps = ns["sources"], ns["targets"], ns["overlaps"]


def solve(src_keys, tlist):
    pieces = sorted((pc["extent"] for k in src_keys for pc in sources[k]["pieces"]), key=lambda iv: iv)
    allowed = [tuple(j for j, t in enumerate(tlist) if not overlaps([iv], t["extent"])) for iv in pieces]
    if any(not a for a in allowed):
        return None
    n, m = len(pieces), len(tlist)

    @lru_cache(maxsize=None)
    def go(i, busy):                     # busy: per target, end of its last passage (0 = free for what follows)
        if i == n:
            return ()
        s, e = pieces[i]
        for j in allowed[i]:
            if busy[j] <= s:
                nb = tuple((e if jj == j else (b if b > s else 0)) for jj, b in enumerate(busy))
                rest = go(i + 1, nb)
                if rest is not None:
                    return (j,) + rest
        return None
    res = go(0, tuple([0] * m))
    go.cache_clear()
    return None if res is None else [(pieces[i], tlist[j]["name"]) for i, j in enumerate(res)]


for k, t in targets.items():
    t["name"] = f"{k[0].split('_0')[0]} {k[1]}"
nokey = "--nokey" in ARGS
keyed = lambda k: not (k[0].startswith("papillon_violin") or k[0].startswith("prism_viola") or k[0].startswith("prism_cello"))
total = 0
plan = {}
for kind in ("violin", "viola", "cello"):
    tlist = [t for k, t in sorted(targets.items()) if t["kind"] == kind]
    srcs = sorted(k for k, s in sources.items() if s["kind"] == kind and (not nokey or keyed(k)))
    best = None
    for r in range(len(srcs), 0, -1):
        for combo in itertools.combinations(srcs, r):
            res = solve(combo, tlist)
            if res is not None:
                best = (combo, res)
                break
        if best:
            break
    got = len(best[0]) if best else 0
    total += got
    left = [f"{k[0].split('_0')[0]} {k[1]}" for k in srcs if not best or k not in best[0]]
    print(f"{kind}: {got} of {len(srcs)} source staves can go into the {len(tlist)} luminasity lower staves; stay: {left}")
    plan[kind] = best
print("TOTAL staves that disappear:", total)

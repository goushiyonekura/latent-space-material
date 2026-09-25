"""Staff merging under the user's rule of 2026-09-25: a passage (phrase) of another part moves WHOLE into the lower
staff of a luminasity instrument of the same kind, only where that luminasity staff has no note for the whole time the
phrase takes (luminasity rests there may be removed); moved phrases on one staff never overlap.  Exact search.

footprint "extent": the moved passage as written (its own leading / trailing rests included)
footprint "notes":  from its first note to the end of its last note (the phrase itself)

usage: python3 dev/merge_phrase.py REPORT.json"""
import itertools, sys
from functools import lru_cache
sys.argv, ARGS = sys.argv[:2], sys.argv[2:]
src = open("dev/merge_analysis.py").read()
ns = {}
exec(compile(src[:src.index("results = {}")], "merge_analysis", "exec"), ns)
sources, targets, overlaps, sc_of = ns["sources"], ns["targets"], ns["overlaps"], ns["sc_of"]
for k, t in targets.items():
    t["name"] = f"{k[0].split('_0')[0]} {sc_of(k[0]).part(k[1]).name}"


def footprint(pc, how):
    if how == "extent" or not pc["notes"]:
        return pc["extent"]
    return (min(s for s, e in pc["notes"]), max(e for s, e in pc["notes"]))


def solve(src_keys, tlist, how):
    pieces = sorted((footprint(pc, how), k) for k in src_keys for pc in sources[k]["pieces"] if pc["notes"])
    allowed = [tuple(j for j, t in enumerate(tlist) if not overlaps([iv], t["notes"])) for iv, k in pieces]
    if any(not a for a in allowed):
        return None

    @lru_cache(maxsize=None)
    def go(i, busy):
        if i == len(pieces):
            return ()
        (s, e), _ = pieces[i]
        for j in allowed[i]:
            if busy[j] <= s:
                r = go(i + 1, tuple((e if jj == j else (b if b > s else 0)) for jj, b in enumerate(busy)))
                if r is not None:
                    return (j,) + r
        return None
    res = go(0, tuple([0] * len(tlist)))
    go.cache_clear()
    return None if res is None else [(pieces[i], tlist[j]["name"]) for i, j in enumerate(res)]


keyed = lambda k: not (k[0].startswith("papillon_violin") or k[0].startswith("prism_viola") or k[0].startswith("prism_cello"))
out = {}
for how in ("extent", "notes"):
    for label, allow in (("all", None), ("no key signature", keyed)):
        total, stay, plans = 0, [], {}
        for kind in ("violin", "viola", "cello"):
            tlist = [t for k, t in sorted(targets.items()) if t["kind"] == kind]
            srcs = sorted(k for k, s in sources.items() if s["kind"] == kind and (allow is None or allow(k)))
            best = None
            for r in range(len(srcs), 0, -1):
                for combo in itertools.combinations(srcs, r):
                    res = solve(combo, tlist, how)
                    if res is not None:
                        best = (combo, res)
                        break
                if best:
                    break
            got = len(best[0]) if best else 0
            total += got
            stay += [f"{k[0].split('_0')[0]} {sc_of(k[0]).part(k[1]).name}" for k in srcs if not best or k not in best[0]]
            plans[kind] = best
        out[(how, label)] = (total, stay, plans)
        print(f"footprint={how:6s} sources={label:16s}: {total} staves disappear; stay: {stay}")
import pickle
pickle.dump({k: (v[0], v[1]) for k, v in out.items()}, open("dev/out/scoremap/merge_phrase.pkl", "wb"))

"""Candidate positions that split one page (output time range) into two halves: no tuplet and no two-note tremolo
may cross; beam groups crossing (would be split), notes crossing (would be tied) and rests crossing are counted;
balance = unique attack times on each side.

usage: python3 dev/halfcut.py REPORT.json T0 T1 [--top 12]"""
import bisect, json, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, ".")
from scoremap import mxl

rep = json.load(open(sys.argv[1]))
T0, T1 = Fr(sys.argv[2]), Fr(sys.argv[3])
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
scores = {}; groups = defaultdict(list); notes = []; rests = []; att = set()
for p in rep["placed"]:
    mat, pid = p["material"], p["part"]
    if mat not in scores:
        scores[mat] = mxl.parse(f"materials/scores-diffusion/{mat}.musicxml")
    sc = scores[mat]; a, b = Fr(p["a"]), Fr(p["b"]); x = Fr(p["x"]).limit_denominator(1 << 16)
    out = lambda q, x=x, a=a: x + (q - a)
    for e in sc.part(pid).elems:
        if e.kind == "note" and not e.grace and not e.chord and e.dur > 0 and a <= e.pos < b:
            (rests if e.rest else notes).append((out(e.pos), out(e.end), f"{mat[:18]} {pid} m{e.measure}"))
            if not e.rest:
                att.add(out(e.pos))
    for g in sc.groups:
        if g.part != pid:
            continue
        s, e = max(g.start, a), min(g.end, b)
        if e > s:
            groups[g.kind].append((out(s), out(e), f"{mat[:18]} {pid}"))
att = sorted(att)
cross = lambda iv, q: [w for (s, e, w) in iv if s < q < e]
total = bisect.bisect_left(att, T1) - bisect.bisect_left(att, T0)
res = []
q = T0 + 2
while q < T1 - 2:
    if not cross(groups["tuplet"], q) and not cross(groups["tremolo"], q):
        left = bisect.bisect_left(att, q) - bisect.bisect_left(att, T0)
        bal = abs(2 * left - total) / max(1, total)
        nb, nn, nr = len(cross(groups["beam"], q)), len(cross(notes, q)), len(cross(rests, q))
        ng = len(cross(groups["gliss"], q))
        score = 10 * bal + 3 * nb + 0.5 * nn + 0.15 * nr
        res.append((score, q, left, total - left, nb, nn, nr, ng))
    q += Fr(1, 16)
res.sort(key=lambda r: r[0])
print(f"page {float(T0)}-{float(T1)} s: {total} distinct attack times; {len(res)} positions cross no tuplet / tremolo")
for r in res[:top]:
    print(f"  {float(r[1]):8.4f} s  attacks {r[2]:3d} | {r[3]:3d}   beams cut {r[4]}  notes tied {r[5]}  rests split {r[6]}  gliss lines across {r[7]}")
if res:
    best = res[0][1]
    print("  beams crossing the best:", cross(groups["beam"], best)[:8])
    print("  notes crossing the best:", cross(notes, best)[:12])

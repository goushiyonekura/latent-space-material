"""Where can the long bars be cut?  For every candidate position (1/16 quarter grid) of the output timeline: which
copied elements a barline there would cross, over all 72 staves.  Hard (cannot be written across a barline without
breaking them): tuplets, two-note tremolos, beam groups, glissandi/slides.  Soft: notes (would need a tie), rests
(split into two rests).

usage: python3 dev/cutsearch.py REPORT.json T0 T1 [--n 3]"""
import json, sys
from collections import defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, ".")
from scoremap import mxl

rep = json.load(open(sys.argv[1]))
T0, T1 = Fr(sys.argv[2]), Fr(sys.argv[3])
scores = {}
hard = defaultdict(list)       # kind -> [(s, e, where)]
notes, rests = [], []
for p in rep["placed"]:
    mat, pid = p["material"], p["part"]
    if mat not in scores:
        scores[mat] = mxl.parse(f"materials/scores-diffusion/{mat}.musicxml")
    sc = scores[mat]
    a, b = Fr(p["a"]), Fr(p["b"]); x = Fr(p["x"]).limit_denominator(1 << 16)
    out = lambda q: x + (q - a)
    part = sc.part(pid)
    for e in part.elems:
        if e.kind == "note" and not e.grace and not e.chord and e.dur > 0 and a <= e.pos < b:
            (rests if e.rest else notes).append((out(e.pos), out(e.end), f"{mat[:16]} {pid} m{e.measure}"))
    for g in sc.groups:
        if g.part != pid:
            continue
        s, e = max(g.start, a), min(g.end, b)
        if e <= s:
            continue
        if g.kind in ("tuplet", "tremolo", "beam", "gliss"):
            hard[g.kind].append((out(s), out(e), f"{mat[:16]} {pid}"))


def crossing(iv, p):
    return [w for (s, e, w) in iv if s < p < e]


cands = []
p = T0 + 1
while p < T1 - 1:
    h = {k: len(crossing(v, p)) for k, v in hard.items()}
    if sum(h.values()) == 0:
        cands.append((p, len(crossing(notes, p)), len(crossing(rests, p))))
    p += Fr(1, 16)
print(f"{len(cands)} positions in {float(T0)}-{float(T1)} s cross no tuplet / tremolo / beam / glissando")
# best per region
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
L = T1 - T0
for k in range(1, n + 1):
    target = T0 + L * k / (n + 1)
    lo, hi = target - L / (2 * (n + 1)), target + L / (2 * (n + 1))
    win = [c for c in cands if lo <= c[0] <= hi]
    win.sort(key=lambda c: (c[1], c[2], abs(c[0] - target), 0 if c[0].denominator == 1 else 1))
    print(f"-- around {float(target):.1f} s ({float(lo):.1f}-{float(hi):.1f}): best", [(float(c[0]), c[1], c[2]) for c in win[:8]])
    if win:
        best = win[0][0]
        print("   notes crossing at", float(best), ":", crossing(notes, best)[:12])

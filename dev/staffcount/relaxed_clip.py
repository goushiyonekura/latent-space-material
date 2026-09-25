"""Variant B (cuts anywhere) with the alignment racing repaired: a prism passage may not advance faster than
1.5 score quarters per heard second (nominal 1.33); its occupancy is clipped accordingly.  Two voices per staff."""
import sys, math, random
from collections import defaultdict
from fractions import Fraction as Fr
S="/private/tmp/claude-501/-Users-goushiyonekura-Claude-latent-space-material/fdc8c174-22fb-4b75-b3c2-64d7c34b0581/scratchpad"
variant = sys.argv[1] if len(sys.argv) > 1 else "B2"
sys.argv = ["relaxed_cuts.py", variant, "nearest"]
src = open(S + "/relaxed_cuts.py").read()
src = src[:src.index("page_starts = [")]
ns = {"__name__": "relaxed"}
exec(compile(src, "relaxed_cuts", "exec"), ns)
placed, cuts = ns["place_with"]("nearest")
units = ns["make_units"](placed)
clipped = 0
for u in units:
    heard = u["s1"] - u["s0"]
    limit = Fr(3, 2) * heard if u["name"].startswith("prism") else Fr(5, 4) * heard
    if u["o1"] - u["o0"] > limit + Fr(1, 2):
        u["o1"] = u["o0"] + limit + Fr(1, 2); clipped += 1
        u["demand"] = [(a, min(b, u["o1"]), n) for a, b, n in u["demand"] if a < u["o1"]]
        u["clefs"] = [(a, min(b, u["o1"]), k) for a, b, k in u["clefs"] if a < u["o1"]]
print(f"{variant}: passages clipped to <= 1.5 (prism) / 1.25 (others) score quarters per heard second (+0.5): {clipped} of {len(units)}")
wr = sum(float(u["o1"] - u["o0"]) for u in units); so = sum(float(u["s1"] - u["s0"]) for u in units)
tot_b = 0; tot_c = 0; det = []
for kind in ("violin", "viola", "cello"):
    us = [u for u in units if u["kind"] == kind]
    (n, tn), h = ns["bounds"](us, 2)
    best = None
    for seed in range(10):
        rng = random.Random(seed)
        for r in ("best", "first", "random"):
            c = ns["construct"](us, 2, rng, rule=r, jitter=seed > 0)
            best = c if best is None else min(best, c)
    det.append(f"{kind[:2]} N>={n}/{best} H={h}")
    tot_b += n + h; tot_c += best + h
print(f"   staves lower bound {tot_b}, constructed {tot_c}   {' | '.join(det)}   paper {wr:.0f} s vs heard {so:.0f} s (x{wr / so:.2f})")

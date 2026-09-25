import sys, math, random
from collections import defaultdict
from fractions import Fraction as Fr
S="/private/tmp/claude-501/-Users-goushiyonekura-Claude-latent-space-material/fdc8c174-22fb-4b75-b3c2-64d7c34b0581/scratchpad"
variant = sys.argv[1]; cap = int(sys.argv[2])
sys.argv = ["relaxed_cuts.py", variant, "nearest"]
src = open(S + "/relaxed_cuts.py").read(); src = src[:src.index("page_starts = [")]
ns = {"__name__": "relaxed"}; exec(compile(src, "relaxed_cuts", "exec"), ns)
placed, cuts = ns["place_with"]("nearest"); units = ns["make_units"](placed)
page_starts = [Fr(0), Fr(9), Fr(43, 2), Fr(79, 2), Fr(111, 2), Fr(351, 4), Fr(185, 2), Fr(1057, 8), Fr(333, 2), Fr(1541, 8), Fr(231)]
tot_b = 0; tot_c = 0; det = []
for kind in ("violin", "viola", "cello"):
    us = [u for u in units if u["kind"] == kind]
    (n, tn), h = ns["bounds"](us, cap)
    best = None
    for seed in range(8):
        rng = random.Random(seed)
        for r in ("best", "first", "random"):
            c = ns["construct"](us, cap, rng, rule=r, jitter=seed > 0); best = c if best is None else min(best, c)
    det.append(f"{kind[:2]} N>={n}/{best} H={h}"); tot_b += n + h; tot_c += best + h
per_page = []
for k in range(10):
    pts = sorted({t for u in units for t in ns["ext"](u) if page_starts[k] <= t < page_starts[k + 1]} | {page_starts[k]})
    m = 0
    for t in pts:
        tot = 0
        for kind in ("violin", "viola", "cello"):
            act = [u for u in units if u["kind"] == kind and ns["ext"](u)[0] <= t < ns["ext"](u)[1]]
            byclef = defaultdict(int)
            for u in act: byclef[ns["clef_at"](u, t)] += ns["dem_at"](u, t)
            tot += sum(math.ceil(v / cap) for v in byclef.values()) + math.ceil(sum(1 for u in act if u["lum"] and u["hnotes"]) / cap)
        m = max(m, tot)
    per_page.append(m)
print(f"{variant} cap={cap} nearest: staves lower bound {tot_b}, constructed {tot_c}   {' | '.join(det)}   per page max {max(per_page)} {per_page}")

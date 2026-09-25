"""Re-align a prism section with stronger penalties on the non-diagonal DTW steps and compare with the current
alignment: racing metric over the map's fragments (score quarters per heard second), pitch / onset evidence.

usage: python3 prism_steps.py SECTION CONFIG   (CONFIG: base | p1 | p2 | p3 | p4)"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material"); sys.path.insert(0, "/Users/goushiyonekura/Claude/latent-space-material/dev")
os.chdir("/Users/goushiyonekura/Claude/latent-space-material")
from scoremap import align, build, mxl
import align_sections as AS, evidence as EV
S = "/private/tmp/claude-501/-Users-goushiyonekura-Claude-latent-space-material/fdc8c174-22fb-4b75-b3c2-64d7c34b0581/scratchpad"
sec, cfg = sys.argv[1], sys.argv[2]
CONFIGS = {
    "base": None,
    "p1": [(1, 1, 0.0), (2, 1, 0.10), (1, 2, 0.10), (3, 1, 0.30), (1, 3, 0.30), (1, 0, 0.03)],
    "p2": [(1, 1, 0.0), (2, 1, 0.20), (1, 2, 0.20), (3, 1, 0.60), (1, 3, 0.60), (1, 0, 0.06)],
    "p3": [(1, 1, 0.0), (2, 1, 0.10), (1, 2, 0.10), (1, 0, 0.03)],                       # slopes 1/2 .. 2 only
    "p4": [(1, 1, 0.0), (2, 1, 0.40), (1, 2, 0.40), (3, 1, 1.20), (1, 3, 1.20), (1, 0, 0.10)],
    "p5": [(1, 1, 0.0), (2, 1, 0.80), (1, 2, 0.80), (3, 1, 2.40), (1, 3, 2.40), (1, 0, 0.20)],
    "p6": [(1, 1, 0.0), (2, 1, 0.40), (1, 2, 0.40), (1, 0, 0.10)],
    "p7": [(1, 1, 0.0), (2, 1, 0.60), (1, 2, 0.60), (3, 1, 1.80), (1, 3, 1.80), (1, 0, 0.15)],
    "p8": [(1, 1, 0.0), (2, 1, 1.60), (1, 2, 1.60), (3, 1, 4.80), (1, 3, 4.80), (1, 0, 0.40)],
}
names = AS.SECTIONS[sec]
scores = [mxl.parse(AS.S + n + ".musicxml") for n in names]
if cfg == "base":
    al = pickle.load(open(f"dev/out/align/{sec}.pkl", "rb"))
else:
    align.STEPS = CONFIGS[cfg]
    kw = dict(AS.weights_of(sec))
    al = align.align(scores, [AS.A + n + ".wav" for n in names], repeats=AS.repeats_of(scores[0]), anchors=AS.anchors_of(sec), **kw)
    pickle.dump(al, open(f"{S}/{sec}_{cfg}.pkl", "wb"))
# racing over the map's fragments of this section's materials
out_dir = "output_diff003/diffusion_1cycle_slow_law_T2_presence_keep_s3"
trace = json.load(open(os.path.join(out_dir, "state_trace.json"))); lanes = list(trace["source_order"])
rates = []
for (lane, t0, t1, s0, s1) in build.merge_continuous(build.map_bars(trace)):
    if build.section_of(lanes[lane]) != sec:
        continue
    spans = al.spans(s0, s1)
    q = sum(b - a for a, b, p in spans)
    rates.append((q / (s1 - s0), t1 - t0, lanes[lane].split("_0")[0], t0, t1))
rates.sort(reverse=True)
fast = [r for r in rates if r[0] > 2.0]
dwell = [r for r in rates if r[0] < 0.3]
# evidence: mean pitch z over notes and onset z over attacks (stem 0, whole section)
tot_pz, tot_oz, nb = 0.0, 0.0, 0
for stem in range(len(names)):
    for (bar, pz, oz, n) in EV.bar_evidence(sec, al, stem=stem):
        if n:
            tot_pz += pz * n; tot_oz += (oz if oz == oz else 0.0) * n; nb += n
# local tempo statistics of the path (quarters per second over 1-second windows)
tt, qq = al.path_t, al.path_q
w = []
for t in np.arange(tt[0], tt[-1] - 1.0, 0.5):
    q0 = al.q_at(t)[0]; q1 = al.q_at(t + 1.0)[0]
    w.append(q1 - q0)
w = np.array(w)
print(f"{sec} {cfg}: cost {al.cost:.4f}; fragments {len(rates)}: >2 q/s {len(fast)} ({sum(r[1] for r in fast):.0f} s of audio), <0.3 q/s {len(dwell)} ({sum(r[1] for r in dwell):.0f} s); "
      f"median {np.median([r[0] for r in rates]):.2f} q/s; 1-s windows: >2.5 q/s {100 * (w > 2.5).mean():.1f}%, <0.3 q/s {100 * (w < 0.3).mean():.1f}%; "
      f"evidence pitch z {tot_pz / nb:.3f}, onset z {tot_oz / nb:.3f}")
print("   fastest:", [(round(r[0], 2), r[2], r[3], r[4]) for r in rates[:6]])

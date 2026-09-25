"""Where do independent stem alignments disagree?  usage: python3 dev/disagree.py SECTION [thr_q]"""
import sys, itertools, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl, align
import align_sections as A
sec = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
names = A.SECTIONS[sec]; scores = [mxl.parse(A.S + n + ".musicxml") for n in names]
w = A.weights_of(sec)
als = [align.align([sc], [A.A + n + ".wav"], repeats=A.repeats_of(sc), anchors=A.anchors_of(sec), **w) for sc, n in zip(scores, names)]
grid = np.arange(0.25, als[0].audio_dur - 0.25, 0.25)
qs = np.array([[a.q_at(t)[0] for t in grid] for a in als])
spread = qs.max(axis=0) - qs.min(axis=0)
print(f"{sec}: stems {len(als)}, spread median {np.median(spread):.3f} q, p90 {np.percentile(spread,90):.2f} q, passes {[a.seg.max()+1 for a in als]}")
bad = spread > thr
runs = []; start = None
for i, b in enumerate(bad):
    if b and start is None: start = i
    if (not b or i == len(bad) - 1) and start is not None:
        runs.append((grid[start], grid[i], spread[start:i + 1].max(), qs[:, start:i + 1].mean(axis=1))); start = None
for a, b, mx, qm in runs:
    print(f"  {a:6.2f}-{b:6.2f}s  max spread {mx:.2f} q   stems at q {np.round(qm, 2)}")

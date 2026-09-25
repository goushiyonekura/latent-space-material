"""Inter-stem agreement of independent alignments (papillon / prism sections) for weight settings.
usage: python3 dev/stemagree.py SECTION [SECTION ...]"""
import sys, itertools, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl, align
import align_sections as A
SETTINGS = [(0.5, 1.5, 1.0), (1.0, 1.0, 0.5), (2.0, 1.0, 0.5), (0.5, 2.5, 1.0), (1.0, 2.0, 0.7), (0.3, 1.0, 1.5)]
for sec in sys.argv[1:]:
    names = A.SECTIONS[sec]; scores = [mxl.parse(A.S + n + ".musicxml") for n in names]
    for wo, wp, wc in SETTINGS:
        als = [align.align([sc], [A.A + n + ".wav"], repeats=A.repeats_of(sc), w_onset=wo, w_pons=wp, w_chroma=wc)
               for sc, n in zip(scores, names)]
        grid = np.arange(0.5, als[0].audio_dur - 0.5, 0.2)
        qs = [np.array([a.q_at(t)[0] for t in grid]) for a in als]
        d = np.concatenate([np.abs(x - y) for x, y in itertools.combinations(qs, 2)])
        print(f"{sec:7s} w_onset {wo} w_pons {wp} w_chroma {wc}: median {np.median(d):.3f} q  p90 {np.percentile(d, 90):.3f} q  p98 {np.percentile(d, 98):.2f} q")

"""Onset-agreement score of an alignment (mean over attacks of onset strength at the mapped time, dynamics
weighted) for several weight settings.  usage: python3 dev/evalw.py SECTION"""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl, align
import align_sections as A, onsets as O
def agreement(sec, al, scores):
    f = O.feats(sec, 0); fl = np.convolve(f["flux"], np.ones(3) / 3, mode="same"); ta = f["t"]
    tot = 0; n = 0
    for part in scores[0].parts:
        for e in part.elems:
            if e.kind == "note" and not e.rest and not e.chord and not e.grace and not any(t.get("type") == "stop" for t in e.el.findall("tie")):
                t = al.t_at(float(e.pos), 0); i = min(len(fl) - 1, np.searchsorted(ta, t))
                tot += fl[i]; n += 1
    return tot / max(n, 1)
if __name__ == "__main__":
    sec = sys.argv[1]
    names = A.SECTIONS[sec]; scores = [mxl.parse(A.S + n + ".musicxml") for n in names]
    for wo, wp, wc in [(0.5, 1.5, 1.0), (1.0, 1.5, 1.0), (1.5, 1.0, 1.0), (1.0, 1.0, 0.5), (2.0, 1.0, 0.5), (0.5, 2.5, 1.0)]:
        al = align.align(scores, [A.A + n + ".wav" for n in names], repeats=A.repeats_of(scores[0]),
                         anchors=A.anchors_of(sec), w_onset=wo, w_pons=wp, w_chroma=wc)
        print(f"{sec} w_onset {wo} w_pons {wp} w_chroma {wc}: agreement {agreement(sec, al, scores):.3f} cost {al.cost:.3f}")

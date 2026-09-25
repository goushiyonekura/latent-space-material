"""Alignment of a section with repeats (papillon_2): the score is unrolled in performance order
([m1] x n1, [m2-m15] x 2, [m16-m17], [m18] x n2, [m19]), aligned with refine_align, then mapped back to source
positions with a pass number per repeat (Alignment.seg).  n1, n2 (the 'ad lib.' counts) are chosen by the fit.

usage: python3 dev/unroll_align.py pap2"""
import dataclasses, pickle, sys
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import align, mxl
import align_sections as AS
import refine_align as RA


def unroll(sc, segs):
    """Score whose timeline is the concatenation of the source spans segs [(q0, q1)]."""
    parts = []
    for p in sc.parts:
        elems, measures = [], []
        off = Fr(0); idx = 0
        for (q0, q1) in segs:
            q0, q1 = Fr(q0), Fr(q1)
            for e in p.elems:
                if q0 <= e.pos < q1 or (e.kind == "attributes" and e.pos <= q0):
                    elems.append(dataclasses.replace(e, pos=(e.pos - q0 if e.pos >= q0 else Fr(0)) + off, idx=e.idx + idx,
                                                     anchor=(e.anchor + idx if e.anchor >= 0 else -1)))
            for m in p.measures:
                if q0 <= m.start < q1:
                    measures.append(mxl.Measure(m.number, m.start - q0 + off, min(m.length, q1 - m.start), m.implicit))
            idx += max(e.idx for e in p.elems) + 1
            off += q1 - q0
        parts.append(mxl.Part(p.pid, p.name, p.score_part, elems, measures, [(Fr(0), d) for _, d in p.divisions[:1]],
                              p.staves, p.first_attributes))
    groups = []
    off = Fr(0)
    for (q0, q1) in segs:
        groups += [dataclasses.replace(g, start=g.start - Fr(q0) + off, end=g.end - Fr(q0) + off) for g in sc.groups
                   if Fr(q0) <= g.start < Fr(q1)]
        off += Fr(q1) - Fr(q0)
    return mxl.Score(sc.path, sc.root, parts, groups)


def back_map(al_u, segs):
    """Unrolled alignment -> Alignment over source positions with a pass number per unrolled segment."""
    bounds = np.cumsum([0.0] + [q1 - q0 for q0, q1 in segs])
    t = al_u.path_t; qu = al_u.path_q
    k = np.clip(np.searchsorted(bounds, qu, side="right") - 1, 0, len(segs) - 1)
    q = np.array([segs[i][0] for i in k]) + (qu - bounds[k])
    return align.Alignment(t, q, k.astype(int), al_u.cost, 0.0, al_u.audio_dur)


def segs_of(n1, n2):
    return [(0, 2)] * n1 + [(2, 30)] * 2 + [(30, 34)] + [(34, 37)] * n2 + [(37, 38)]


if __name__ == "__main__":
    sec = sys.argv[1]
    names = AS.SECTIONS[sec]
    srcs = [mxl.parse(AS.S + n + ".musicxml") for n in names]
    wavs = [AS.A + n + ".wav" for n in names]
    dur = min(RA.Stem(w, s).dur for w, s in zip(wavs[:1], srcs[:1]))
    res = []
    for n1 in range(4, 8):
        for n2 in range(3, 7):
            segs = segs_of(n1, n2)
            Lu = sum(b - a for a, b in segs)
            scs = [unroll(s, segs) for s in srcs]
            al_u, knots, times = RA.refine(sec, None, R=14.0, mu=1.0, scores=scs, wavs=wavs,
                                           prior_fn=lambda q, Lu=Lu: q * dur / Lu, spq_fn=lambda q: 60.0 / 58.0,
                                           time_weight=True)
            gain = -al_u.cost
            res.append((gain / Lu, gain, n1, n2, al_u, segs))
            print(f"n1={n1} n2={n2}: gain {gain:.2f} per quarter {gain / Lu:.3f}", flush=True)
    res.sort(key=lambda r: -r[1])
    g, gl, n1, n2, al_u, segs = res[0]
    al = back_map(al_u, segs)
    pickle.dump(al, open(f"dev/out/align/{sec}_unrolled.pkl", "wb"))
    print("best n1, n2 =", n1, n2)
    starts = np.cumsum([0.0] + [b - a for a, b in segs])
    print("pass starts:", [f"{segs[i]}@{al_u.t_at(starts[i], 0):.1f}" for i in range(len(segs))])

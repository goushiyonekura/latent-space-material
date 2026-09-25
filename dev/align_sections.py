"""Run the audio-to-score alignment per section and draw diagnostic images.

usage: python3 dev/align_sections.py SECTION [--stem K] [--t0 A --t1 B] [--px 40]
Sections: lum7 lum8 lum9 lum10 lum11 lum14 pap1 pap2 pap4 pap5 prism1 prism2
"""
import pickle, sys, os
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl, align
import plotalign

S = "materials/scores-diffusion/"
A = "inputs/diffusion-003/"
SECTIONS = {
    "lum7": ["luminusity_7_02.10.047_02.29.381"],
    "lum8": ["luminasity_8_02.29.381_02.47.541"],
    "lum9": ["luminasity_9_02.47.541_03.07.094"],
    "lum10": ["luminasity_10_03.07.094_03.27.266"],
    "lum11": ["luminasity_11_03.27.266_03.50.423"],
    "lum14": ["luminasity_14_04.39.213_04.59"],
    "pap1": ["papillon_1_00.00.000_01.29.318", "papillon_viola_1_00.00.000_01.29.318", "papillon_violin_1_00.00.000_01.29.318"],
    "pap2": ["papillon_2_01.29.318_03.06.429", "papillon_viola_2_01.29.318_03.06.429", "papillon_violin_2_01.29.318_03.06.429"],
    "pap4": ["papillon_4_04.34.344_06.43.645", "papillon_viola_4_04.34.344_06.43.645", "papillon_violin_4_04.34.344_06.43.645"],
    "pap5": ["papillon_5_06.43.645_08.13.352", "papillon_viola_5_06.43.645_08.13.352", "papillon_violin_5_06.43.645_08.13.352"],
    "prism1": ["prism_1_00.00.000_01.28.904", "prism_cello_1_00.00.000_01.28.904", "prism_viola_1_00.00.000_01.28.904"],
    "prism2": ["prism_2_01.28.904_02.38.937", "prism_cello_2_01.28.904_02.38.937", "prism_viola_2_01.28.904_02.38.937"],
}


def repeats_of(score):
    """(start_q, end_q) of repeat sections from the repeat barlines."""
    out, start = [], None
    part = score.parts[0]
    for e in part.elems:
        if e.kind != "barline":
            continue
        r = e.el.find("repeat")
        if r is None:
            continue
        if r.get("direction") == "forward":
            start = e.pos
        elif r.get("direction") == "backward":
            out.append((float(start if start is not None else 0), float(e.pos)))
            start = None
    return out


ANCHOR_FILE = "dev/out/align/anchors.json"
WEIGHTS = {"lum": dict(w_onset=2.0, w_pons=1.0, w_chroma=0.5), "pap": dict(w_onset=0.3, w_pons=1.0, w_chroma=1.5),
           "prism": dict(w_onset=2.0, w_pons=1.0, w_chroma=0.5, steps=align.STEPS_STEADY)}   # steps: 2026-09-25 (§12.5)


def weights_of(sec):
    return WEIGHTS["prism" if sec.startswith("prism") else sec[:3]]


def anchors_of(sec):
    import json
    if os.path.exists(ANCHOR_FILE):
        return [tuple(a[:2]) for a in json.load(open(ANCHOR_FILE)).get(sec, [])]
    return []


def run(sec, ref_tempo=None, cache=True, **kw):
    names = SECTIONS[sec]
    path = f"dev/out/align/{sec}.pkl"
    scores = [mxl.parse(S + n + ".musicxml") for n in names]
    if cache and os.path.exists(path) and not kw:
        al = pickle.load(open(path, "rb"))
    else:
        kw2 = dict(weights_of(sec)); kw2.update(kw)
        al = align.align(scores, [A + n + ".wav" for n in names], ref_tempo=ref_tempo,
                         repeats=repeats_of(scores[0]), anchors=anchors_of(sec), **kw2)
        os.makedirs("dev/out/align", exist_ok=True)
        pickle.dump(al, open(path, "wb"))
    return names, scores, al


def draw(sec, al, names, scores, stem=0, t0=None, t1=None, px=40, out=None):
    sc = scores[stem]
    ev = align.score_events(sc)
    events = []
    # map each score event through every pass that covers it
    for s in range(al.seg.max() + 1):
        m = al.seg == s
        qmin, qmax = al.path_q[m].min(), al.path_q[m].max()
        for st, en, pc, w, attack, mid in zip(ev.starts, ev.ends, ev.pcs, ev.weights, ev.attack, ev.midis):
            if st < qmin - 1e-9 or st > qmax + 1e-9:
                continue
            ts = al.t_at(st, s); te = al.t_at(min(en, qmax), s)
            events.append((ts, te, mid, attack))
    meas = []
    for s in range(al.seg.max() + 1):
        m = al.seg == s
        qmin, qmax = al.path_q[m].min(), al.path_q[m].max()
        for mm in sc.parts[0].measures:
            if qmin - 1e-9 <= float(mm.start) <= qmax + 1e-9:
                meas.append((al.t_at(float(mm.start), s), "m" + mm.number))
    t0 = 0.0 if t0 is None else t0
    t1 = al.audio_dur if t1 is None else t1
    out = out or f"dev/out/align/{sec}_{stem}_{int(t0)}_{int(t1)}.png"
    plotalign.render(out, A + names[stem] + ".wav", events, meas, t0, t1, px_per_s=px)
    return out


if __name__ == "__main__":
    sec = sys.argv[1]
    args = sys.argv[2:]
    get = lambda k, d=None, f=float: f(args[args.index(k) + 1]) if k in args else d
    names, scores, al = run(sec, cache="--fresh" not in args)
    print(f"{sec}: cost {al.cost:.4f}, passes {al.seg.max() + 1}, audio {al.audio_dur:.1f} s")
    print(draw(sec, al, names, scores, stem=get("--stem", 0, int), t0=get("--t0"), t1=get("--t1"), px=get("--px", 40)))

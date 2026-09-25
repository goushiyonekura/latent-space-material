"""Silence landmarks: audio re-entries after silence vs score re-entries after rests (all parts), through an
alignment.  A re-entry of the score should sit on a re-entry of the audio.

usage: python3 dev/landmarks.py SECTION [stem] [--pkl PATH]"""
import pickle, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from latent_space.audio_io import read_wav
from scoremap import mxl
import align_sections as AS


def audio_reentries(path, thr_db=-42.0, min_sil=0.35, hop=0.01):
    x, info = read_wav(path); fs = int(info.sample_rate); x = x.astype(float).mean(1)
    h = int(hop * fs); w = int(0.03 * fs)
    env = np.array([np.sqrt(np.mean(x[i:i + w] ** 2)) for i in range(0, len(x) - w, h)])
    db = 20 * np.log10(env + 1e-9)
    sil = db < thr_db
    out = []            # (silence start, re-entry time)
    i = 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) * hop >= min_sil and j < len(sil):
                out.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return out, db, hop


def score_reentries(sc, min_rest=0.5):
    """Positions where some note starts after all parts were silent for >= min_rest quarters."""
    iv = sorted((e.pos, e.end) for p in sc.parts for e in p.elems
                if e.kind == "note" and not e.rest and not e.grace and e.dur > 0)
    out = []
    end = None
    for s, e in iv:
        if end is not None and s - end >= min_rest:
            out.append((float(end), float(s)))
        end = e if end is None else max(end, e)
    return out


if __name__ == "__main__":
    sec = sys.argv[1]; stem = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
    pkl = sys.argv[sys.argv.index("--pkl") + 1] if "--pkl" in sys.argv else f"dev/out/align/{sec}.pkl"
    al = pickle.load(open(pkl, "rb"))
    names = AS.SECTIONS[sec]
    sc = mxl.parse(AS.S + names[stem] + ".musicxml")
    au, db, hop = audio_reentries(AS.A + names[stem] + ".wav")
    sr = score_reentries(sc)
    print(f"{sec} stem {stem}: audio re-entries after silence: " + " ".join(f"{b:.2f}(sil {b - a:.1f})" for a, b in au))
    for (qe, qs) in sr:
        for s in range(al.seg.max() + 1):
            m = al.seg == s
            if not (al.path_q[m].min() - 1e-6 <= qs <= al.path_q[m].max() + 1e-6):
                continue
            t = al.t_at(qs, s); te = al.t_at(qe, s)
            near = min(au, key=lambda x: abs(x[1] - t)) if au else (0, float("nan"))
            meas = next((mm.number for mm in sc.parts[0].measures if mm.start <= qs < mm.start + mm.length), "?")
            print(f"   score re-entry q{qs:6.2f} (m{meas}, rest from q{qe:.2f}) pass {s} -> t {t:6.2f} (rest from {te:6.2f}); nearest audio re-entry {near[1]:6.2f}  diff {near[1] - t:+.2f}")

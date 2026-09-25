"""Align a run of consecutive sections (contiguous audio excerpts + consecutive score excerpts) in one piece, as an
independent check of the per-section alignments (no assumption that every excerpt boundary is a barline).

usage: python3 dev/joint_align.py lum7 lum8 lum9 lum10 lum11 [--stem K]
Writes dev/out/align/joint_<first>_<last>[_stemK].pkl (Alignment over the concatenation) and prints, per section,
where the joint alignment puts the section's first/last barline against the audio boundaries.
"""
import dataclasses, os, pickle, sys
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from latent_space.audio_io import read_wav, write_wav_float32
from scoremap import mxl, align
import align_sections as AS

SCR = os.environ.get("SCR", "/private/tmp/claude-501/-Users-goushiyonekura-Claude-latent-space-material/ba5501c8-6431-4ac4-863a-0398a431373b/scratchpad")


def concat_scores(scores):
    base = scores[0]
    parts = []
    for k, p0 in enumerate(base.parts):
        elems, measures, divs = [], [], []
        off = Fr(0); idx = 0
        for sc in scores:
            p = sc.parts[k]
            for e in p.elems:
                elems.append(dataclasses.replace(e, pos=e.pos + off, idx=e.idx + idx, anchor=(e.anchor + idx if e.anchor >= 0 else -1)))
            for m in p.measures:
                measures.append(mxl.Measure(m.number, m.start + off, m.length, m.implicit))
            divs += [(q + off, d) for q, d in p.divisions]
            idx += max(e.idx for e in p.elems) + 1
            off += sc.length
        parts.append(mxl.Part(p0.pid, p0.name, p0.score_part, elems, measures, divs, p0.staves, p0.first_attributes))
    groups = []
    off = Fr(0)
    for sc in scores:
        groups += [dataclasses.replace(g, start=g.start + off, end=g.end + off) for g in sc.groups]
        off += sc.length
    return mxl.Score("+".join(s.path for s in scores), base.root, parts, groups)


def main(args):
    stem = int(args[args.index("--stem") + 1]) if "--stem" in args else 0
    secs = [a for a in args if not a.startswith("--") and not a.isdigit()]
    names = [AS.SECTIONS[s][stem] for s in secs]
    scores = [mxl.parse(AS.S + n + ".musicxml") for n in names]
    joint = concat_scores(scores)
    xs, fs = [], None
    for n in names:
        x, info = read_wav(AS.A + n + ".wav")
        fs = int(info.sample_rate); xs.append(x.astype(np.float32))
    wav = os.path.join(SCR, f"joint_{secs[0]}_{secs[-1]}_{stem}.wav")
    write_wav_float32(wav, np.concatenate(xs), fs)
    kw = AS.weights_of(secs[0])
    al = align.align([joint], [wav], **kw)
    out = f"dev/out/align/joint_{secs[0]}_{secs[-1]}" + (f"_stem{stem}" if stem else "") + ".pkl"
    pickle.dump((secs, [float(s.length) for s in scores], [len(x) / fs for x in xs], al), open(out, "wb"))
    t_off = 0.0; q_off = 0.0
    for sec, sc, x in zip(secs, scores, xs):
        dur = len(x) / fs
        tq0 = al.t_at(q_off, 0); tq1 = al.t_at(q_off + float(sc.length), 0)
        q0 = al.q_at(t_off)[0] - q_off; q1 = al.q_at(t_off + dur)[0] - q_off
        print(f"{sec:6s} audio {t_off:7.2f}-{t_off + dur:7.2f}  joint: first bar at {tq0:7.2f} ({tq0 - t_off:+.2f}), "
              f"end at {tq1:7.2f} ({tq1 - t_off - dur:+.2f}) | audio start = q {q0:+.2f}, audio end = q {q1 - float(sc.length):+.2f} vs end")
        t_off += dur; q_off += float(sc.length)
    print(out, f"cost {al.cost:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])


def local(joint_pkl, sec):
    """The joint alignment restricted to one section, in that section's own audio seconds / score quarters."""
    secs, lens, durs, J = pickle.load(open(joint_pkl, "rb"))
    k = secs.index(sec)
    toff = sum(durs[:k]); qoff = sum(lens[:k])
    t = J.path_t - toff; q = J.path_q - qoff
    m = (t >= -1e-9) & (t <= durs[k] + 1e-9)
    tt = t[m]; qq = np.clip(q[m], 0.0, lens[k])
    return align.Alignment(tt, qq, np.zeros(len(tt), dtype=int), J.cost, J.q_per_frame, durs[k])

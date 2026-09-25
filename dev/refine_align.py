"""Refine an alignment by dynamic programming over score knots (every half quarter), all stems together.

For every knot the audio time is chosen from a band around the prior alignment (+-R s).  Between knots the map is
linear.  Gain of a segment, summed over the stems (each stem against its own score, same positions):
  * pitch: for every note starting in the segment, the audio's 3-harmonic salience z-score at its sounding pitch
    (natural-harmonic candidates: the best one), averaged over the first 0.25 s after the attack;
  * onset: attack strength (dynamics) x audio onset strength at the attack;
  * rests: where the score has a rest of every part, loud audio is penalised (reverb allowed: 0.3 s grace).
Penalty: mu * dq * log(local tempo / reference tempo)^2, the reference being the prior's tempo smoothed over
+-4 quarters (so libero passages and fermatas stay possible).  Returns a dense Alignment (single pass).

usage: python3 dev/refine_align.py SECTION [--R 8] [--mu 0.6] [--out PATH]
"""
import pickle, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from latent_space.audio_io import read_wav
from scoremap import align, mxl
import align_sections as AS, evidence as EV, tutti
from smooth_align import onset_strength


def rms_db(path, hop=0.01):
    x, info = read_wav(path); fs = int(info.sample_rate); x = x.astype(float).mean(1)
    h = int(hop * fs); w = int(0.04 * fs)
    c = np.concatenate([[0.0], np.cumsum(x ** 2)])
    idx = np.arange(0, len(x) - w, h)
    e = (c[idx + w] - c[idx]) / w
    return 20 * np.log10(np.sqrt(e) + 1e-9), hop


SAMPLE = 0.125        # score sampling step (quarters): every term is time-uniform in score time
GRACE = 2.0           # seconds of a rest in which the audio may still ring (reverb) without penalty


class Stem:
    def __init__(self, wav, sc):
        Z, _, self.dtz, self.lo = EV.salience(wav)
        n = max(1, int(round(0.05 / self.dtz)))                    # +-50 ms smoothing
        ker = np.ones(2 * n + 1) / (2 * n + 1)
        self.Z = np.apply_along_axis(lambda c: np.convolve(c, ker, "same"), 0, Z)
        self.O, self.dto, self.dur = onset_strength(wav)
        self.db, self.dtd = rms_db(wav)
        floor = np.percentile(self.db, 5)
        self.loud = np.clip((self.db - (floor + 18.0)) / 12.0, 0, 1.5)
        # level drop over the next 0.25 s (a note-off of everything): 10 dB -> 1
        k = int(round(0.25 / self.dtd))
        drop = np.zeros(len(self.db))
        drop[:-k] = self.db[:-k] - self.db[k:]
        self.off = np.clip(drop / 10.0, 0, 1.5) * (self.db > floor + 12.0)
        ev = align.score_events(sc)
        L = float(sc.length)
        self.qs = np.arange(0.0, L, SAMPLE) + SAMPLE / 2          # sample centres
        act = [[] for _ in self.qs]
        groups = {}
        for st, en, md, w, att in zip(ev.starts, ev.ends, ev.midis, ev.weights, ev.attack):
            kk = int(round(md)) - self.lo
            if 0 <= kk < Z.shape[1]:
                groups.setdefault((round(st, 4), round(en, 4)), []).append((kk, min(1.0, w + 0.3)))
        for (st, en), c in groups.items():
            i0 = int(np.floor(st / SAMPLE)); i1 = int(np.ceil(en / SAMPLE))
            for i in range(max(0, i0), min(len(self.qs), max(i1, i0 + 1))):
                act[i].append(c)
        self.act = act
        qa, wa = tutti.attack_strength(sc)
        self.att = np.zeros(len(self.qs))
        for q, w in zip(qa, wa):
            i = min(len(self.qs) - 1, int(q / SAMPLE))
            self.att[i] = max(self.att[i], w)
        # rest samples (no sounding note of any part) and the start of their rest
        sounding = np.array([len(a) > 0 for a in act])
        self.rest_from = np.full(len(self.qs), -1.0)
        start = None
        for i, snd in enumerate(sounding):
            if not snd:
                start = self.qs[i] - SAMPLE / 2 if start is None else start
                self.rest_from[i] = start
            else:
                start = None


def refine(sec, prior, R=8.0, step=0.04, mu=0.6, w_pitch=1.0, w_on=0.6, w_rest=1.5, w_off=1.0, stems=None, scores=None, wavs=None,
           prior_fn=None, spq_fn=None, time_weight=False):
    """spq_fn: reference seconds per quarter as a function of the score position (default: the prior's tempo,
    smoothed).  time_weight: pitch / rest terms weighted by audio time instead of score time (comparable totals for
    different unrollings)."""
    names = AS.SECTIONS.get(sec, [])
    if scores is None:
        stems = range(len(names)) if stems is None else stems
        scores = [mxl.parse(AS.S + names[k] + ".musicxml") for k in stems]
        wavs = [AS.A + names[k] + ".wav" for k in stems]
    S = [Stem(w, sc) for w, sc in zip(wavs, scores)]
    dur = min(s.dur for s in S)
    L = float(scores[0].length)
    knots = np.arange(0.0, L + 1e-9, 0.5)
    if knots[-1] < L - 1e-9:
        knots = np.append(knots, L)
    K = len(knots)
    # prior times and a smoothed reference tempo (seconds per quarter)
    tp = np.array([prior.t_at(q, 0) if prior_fn is None else prior_fn(q) for q in knots])
    spq = np.gradient(tp, knots)
    ker = np.ones(17) / 17.0
    spq_ref = np.convolve(np.pad(spq, 8, mode="edge"), ker, "valid")
    spq_ref = np.clip(spq_ref, 0.15, 8.0)
    if spq_fn is not None:
        spq_ref = np.array([spq_fn(q) for q in knots])
    cand = []
    for k in range(K):
        c = np.arange(max(-0.3, tp[k] - R), min(dur + 0.3, tp[k] + R) + 1e-9, step)
        cand.append(c)

    def seg_gain(k, ta, tb):
        q0, q1 = knots[k - 1], knots[k]
        g = np.zeros((ta.shape[0], tb.shape[1]))
        tw = ((tb - ta) / (q1 - q0)) if time_weight else 1.0     # seconds per quarter of this segment
        for s in S:
            i0 = np.searchsorted(s.qs, q0); i1 = np.searchsorted(s.qs, q1)
            for i in range(i0, i1):
                f = (s.qs[i] - q0) / (q1 - q0)
                t = ta + f * (tb - ta)
                if s.act[i]:
                    iz = np.clip((t / s.dtz).astype(int), 0, len(s.Z) - 1)
                    tot = 0.0
                    for cands in s.act[i]:
                        best = None
                        for kk, w in cands:
                            v = w * s.Z[iz, kk]
                            best = v if best is None else np.maximum(best, v)
                        tot = tot + np.clip(best, -1.5, 3.0)
                    g += w_pitch * SAMPLE * tw * tot / len(s.act[i]) / len(S)
                if s.att[i] > 0:
                    # attack at the start of the sample cell
                    ta_ = ta + ((s.qs[i] - SAMPLE / 2 - q0) / (q1 - q0)) * (tb - ta)
                    io = np.clip((ta_ / s.dto).astype(int), 0, len(s.O) - 1)
                    g += w_on * SAMPLE * 4 * s.att[i] * s.O[io] / len(S)
                if s.rest_from[i] >= 0:
                    tr = ta + ((max(s.rest_from[i], q0 - 8) - q0) / (q1 - q0)) * (tb - ta)
                    idb = np.clip((t / s.dtd).astype(int), 0, len(s.loud) - 1)
                    pen = np.where(t - tr < GRACE, 0.0, s.loud[idb])
                    g -= w_rest * SAMPLE * tw * pen / len(S)
                    if abs(s.rest_from[i] - (s.qs[i] - SAMPLE / 2)) < 1e-9:
                        # a rest of every part starts here: the audio level should fall (note-off)
                        io = np.clip((tr / s.dtd).astype(int), 0, len(s.off) - 1)
                        g += w_off * s.off[io] / len(S)
        return g

    V = -0.5 * (cand[0] / 1.0) ** 2 * 0                      # free start (pre-roll silence allowed)
    V = np.where(cand[0] < 0, -5.0, V)
    back = []
    for k in range(1, K):
        ta = cand[k - 1][:, None]; tb = cand[k][None, :]
        dq = knots[k] - knots[k - 1]
        d = tb - ta
        ok = d > 0.02
        r = np.where(ok, d / (dq * spq_ref[k - 1]), 1.0)
        pen = mu * dq * np.log(np.maximum(r, 1e-3)) ** 2
        tot = V[:, None] + seg_gain(k, ta, tb) - pen
        tot = np.where(ok, tot, -1e9)
        arg = tot.argmax(axis=0)
        V = tot[arg, np.arange(tot.shape[1])]
        back.append(arg)
    V = np.where(cand[-1] > dur + 0.05, V - 5.0, V)
    j = int(V.argmax())
    times = [cand[-1][j]]
    for k in range(K - 1, 0, -1):
        j = int(back[k - 1][j]); times.append(cand[k - 1][j])
    times = np.clip(np.array(times[::-1]), 0, dur)
    t_d = np.linspace(0, dur, 4000)
    q_d = np.interp(t_d, times, knots)
    return align.Alignment(t_d, q_d, np.zeros(len(t_d), dtype=int), float(-V.max()), 0.0, dur), knots, times


if __name__ == "__main__":
    sec = sys.argv[1]; a = sys.argv[2:]
    get = lambda k, d: float(a[a.index(k) + 1]) if k in a else d
    prior = pickle.load(open(a[a.index("--prior") + 1] if "--prior" in a else f"dev/out/align/{sec}.pkl", "rb"))
    if prior.seg.max() > 0:
        print("prior has repeat passes; not supported"); sys.exit(1)
    al, knots, times = refine(sec, prior, R=get("--R", 8.0), mu=get("--mu", 0.6))
    out = a[a.index("--out") + 1] if "--out" in a else f"dev/out/align/{sec}_refined.pkl"
    pickle.dump(al, open(out, "wb"))
    sc = mxl.parse(AS.S + AS.SECTIONS[sec][0] + ".musicxml")
    print(sec, "->", out)
    print("  " + " ".join(f"m{m.number}:{np.interp(float(m.start), knots, times):.1f}({prior.t_at(float(m.start), 0):.1f})" for m in sc.parts[0].measures))

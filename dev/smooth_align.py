"""Tempo-smooth alignment for sections played at a steady pulse (luminasity): the score time map is piecewise
linear with a knot at every quarter; knot times are chosen by dynamic programming to put the score's attacks
(weighted by how many parts attack and how loud) on strong audio onsets and the notes' sounding pitches on audible
pitch energy, while the local tempo stays near the section's average (penalty on log tempo ratio).

usage: python3 dev/smooth_align.py SECTION [--R 2.0] [--mu 8] [--nu 0.15]  -> dev/out/align/SECTION_smooth.pkl
"""
import pickle, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from latent_space.audio_io import read_wav
from scoremap import align, mxl
import align_sections as AS, evidence as EV, tutti


def onset_strength(path, hop=256, nfft=2048):
    x, info = read_wav(path); fs = int(info.sample_rate); x = x.astype(float).mean(1)
    pad = np.concatenate([np.zeros(nfft // 2), x, np.zeros(nfft // 2)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, nfft)[::hop]
    S = np.abs(np.fft.rfft(fr * np.hanning(nfft), axis=1))
    L = np.log1p(100 * S / S.max())
    fl = np.maximum(0, np.diff(L, axis=0, prepend=L[:1])).sum(1)
    dt = hop / fs
    k = np.exp(-0.5 * (np.arange(-6, 7) * dt / 0.015) ** 2); k /= k.sum()
    fl = np.convolve(fl, k, "same")
    fl = (fl - np.median(fl)) / (np.percentile(fl, 99) - np.median(fl) + 1e-9)
    fl = np.clip(fl, 0, 1.5)
    # tolerance: max over +-25 ms
    w = int(round(0.025 / dt))
    fm = np.array([fl[max(0, i - w):i + w + 1].max() for i in range(len(fl))])
    return fm, dt, len(x) / fs


def fit(sec, R=2.0, step=0.01, mu=8.0, nu=0.15, stem=0, t_start=None):
    names = AS.SECTIONS[sec]
    sc = mxl.parse(AS.S + names[stem] + ".musicxml")
    wav = AS.A + names[stem] + ".wav"
    O, dto, dur = onset_strength(wav)
    Z, _, dtz, lo = EV.salience(wav)
    L = float(sc.length)
    Tn = 60.0 * L / dur                                  # average tempo (quarters per minute)
    spq = 60.0 / Tn                                      # seconds per quarter
    qa, wa = tutti.attack_strength(sc)
    ev = align.score_events(sc)
    # note onsets with pitch (first 150 ms of every attack) for the pitch term
    npos = ev.starts[ev.attack]; nmid = ev.midis[ev.attack]; nw = ev.weights[ev.attack]
    knots = np.arange(0.0, np.floor(L) + 1.0)
    if knots[-1] < L - 1e-9:
        knots = np.append(knots, L)
    K = len(knots)
    cand = []
    for q in knots:
        c = q * spq + np.arange(-R, R + 1e-9, step)
        c = c[(c >= -0.6) & (c <= dur + 0.6)]
        cand.append(c)

    def seg_gain(k, ta, tb):
        """gain of segment [knot k-1, knot k] for all (ta, tb) pairs: arrays ta (n,1), tb (1,m)."""
        q0, q1 = knots[k - 1], knots[k]
        g = np.zeros((ta.shape[0], tb.shape[1]))
        sel = (qa >= q0 - 1e-9) & (qa < q1 - 1e-9) if k < K - 1 else (qa >= q0 - 1e-9) & (qa <= q1 + 1e-9)
        for q, w in zip(qa[sel], wa[sel]):
            f = (q - q0) / (q1 - q0)
            t = ta + f * (tb - ta)
            i = np.clip((t / dto).astype(int), 0, len(O) - 1)
            g += w * O[i]
        sel = (npos >= q0 - 1e-9) & (npos < q1 - 1e-9)
        for q, m, w in zip(npos[sel], nmid[sel], nw[sel]):
            kk = int(round(m)) - lo
            if not (0 <= kk < Z.shape[1]):
                continue
            f = (q - q0) / (q1 - q0)
            t = ta + f * (tb - ta) + 0.08
            i = np.clip((t / dtz).astype(int), 0, len(Z) - 1)
            g += nu * min(1.0, w + 0.3) * Z[i, kk]
        return g

    V = -3.0 * (cand[0] / 0.5) ** 2                      # the audio should start near the first barline
    back = []
    for k in range(1, K):
        ta = cand[k - 1][:, None]; tb = cand[k][None, :]
        dq = knots[k] - knots[k - 1]
        d = tb - ta
        ok = d > 0.25 * dq * spq
        r = np.where(ok, d / (dq * spq), 1.0)
        pen = mu * dq * np.log(r) ** 2
        tot = V[:, None] + seg_gain(k, ta, tb) - pen
        tot = np.where(ok, tot, -1e9)
        arg = tot.argmax(axis=0)
        V = tot[arg, np.arange(tot.shape[1])]
        back.append(arg)
    V = V - 3.0 * ((cand[-1] - dur) / 0.5) ** 2          # ... and end near the last barline
    j = int(V.argmax())
    times = [cand[-1][j]]
    for k in range(K - 1, 0, -1):
        j = int(back[k - 1][j])
        times.append(cand[k - 1][j])
    times = np.array(times[::-1])
    tt = np.clip(times, 0, dur); qq = knots.copy()
    # dense path
    t_d = np.linspace(0, dur, 2000)
    q_d = np.interp(t_d, tt, qq)
    al = align.Alignment(t_d, q_d, np.zeros(len(t_d), dtype=int), float(-V.max()), 0.0, dur)
    return al, knots, times, Tn


if __name__ == "__main__":
    sec = sys.argv[1]
    a = sys.argv[2:]
    get = lambda k, d: float(a[a.index(k) + 1]) if k in a else d
    al, knots, times, Tn = fit(sec, R=get("--R", 2.0), mu=get("--mu", 8.0), nu=get("--nu", 0.15))
    pickle.dump(al, open(f"dev/out/align/{sec}_smooth.pkl", "wb"))
    spq = 60 / Tn
    print(sec, f"avg ♩={Tn:.1f}")
    print("  knot q:  " + " ".join(f"{q:5.1f}" for q in knots))
    print("  t:       " + " ".join(f"{t:5.2f}" for t in times))
    print("  vs lin:  " + " ".join(f"{t - q * spq:+5.2f}" for q, t in zip(knots, times)))
    print("  ♩ local: " + " ".join(f"{60 * (knots[k] - knots[k - 1]) / (times[k] - times[k - 1]):5.0f}" for k in range(1, len(knots))))

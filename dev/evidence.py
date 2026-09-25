"""Independent-ish check of an alignment: for every score note (sounding pitch; harmonics resolved as in
scoremap.align), how prominent that exact pitch is in the audio during the note's aligned time span (z-score of a
3-harmonic pitch salience against all pitches in the same frames), and how strong the audio onset is at every aligned
attack.  Averages per bar.  Compare alignments of the same section bar by bar.

usage (module): bar_evidence(sec, alignment[, stem]) -> list of (bar, pitch_z, onset_z, n_notes)
"""
import sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from latent_space.audio_io import read_wav
from scoremap import align, mxl
import align_sections as AS

_cache = {}


def salience(path, hop=512, nfft=8192, lo=36, hi=108):
    if path in _cache:
        return _cache[path]
    x, info = read_wav(path)
    fs = int(info.sample_rate)
    x = x.astype(np.float64).mean(axis=1)
    pad = np.concatenate([np.zeros(nfft // 2), x, np.zeros(nfft // 2)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, nfft)[::hop]
    S = np.abs(np.fft.rfft(fr * np.hanning(nfft), axis=1))
    L = np.log1p(1000 * S / (S.max() + 1e-12))
    L = np.maximum(0.0, L - np.median(L, axis=0, keepdims=True))          # whiten: drop what is steady
    f = np.fft.rfftfreq(nfft, 1 / fs)
    mids = np.arange(lo, hi + 1)
    sal = np.zeros((len(fr), len(mids)))
    for k, m in enumerate(mids):
        for h, w in ((1, 1.0), (2, 0.6), (3, 0.4)):
            fc = h * 440 * 2 ** ((m - 69) / 12)
            if fc > fs / 2 - 100:
                continue
            b = fc / (fs / nfft)
            b0, b1 = int(np.floor(b * 2 ** (-0.35 / 12))), int(np.ceil(b * 2 ** (0.35 / 12)))
            sal[:, k] += w * L[:, max(1, b0):b1 + 1].max(axis=1)
    z = (sal - sal.mean(axis=1, keepdims=True)) / (sal.std(axis=1, keepdims=True) + 1e-9)
    flux = np.maximum(0.0, np.diff(L, axis=0, prepend=L[:1])).sum(axis=1)
    flux = (flux - np.median(flux)) / (np.percentile(flux, 90) - np.median(flux) + 1e-9)
    _cache[path] = (z, flux, hop / fs, lo)
    return _cache[path]


def bar_evidence(sec, al, stem=0, names=None, score=None):
    names = names or AS.SECTIONS[sec]
    sc = score or mxl.parse(AS.S + names[stem] + ".musicxml")
    z, flux, dt, lo = salience(AS.A + names[stem] + ".wav")
    ev = align.score_events(sc)
    bars = sc.parts[0].measures
    out = []
    for m in bars:
        a, b = float(m.start), float(m.start + m.length)
        pz, oz, n, wsum = 0.0, [], 0, 0.0
        best = {}
        for st, en, md, w, att in zip(ev.starts, ev.ends, ev.midis, ev.weights, ev.attack):
            if not (a <= st < b):
                continue
            ts, te = al.t_at(st, 0), al.t_at(min(en, b), 0)
            i0, i1 = int(ts / dt), max(int(ts / dt) + 1, int(te / dt))
            i1 = min(i1, len(z)); i0 = min(i0, i1 - 1)
            k = int(round(md)) - lo
            if not (0 <= k < z.shape[1]) or i1 <= i0:
                continue
            v = float(z[i0:i1, max(0, k - 0):k + 1].mean())
            key = (round(st, 4), round(en, 4), int(att))
            best[key] = max(best.get(key, -9), v * min(1.0, w + 0.3))       # natural-harmonic candidates: best one
            if att:
                j = int(ts / dt)
                oz.append(float(flux[max(0, j - 3):j + 4].max()))
        if best:
            vals = list(best.values())
            out.append((m.number, float(np.mean(vals)), float(np.mean(oz)) if oz else float("nan"), len(vals)))
        else:
            out.append((m.number, float("nan"), float("nan"), 0))
    return out


def linear(sec, stem=0):
    names = AS.SECTIONS[sec]
    sc = mxl.parse(AS.S + names[stem] + ".musicxml")
    x, info = read_wav(AS.A + names[stem] + ".wav")
    dur = len(x) / info.sample_rate
    L = float(sc.length)
    t = np.array([0.0, dur]); q = np.array([0.0, L])
    return align.Alignment(t, q, np.zeros(2, dtype=int), 0.0, 0.0, dur)

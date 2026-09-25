"""Monophonic pitch segments of a stem: f0 by harmonic summation every 20 ms, then stable-pitch segments
(>= min_len s within +-0.5 semitone) with their level.  Prints 'start-end MIDI(name) dB'.

usage: python3 dev/f0segments.py WAV [t0 t1] [--min 0.15]"""
import sys
import numpy as np
sys.path.insert(0, ".")
from latent_space.audio_io import read_wav

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def nm(m):
    m = int(round(m)); return NAMES[m % 12] + str(m // 12 - 1)


def f0_track(path, t0=0.0, t1=None, hop=0.02, nfft=8192, lo=36, hi=100):
    x, info = read_wav(path); fs = int(info.sample_rate); x = x.astype(np.float64).mean(axis=1)
    t1 = len(x) / fs if t1 is None else t1
    x = x[int(t0 * fs):int(t1 * fs)]
    h = int(hop * fs)
    pad = np.concatenate([np.zeros(nfft // 2), x, np.zeros(nfft // 2)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, nfft)[::h]
    S = np.abs(np.fft.rfft(fr * np.hanning(nfft), axis=1))
    L = np.log1p(1000 * S / (S.max() + 1e-12))
    cands = np.arange(lo, hi, 0.1); f0s = 440 * 2 ** ((cands - 69) / 12)
    H = np.zeros((len(fr), len(cands)))
    for k in range(1, 7):
        idx = np.clip(np.round(f0s * k / (fs / nfft)).astype(int), 0, S.shape[1] - 1)
        H += L[:, idx] * (0.8 ** (k - 1))
    # subharmonic penalty: prefer the candidate whose half-frequency is not supported
    best = cands[np.argmax(H, axis=1)]
    conf = H.max(axis=1) / (np.median(H, axis=1) + 1e-9)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    t = t0 + np.arange(len(fr)) * hop
    return t, best, conf, db


def segments(t, m, conf, db, min_len=0.15, db_min=-50, tol=0.5):
    out = []
    i = 0
    n = len(t)
    while i < n:
        if db[i] < db_min:
            i += 1; continue
        j = i + 1
        ref = m[i]
        while j < n and db[j] >= db_min and abs(m[j] - np.median(m[i:j])) <= tol:
            j += 1
        if t[j - 1] - t[i] + (t[1] - t[0]) >= min_len:
            out.append((t[i], t[j - 1] + (t[1] - t[0]), float(np.median(m[i:j])), float(np.max(db[i:j]))))
        i = j
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    path = a[0]
    t0 = float(a[1]) if len(a) > 1 and not a[1].startswith("--") else 0.0
    t1 = float(a[2]) if len(a) > 2 and not a[2].startswith("--") else None
    mn = float(a[a.index("--min") + 1]) if "--min" in a else 0.15
    t, m, c, d = f0_track(path, t0, t1)
    for s, e, mm, dd in segments(t, m, c, d, min_len=mn):
        print(f"{s:6.2f}-{e:6.2f} {mm:5.1f} {nm(mm):4s} {dd:5.0f}dB")

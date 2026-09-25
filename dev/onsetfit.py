"""Precise audio onsets (hop 128) and their residuals to aligned score attacks."""
import sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from latent_space.audio_io import read_wav
from scoremap import align, mxl
import align_sections as AS


def onsets(path, hop=128, nfft=1024, top=40, sep=0.08):
    x, info = read_wav(path); fs = int(info.sample_rate); x = x.astype(float).mean(1)
    pad = np.concatenate([np.zeros(nfft // 2), x, np.zeros(nfft // 2)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, nfft)[::hop]
    S = np.abs(np.fft.rfft(fr * np.hanning(nfft), axis=1))
    L = np.log1p(100 * S / S.max())
    fl = np.maximum(0, np.diff(L, axis=0, prepend=L[:1])).sum(1)
    fl = np.convolve(fl, np.ones(3) / 3, 'same')
    dt = hop / fs
    idx = [i for i in range(3, len(fl) - 3) if fl[i] == fl[i - 3:i + 4].max()]
    idx = sorted(idx, key=lambda i: -fl[i])
    keep = []
    for i in idx:
        if all(abs(i - j) * dt > sep for j in keep):
            keep.append(i)
        if len(keep) >= top:
            break
    keep = np.sort(np.array(keep))
    return keep * dt, fl[keep], len(x) / fs


def attacks(sc):
    ev = align.score_events(sc)
    return np.unique(np.round(ev.starts[ev.attack], 4))


def residuals(pk, ta):
    d = pk[:, None] - ta[None, :]
    k = np.abs(d).argmin(1)
    return d[np.arange(len(pk)), k]

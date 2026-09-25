"""Diagnostic image: audio pitchogram (log-frequency spectrogram, MIDI 36-100) with the score's sounding notes
overlaid through an alignment (green = attack, dim = tied continuation), measure starts as vertical lines with
their numbers.  numpy + zlib only (tiny PNG writer and 3x5 digit font).

usage: python3 dev/plotalign.py SECTION [t0 t1]   (SECTION as in dev/align_sections.py)
"""
import struct, sys, zlib
import numpy as np
sys.path.insert(0, ".")
from latent_space.audio_io import read_wav

FONT = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
        "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
        "8": "111101111101111", "9": "111101111001111", "m": "000111111101101", ".": "000000000000010",
        "s": "000011010001110"}


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def text(img, x, y, s, col, scale=2):
    for ch in s:
        g = FONT.get(ch)
        if g is None:
            x += 4 * scale; continue
        for r in range(5):
            for c in range(3):
                if g[r * 3 + c] == "1":
                    img[y + r * scale:y + (r + 1) * scale, x + c * scale:x + (c + 1) * scale] = col
        x += 4 * scale


def pitchogram(path, t0, t1, px_per_s, lo=36, hi=100, row=6):
    x, info = read_wav(path)
    fs = info.sample_rate
    x = x.astype(np.float64).mean(axis=1)[int(t0 * fs):int(t1 * fs)]
    nfft, hop = 8192, int(fs / px_per_s)
    pad = np.concatenate([np.zeros(nfft // 2), x, np.zeros(nfft // 2)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, nfft)[::hop]
    S = np.abs(np.fft.rfft(fr * np.hanning(nfft), axis=1))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    img = np.zeros(((hi - lo) * row, len(fr)))
    for k, m in enumerate(np.arange(lo, hi, 1.0 / row)):
        fc = 440 * 2 ** ((m - 69) / 12)
        b = int(round(fc / (fs / nfft)))
        if 0 < b < S.shape[1] - 1:
            img[(hi - lo) * row - 1 - k] = S[:, b - 1:b + 2].max(axis=1)
    img = np.log1p(3000 * img / (img.max() + 1e-12))
    img = img / img.max()
    return img


def render(out, audio, events, measures, t0, t1, px_per_s=40, lo=36, hi=100, row=6, extra_lines=()):
    """events: list of (t_start, t_end, midi, attack); measures: list of (t, label)."""
    g = pitchogram(audio, t0, t1, px_per_s, lo, hi, row)
    H, W = g.shape
    top = 24
    rgb = np.zeros((H + top, W, 3))
    rgb[top:] = (g[..., None] * np.array([235, 235, 235]))
    for (ts, te, m, attack) in events:
        if te < t0 or ts > t1 or not (lo <= m < hi):
            continue
        xa = int((ts - t0) * px_per_s); xb = max(xa + 1, int((te - t0) * px_per_s))
        y = int(top + round((hi - m - 1) * row + row // 2))
        col = np.array([60, 220, 90]) if attack else np.array([40, 120, 60])
        xa = max(0, xa); xb = min(W, xb)
        rgb[y - 1:y + 1, xa:xb] = 0.55 * rgb[y - 1:y + 1, xa:xb] + 0.45 * col
        if attack and 0 <= xa < W:
            rgb[y - 3:y + 3, xa:xa + 2] = col
    for (t, lab) in measures:
        x = int((t - t0) * px_per_s)
        if 0 <= x < W:
            rgb[top:, x] = rgb[top:, x] * 0.4 + np.array([240, 120, 40]) * 0.6
            text(rgb, min(W - 20, x + 2), 4, lab, np.array([240, 150, 60]))
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        x = int((s - t0) * px_per_s)
        if 0 <= x < W:
            rgb[top - 4:top, x] = 200
            if s % 5 == 0:
                text(rgb, x + 1, 14, str(s), np.array([180, 180, 255]), scale=1)
    for (t, col) in extra_lines:
        x = int((t - t0) * px_per_s)
        if 0 <= x < W:
            rgb[top:, x] = col
    for m in range(lo, hi):
        if m % 12 == 0:                                       # C lines
            y = top + (hi - m - 1) * row + row // 2
            rgb[y, :] = rgb[y, :] * 0.6 + np.array([90, 90, 160]) * 0.4
            text(rgb, 1, y - 5, str(m // 12 - 1), np.array([120, 120, 220]), scale=1)
    png(out, np.clip(rgb, 0, 255))

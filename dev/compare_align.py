"""Stacked panels on one time axis: the audio pitchogram, then the score's sounding notes warped by each of several
alignments (white = attack region, grey = sustain; orange = bar lines).  For judging which alignment is right.

usage (module): render(out, sec, [(label, Alignment), ...], t0, t1, px=70, lo=40, hi=104, row=4)
"""
import sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
import plotalign as PA
import align_sections as AS
from scoremap import align, mxl


def render(out, sec, aligns, t0, t1, px=70, lo=40, hi=104, row=4, stem=0, whiten=True):
    names = AS.SECTIONS[sec]
    sc = mxl.parse(AS.S + names[stem] + ".musicxml")
    g = PA.pitchogram(AS.A + names[stem] + ".wav", t0, t1, px, lo, hi, row)
    if whiten:
        g = np.maximum(0, g - np.median(g, axis=1, keepdims=True))
        g = g / (g.max() + 1e-9)
    H, W = g.shape
    ev = align.score_events(sc)
    panels = [np.repeat(g[..., None], 3, axis=2) * 255]
    for label, al in aligns:
        img = np.zeros((H, W, 3))
        for st, en, md, w, att in zip(ev.starts, ev.ends, ev.midis, ev.weights, ev.attack):
            if not (lo <= md < hi):
                continue
            ts, te = al.t_at(st, 0), al.t_at(en, 0)
            if te < t0 or ts > t1:
                continue
            xa = int((ts - t0) * px); xb = max(xa + 2, int((te - t0) * px))
            y = int((hi - md - 1) * row)
            xa2, xb2 = max(0, xa), min(W, xb)
            lev = 110 + 100 * min(1.0, w)
            img[y:y + row - 1, xa2:xb2] = np.maximum(img[y:y + row - 1, xa2:xb2], lev * 0.6)
            if att and 0 <= xa < W:
                img[y:y + row - 1, xa:min(W, xa + 5)] = 255
        for m in sc.parts[0].measures:
            x = int((al.t_at(float(m.start), 0) - t0) * px)
            if 0 <= x < W:
                img[:, x] = [240, 120, 40]
                PA.text(img, min(W - 30, x + 2), 2, "m" + m.number, np.array([240, 150, 60]))
        PA.text(img, 2, H - 12, label, np.array([120, 200, 255]))
        panels.append(img)
    sep = np.full((3, W, 3), 80.0)
    top = np.zeros((16, W, 3))
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        x = int((s - t0) * px)
        if 0 <= x < W:
            top[10:, x] = 200
            PA.text(top, x + 1, 1, str(s), np.array([180, 180, 255]), scale=1)
            for p in panels:
                p[:, x] = p[:, x] * 0.7 + 60 * 0.3
    stack = [top]
    for p in panels:
        # C lines
        for m in range(lo, hi):
            if m % 12 == 0:
                y = (hi - m - 1) * row + row // 2
                p[y, :] = p[y, :] * 0.7 + np.array([90, 90, 160]) * 0.3
        stack += [p, sep]
    PA.png(out, np.clip(np.concatenate(stack), 0, 255))
    return out

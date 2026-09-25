"""Audio-to-score alignment.

Audio features: chroma (log-magnitude spectrum folded onto 12 pitch classes, 80 Hz-5 kHz) and an onset curve
(positive spectral flux), hop 1024 samples at 44.1 kHz (23.2 ms).  Score features on a grid of score frames:
a chroma template of the sounding notes (pitch class + weak 5th and 3rd overtones; harmonics weighted lower)
and an onset template at note starts.  Alignment: dynamic time warping with slope-limited symmetric steps
(local tempo between 1/3 and 3 times the reference) and, for repeat sections, optional jumps from the section
end back to its start (ad-lib repeats are counted by the path itself).  Several synchronous audio stems of the
same passage can be aligned jointly (their costs are summed).
"""
import re
import numpy as np
from dataclasses import dataclass
from fractions import Fraction as Fr
from typing import List, Optional, Sequence, Tuple

from latent_space.audio_io import read_wav
from . import mxl

HOP = 1024
NFFT = 8192


# ------------------------------------------------------------------ audio
def audio_features(path: str):
    x, info = read_wav(path)
    fs = int(info.sample_rate)
    x = x.astype(np.float64).mean(axis=1)
    n = len(x)
    pad = np.concatenate([np.zeros(NFFT // 2), x, np.zeros(NFFT // 2)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, NFFT)[::HOP]
    win = np.hanning(NFFT)
    S = np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)
    sel = (freqs >= 80) & (freqs <= 5000)
    midi = 69 + 12 * np.log2(freqs[sel] / 440.0)
    W = np.zeros((sel.sum(), 12), dtype=np.float32)
    for pc in range(12):
        d = (midi - pc + 6) % 12 - 6                    # distance to the nearest pc in semitones
        W[:, pc] = np.exp(-0.5 * (d / 0.35) ** 2)
    L = np.log1p(1000.0 * S[:, sel] / (S[:, sel].max() + 1e-12))
    # spectral whitening: remove what is steady over the whole file (hum, drones, room noise)
    L = np.maximum(0.0, L - np.median(L, axis=0, keepdims=True) - 0.05)
    chroma = L @ W
    energy = np.sqrt((frames ** 2).mean(axis=1))
    ed = 20 * np.log10(energy + 1e-9)
    chroma = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9)
    LL = np.log1p(1000.0 * S / (S.max() + 1e-12))
    LL = np.maximum(0.0, LL - np.median(LL, axis=0, keepdims=True))
    flux = np.maximum(0.0, np.diff(LL, axis=0, prepend=LL[:1])).sum(axis=1)
    flux = np.convolve(flux, np.ones(3) / 3, mode="same")
    flux = (flux - np.median(flux)) / (np.percentile(flux, 95) - np.median(flux) + 1e-9)
    flux = np.clip(flux, 0, 3)
    # pitch salience (MIDI 36..107, 4 harmonics) and its increase = pitch-specific onsets
    Lw = np.maximum(0.0, LL - 0.0)
    df = fs / NFFT
    sal = np.zeros((len(frames), 72), dtype=np.float32)
    for k, pm in enumerate(range(36, 108)):
        f0 = 440.0 * 2 ** ((pm - 69) / 12.0)
        acc = np.zeros(len(frames), dtype=np.float32)
        for h, wh in ((1, 1.0), (2, 0.6), (3, 0.4), (4, 0.3)):
            b = int(round(f0 * h / df))
            if b + 1 < Lw.shape[1]:
                acc += wh * Lw[:, b - 1:b + 2].max(axis=1)
        sal[:, k] = acc
    ons = np.maximum(0.0, sal - np.maximum(np.roll(sal, 1, axis=0), np.roll(sal, 2, axis=0)))
    ons[:2] = 0
    ons = ons / (np.percentile(ons[ons > 0], 99) + 1e-9) if (ons > 0).any() else ons
    t = np.arange(len(frames)) * HOP / fs
    return {"t": t, "chroma": chroma, "flux": flux, "db": ed, "fs": fs, "n": n, "dur": n / fs, "pons": np.clip(ons, 0, 2)}


# ------------------------------------------------------------------ score
@dataclass
class ScoreEvents:
    starts: np.ndarray        # quarters (sounding start)
    ends: np.ndarray
    pcs: np.ndarray           # pitch class
    weights: np.ndarray
    attack: np.ndarray        # False for tied continuations (keep sounding, no new onset)
    length: float             # quarters
    midis: np.ndarray = None  # sounding pitch (MIDI, harmonics resolved)


OPEN_STRINGS = {"violin": [55, 62, 69, 76], "viola": [48, 55, 62, 69], "violoncello": [36, 43, 50, 57]}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def instrument_of(part_name: str) -> str:
    n = part_name.lower()
    if "cello" in n:
        return "violoncello"
    if "viola" in n:
        return "viola"
    return "violin"


def _string_words(text: str):
    """'II' -> [2]; '(I,II)' -> [1, 2]; 'III/IV' -> [3, 4]; 'IV.' -> [4]; else None."""
    t = text.strip().strip("().").replace(" ", "")
    parts = re.split(r"[,/]", t)
    if parts and all(x in ROMAN for x in parts):
        return [ROMAN[x] for x in parts]
    return None


def natural_harmonic_candidates(touch: float, opens: List[int], strings: Optional[List[int]] = None):
    """All sounding pitches a natural harmonic touched at `touch` can give: every (string, partial) whose node
    lies at the touch point (partials 2..8).  Weight 1 on the indicated string(s), 0.6 elsewhere; lower partials
    slightly preferred."""
    out = {}
    for si in range(1, len(opens) + 1):
        o = opens[len(opens) - si]
        for n in range(2, 9):
            for k in range(1, n):
                if np.gcd(n, k) != 1:
                    continue
                node = o + 12 * np.log2(n / (n - k))
                if abs(node - touch) < 0.4:
                    snd = round(o + 12 * np.log2(n))
                    w = (1.0 if (strings and si in strings) else 0.6) * (1.0 - 0.04 * (n - 2))
                    out[snd] = max(out.get(snd, 0.0), w)
    return sorted(out.items(), key=lambda x: -x[1])


ART = {12: 12, 7: 19, 5: 24, 4: 28, 3: 31}


def score_events(score: mxl.Score) -> ScoreEvents:
    """Sounding events of the score.  Diamond noteheads are harmonics: with a stopped note in the same chord
    they are artificial harmonics (sounding = stopped + partial), alone they are natural harmonics on the string
    named by the latest string indication (I = highest string)."""
    st, en, pc, w, at, md = [], [], [], [], [], []
    for part in score.parts:
        opens = OPEN_STRINGS[instrument_of(part.name)]
        cur_strings = {}                                   # staff -> [string numbers]
        chords = {}
        for e in part.elems:
            if e.kind == "note" and not e.rest:
                chords.setdefault((e.staff, e.voice, e.anchor if e.chord else e.idx), []).append(e)
        for e in part.elems:
            if e.kind == "direction":
                for wd in e.el.iter("words"):
                    ss = _string_words(wd.text or "")
                    if ss:
                        cur_strings[e.staff] = ss
                continue
            if e.kind != "note" or e.rest:
                continue
            m = mxl.midi_of(e.el)
            if m is None:
                continue
            nh = (e.el.findtext("notehead") or "").strip()
            diamond = nh == "diamond" or e.el.find("notations/technical/harmonic") is not None
            wt = 1.0
            sound = m
            extra = []
            if diamond:
                group = chords.get((e.staff, e.voice, e.anchor if e.chord else e.idx), [])
                stopped = [g for g in group if (g.el.findtext("notehead") or "").strip() != "diamond"
                           and g.el.find("notations/technical/harmonic") is None]
                if stopped:
                    base = min(mxl.midi_of(g.el) for g in stopped)
                    iv = int(round(m - base))
                    sound = base + ART.get(iv, 24)
                    wt = 1.0
                else:
                    cands = natural_harmonic_candidates(m, opens, cur_strings.get(e.staff))
                    if not cands:
                        sound, wt = m, 0.4
                    else:
                        sound, wt = cands[0]
                        extra = cands[1:4]
            if e.grace:
                start, dur = float(e.pos) - 0.08, 0.08
            else:
                start, dur = float(e.pos), float(e.dur)
            tie_stop = any(t.get("type") == "stop" for t in e.el.findall("tie"))
            st.append(max(0.0, start)); en.append(start + dur); pc.append(int(round(sound)) % 12)
            w.append(wt); at.append(not tie_stop); md.append(float(sound))
            for snd2, w2 in extra:                              # other possible partials of a natural harmonic
                st.append(max(0.0, start)); en.append(start + dur); pc.append(int(round(snd2)) % 12)
                w.append(w2); at.append(not tie_stop); md.append(float(snd2))
            if diamond and int(round(sound)) % 12 != int(round(m)) % 12:
                st.append(max(0.0, start)); en.append(start + dur); pc.append(int(round(m)) % 12)
                w.append(0.3); at.append(False); md.append(float(m))
    return ScoreEvents(np.array(st), np.array(en), np.array(pc), np.array(w), np.array(at, dtype=bool),
                       float(score.length), np.array(md))


def score_attacks(ev: ScoreEvents, q_per_frame: float, n_frames: int):
    """Per score frame: indicator matrix (frames x 72 pitches) of attacked sounding pitches (with the octave
    below/above at lower weight, for uncertain harmonic partials)."""
    A = np.zeros((n_frames, 72), dtype=np.float32)
    for s, p, w, attack in zip(ev.starts, ev.midis, ev.weights, ev.attack):
        if not attack or w < 0.35:
            continue
        j = int(np.floor(s / q_per_frame))
        if not (0 <= j < n_frames):
            continue
        k = int(round(p)) - 36
        for dk, dw in ((0, 1.0), (-12, 0.35), (12, 0.35)):
            if 0 <= k + dk < 72:
                A[j, k + dk] = max(A[j, k + dk], dw * w)
    return A


def score_features(ev: ScoreEvents, q_per_frame: float, n_frames: int):
    chroma = np.zeros((n_frames, 12), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    active = np.zeros(n_frames, dtype=np.float32)
    for s, e, p, w, attack in zip(ev.starts, ev.ends, ev.pcs, ev.weights, ev.attack):
        a = int(np.floor(s / q_per_frame))
        b = int(np.ceil(e / q_per_frame))
        a = max(0, a); b = min(n_frames, max(b, a + 1))
        chroma[a:b, p] += w
        chroma[a:b, (p + 7) % 12] += 0.35 * w
        chroma[a:b, (p + 4) % 12] += 0.15 * w
        active[a:b] = 1.0
        if attack and a < n_frames:
            onset[a] += w
    chroma = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9)
    # smear the onset impulses a little (about 60 ms at the reference tempo)
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2)
    onset = np.convolve(onset, k / k.sum(), mode="same")
    onset = onset / (np.percentile(onset[onset > 0], 95) + 1e-9) if (onset > 0).any() else onset
    return chroma, np.clip(onset, 0, 3), active


# ------------------------------------------------------------------ DTW
STEPS = [(1, 1, 0.0), (2, 1, 0.02), (1, 2, 0.02), (3, 1, 0.05), (1, 3, 0.05), (1, 0, 0.005)]


def cost_matrix(a_chroma, a_flux, a_db, s_chroma, s_onset, s_active, w_onset=0.5, silence_db=40.0,
                a_pons=None, s_att=None, w_pons=1.5, w_chroma=1.0):
    C = w_chroma * (1.0 - a_chroma @ s_chroma.T)              # cosine distance (both L2-normalised)
    C += w_onset * np.abs(a_flux[:, None] / 3.0 - s_onset[None, :] / 3.0)
    has = np.zeros(C.shape[1], dtype=bool)
    if a_pons is not None and s_att is not None:
        # score frames with attacks: reward audio onsets at those pitches; frames without: neutral
        has = s_att.sum(axis=1) > 0
        ev = a_pons @ s_att[has].T                            # audio frames x attack frames
        norm = s_att[has].sum(axis=1)[None, :]
        C[:, has] += w_pons * (1.0 - np.tanh(2.0 * ev / (norm + 1e-9)))
    # silent audio frames (pauses, decays) carry no pitch information: neutral against sounding notes,
    # a small bonus against rests, a mismatch against an attack
    silent = a_db < (np.percentile(a_db, 95) - silence_db)
    row = np.where(s_active > 0, 0.6, 0.3).astype(np.float32)
    row[has] = 0.6 + w_pons
    C[silent, :] = row[None, :]
    return C.astype(np.float32)


# stronger penalties on the non-diagonal steps (2026-09-25, prism): where the cost is flat (sustained notes) the
# default steps let the path race through the score and wait elsewhere almost for free; these make a steady
# tempo the cheapest path and fit the audio evidence better (pitch / onset z-scores up, see HANDOFF §12.5)
STEPS_STEADY = [(1, 1, 0.0), (2, 1, 0.80), (1, 2, 0.80), (3, 1, 2.40), (1, 3, 2.40), (1, 0, 0.20)]


def dtw(C: np.ndarray, jumps: Sequence[Tuple[int, int]] = (), jump_pen: float = 0.5,
        stay_pen: Optional[np.ndarray] = None, steps: Optional[Sequence[Tuple[int, int, float]]] = None):
    """Symmetric slope-limited DTW from (0,0) to (N-1,M-1).  `jumps`: (from_col, to_col) pairs meaning that after
    score frame from_col the path may continue at to_col (repeat back).  `steps`: (audio frames, score frames,
    penalty) step patterns (default STEPS)."""
    STEPS = list(steps) if steps is not None else globals()["STEPS"]
    N, M = C.shape
    INF = np.float32(1e30)
    D = np.full((N, M), INF, dtype=np.float32)
    B = np.full((N, M), -1, dtype=np.int8)
    D[0, 0] = 2 * C[0, 0]
    for i in range(1, N):
        best = np.full(M, INF, dtype=np.float32)
        arg = np.full(M, -1, dtype=np.int8)
        for k, (di, dj, pen) in enumerate(STEPS):
            if i - di < 0:
                continue
            prev = np.full(M, INF, dtype=np.float32)
            if dj == 0:
                prev[:] = D[i - di]
            else:
                prev[dj:] = D[i - di, :M - dj]
            add = np.zeros(M, dtype=np.float32)
            if dj == 0:                                    # the score waits (fermata, pause, held note)
                add = C[i].copy()
            elif di == 1 and dj == 1:
                add = 2 * C[i]
            elif dj == 1:                                  # several audio frames on one score frame
                add = C[i].copy()
                for r in range(1, di):
                    add += (2 if r == di - 1 else 1) * C[i - r]
            else:                                          # several score frames on one audio frame
                add = C[i].copy()
                for r in range(1, dj):
                    sh = np.full(M, INF, dtype=np.float32)
                    sh[r:] = C[i, :M - r]
                    add = add + (2 if r == dj - 1 else 1) * sh
            cand = prev + add + pen
            if stay_pen is not None and di > dj:
                # several audio frames on one score column: an attack happens once, so staying on an
                # attack column costs extra (waits belong before the attack)
                cand = cand + stay_pen * (di - dj)
            upd = cand < best
            best[upd] = cand[upd]
            arg[upd] = k
        for (fc, tc) in jumps:                             # repeat: from the end column back to the start column
            cand = D[i - 1, fc] + 2 * C[i, tc] + jump_pen
            if cand < best[tc]:
                best[tc] = cand
                arg[tc] = len(STEPS) + [j for j in jumps].index((fc, tc))
        D[i] = best
        B[i] = arg
    # backtrack
    path = [(N - 1, M - 1)]
    i, j = N - 1, M - 1
    while i > 0 or j > 0:
        k = int(B[i, j])
        if k < 0:
            raise RuntimeError(f"DTW backtrack failed at {(i, j)}")
        if k < len(STEPS):
            di, dj, _ = STEPS[k]
            i, j = i - di, j - dj
        else:
            fc, tc = jumps[k - len(STEPS)]
            i, j = i - 1, fc
        path.append((i, j))
    path.reverse()
    return np.array(path), float(D[N - 1, M - 1] / (N + M))


@dataclass
class Alignment:
    """Audio seconds <-> score position (quarters, notation coordinates).  `path_t`/`path_q` are the warping path
    points in performance order; `seg` numbers the passes (a new pass starts after every repeat jump)."""
    path_t: np.ndarray
    path_q: np.ndarray
    seg: np.ndarray
    cost: float
    q_per_frame: float
    audio_dur: float

    def q_at(self, t: float) -> Tuple[float, int]:
        """Score position and pass at audio time t (linear interpolation inside a pass)."""
        t = min(max(t, self.path_t[0]), self.path_t[-1])
        k = int(np.searchsorted(self.path_t, t, side="right") - 1)
        k = max(0, min(k, len(self.path_t) - 2))
        s = self.seg[k]
        if self.seg[k + 1] != s or self.path_t[k + 1] == self.path_t[k]:
            return float(self.path_q[k]), int(s)
        f = (t - self.path_t[k]) / (self.path_t[k + 1] - self.path_t[k])
        return float(self.path_q[k] + f * (self.path_q[k + 1] - self.path_q[k])), int(s)

    def t_at(self, q: float, seg: int) -> float:
        """Audio time at score position q inside pass `seg`.  On a DTW path (score frames of q_per_frame quarters)
        this is the moment the path enters the frame that contains q (a path may wait on a frame; the note at q
        starts when the frame is entered, not when it is left); on a dense path, linear interpolation."""
        m = self.seg == seg
        tq = self.path_q[m]; tt = self.path_t[m]
        if len(tq) == 0:
            return float("nan")
        if q <= tq[0]:
            return float(tt[0])
        if q >= tq[-1]:
            return float(tt[-1])
        if self.q_per_frame > 0:
            k = int(np.searchsorted(tq, q + 1e-9, side="right") - 1)   # last point whose frame starts <= q
            j = int(np.searchsorted(tq, tq[k], side="left"))          # first point of that frame
            if q < tq[j] + self.q_per_frame - 1e-9:
                f = (q - tq[j]) / self.q_per_frame
                t_next = tt[k + 1] if k + 1 < len(tt) else tt[k]
                return float(tt[j] + f * max(0.0, min(t_next - tt[j], (tt[j + 1] - tt[j]) if j + 1 < len(tt) else 0.0)))
        k = int(np.searchsorted(tq, q, side="right") - 1)
        if tq[k + 1] == tq[k]:
            return float(tt[k])
        f = (q - tq[k]) / (tq[k + 1] - tq[k])
        return float(tt[k] + f * (tt[k + 1] - tt[k]))

    def spans(self, t0: float, t1: float) -> List[Tuple[float, float, int]]:
        """Score spans (q_start, q_end, pass) heard between audio times t0 and t1, in performance order."""
        q0, s0 = self.q_at(t0)
        q1, s1 = self.q_at(t1)
        if s0 == s1:
            return [(q0, q1, s0)]
        out = []
        m0 = self.seg == s0
        out.append((q0, float(self.path_q[m0][-1]), s0))
        for s in range(s0 + 1, s1):
            m = self.seg == s
            out.append((float(self.path_q[m][0]), float(self.path_q[m][-1]), s))
        m1 = self.seg == s1
        out.append((float(self.path_q[m1][0]), q1, s1))
        return out


def align(scores: Sequence[mxl.Score], audios: Sequence[str], ref_tempo: Optional[float] = None,
          repeats: Sequence[Tuple[float, float]] = (), jump_pen: float = 0.5,
          anchors: Sequence[Tuple[float, float]] = (), w_onset: float = 0.5, w_pons: float = 1.5,
          w_chroma: float = 1.0, steps: Optional[Sequence[Tuple[int, int, float]]] = None) -> Alignment:
    """Align one or more synchronous audio stems to their scores (same notation positions).  `repeats`: score
    sections (start_q, end_q) that may be played more than once.  `ref_tempo` (quarters per minute) sets the score
    frame size; default = the average tempo implied by the notated length (no repeats)."""
    feats = [audio_features(a) for a in audios]
    N = min(len(f["t"]) for f in feats)
    dur = feats[0]["dur"]
    length = float(scores[0].length)
    if ref_tempo is None:
        ref_tempo = 60.0 * length / dur
    q_per_frame = ref_tempo / 60.0 * HOP / feats[0]["fs"]
    M = int(np.ceil(length / q_per_frame)) + 1
    PAD = int(round(1.0 / (HOP / feats[0]["fs"])))        # about one second of silence before and after
    C = None
    for sc, f in zip(scores, feats):
        ev = score_events(sc)
        s_chroma, s_onset, s_active = score_features(ev, q_per_frame, M)
        s_att = score_attacks(ev, q_per_frame, M)
        z = lambda a: np.concatenate([np.zeros((PAD,) + a.shape[1:], a.dtype), a, np.zeros((PAD,) + a.shape[1:], a.dtype)])
        s_chroma = z(s_chroma); s_chroma[:PAD] = 1.0 / np.sqrt(12); s_chroma[-PAD:] = 1.0 / np.sqrt(12)
        c = cost_matrix(f["chroma"][:N], f["flux"][:N], f["db"][:N], s_chroma, z(s_onset), z(s_active),
                        a_pons=f["pons"][:N], s_att=z(s_att), w_onset=w_onset, w_pons=w_pons, w_chroma=w_chroma)
        C = c if C is None else C + c
    C /= len(scores)
    att_cols = np.zeros(C.shape[1], dtype=np.float32)
    for sc in scores:
        a = score_attacks(score_events(sc), q_per_frame, M).sum(axis=1) > 0
        att_cols[PAD:PAD + M] = np.maximum(att_cols[PAD:PAD + M], a.astype(np.float32))
    stay_pen = 0.8 * att_cols
    jumps = [(PAD + min(M - 1, int(round(e / q_per_frame))), PAD + int(round(s / q_per_frame))) for (s, e) in repeats]
    if anchors:
        # anchored: fixed (audio time, score position) points; DTW only between consecutive anchors
        hop_s = HOP / feats[0]["fs"]
        pts = [(0, 0)] + sorted((int(round(t / hop_s)), PAD + int(np.floor(q / q_per_frame + 1e-9))) for t, q in anchors) \
              + [(N - 1, C.shape[1] - 1)]
        pieces, tot = [], 0.0
        for n_, ((i0, j0), (i1, j1)) in enumerate(zip(pts[:-1], pts[1:])):
            if i1 <= i0 or j1 < j0:
                raise ValueError(f"anchors not increasing: {(i0, j0)} -> {(i1, j1)}")
            # an anchor pins the moment the path ENTERS its score frame: the segment before it ends one audio
            # frame and one score frame earlier (the last segment ends at the corner itself)
            last = n_ == len(pts) - 2
            ie, je = (i1, j1) if last else (i1 - 1, max(j0, j1 - 1))
            if ie < i0:
                ie = i0
            sub = C[i0:ie + 1, j0:je + 1]
            sj = [(a - j0, b - j0) for a, b in jumps if j0 <= a <= je and j0 <= b <= je]
            pth, c = dtw(sub, sj, jump_pen, stay_pen[j0:je + 1], steps=steps)
            pth = pth + np.array([i0, j0])
            pieces.append(pth)
            tot += c * (sub.shape[0] + sub.shape[1])
        path = np.concatenate(pieces)
        cost = tot / (N + C.shape[1])
    else:
        path, cost = dtw(C, jumps, jump_pen, stay_pen, steps=steps)
    t = path[:, 0] * HOP / feats[0]["fs"]
    q = (path[:, 1] - PAD) * q_per_frame
    q = np.where(q < 0, -1e-3, np.minimum(q, length))      # leading padding just below 0: t_at(0) = entry of frame 0
    seg = np.zeros(len(path), dtype=int)
    for k in range(1, len(path)):
        seg[k] = seg[k - 1] + (1 if path[k, 1] < path[k - 1, 1] else 0)
    return Alignment(t, q, seg, cost, q_per_frame, dur)

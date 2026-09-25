"""Cut points: which source passage corresponds to an audio span, cut only where nothing breaks.

A clean position of a source score is a position where, across ALL parts/staves of the material, no note or
rest sounds across it and no hard group (beam, tuplet, two-note tremolo, glissando/slide) spans across it.  Soft
groups (ties, slurs, hairpins, trill lines, octave shifts) crossing it add a penalty.  For an audio span the
start / end cut points are the clean positions whose audio times (through the alignment) are closest to the
span's start / end, penalties included.
"""
import bisect
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Tuple

from . import mxl
from .align import Alignment

SOFT_WEIGHT = {"tie": 0.12, "slur": 0.35, "wedge": 0.15, "trill": 0.35, "octave": 0.3,
               "beam": 0.2, "tuplet": 0.5}          # beam / tuplet: only when cutting them is allowed (2026-09-25)


@dataclass
class CleanPoints:
    pos: List[Fr]                 # sorted clean positions
    penalty: Dict[Fr, float]      # soft penalty at each clean position

    def between(self, a: Fr, b: Fr) -> List[Fr]:
        i = bisect.bisect_left(self.pos, a)
        j = bisect.bisect_right(self.pos, b)
        return self.pos[i:j]


def clean_points(score: mxl.Score, pid: Optional[str] = None, cut_beams: bool = False, cut_tuplets: bool = False) -> CleanPoints:
    """Clean positions of the whole material (pid None) or of one part (all its staves).  With `cut_beams` /
    `cut_tuplets` (user rule 2026-09-25) a cut may fall inside a beam group (the notes are re-beamed) or inside a
    tuplet (the written notes get approximate plain values); both then count as soft penalties."""
    length = score.length
    cuttable = {k for k, ok in (("beam", cut_beams), ("tuplet", cut_tuplets)) if ok}
    cand = {Fr(0), length}
    spans = []                                       # open intervals that must not be cut
    for part in score.parts:
        if pid is not None and part.pid != pid:
            continue
        for m in part.measures:
            cand.add(m.start)
        for e in part.elems:
            if e.kind == "note" and not e.grace:
                cand.add(e.pos); cand.add(e.end)
                if e.dur > 0 and not e.chord:
                    spans.append((e.pos, e.end))
    for g in score.groups:
        if g.hard and g.kind not in cuttable and (pid is None or g.part == pid):
            spans.append((g.start, g.end))
    spans.sort()
    # a position p is blocked if some span has start < p < end
    blocked = set()
    cands = sorted(cand)
    for (s, e) in spans:
        i = bisect.bisect_right(cands, s)
        while i < len(cands) and cands[i] < e:
            blocked.add(cands[i]); i += 1
    # positions inside tuplets are blocked already; keep only notatable ones (multiples of a 256th note)
    pos = [p for p in cands if p not in blocked and Fr(0) <= p <= length and (p * 64).denominator == 1]
    pen = {p: 0.0 for p in pos}
    for g in score.groups:
        if (g.hard and g.kind not in cuttable) or (pid is not None and g.part != pid):
            continue
        w = SOFT_WEIGHT.get(g.kind, 0.2)
        i = bisect.bisect_right(pos, g.start)
        while i < len(pos) and pos[i] < g.end:
            pen[pos[i]] += w; i += 1
    return CleanPoints(pos, pen)


@dataclass
class Piece:
    a: Fr                  # source start (quarters)
    b: Fr                  # source end
    passno: int = 0        # repeat pass (0 = first)
    pid: str = ""          # source part
    ref: Fr = Fr(0)        # output offset of the piece start from the fragment reference (quarters)


@dataclass
class Option:
    """One way to notate a fragment in one part: pieces (one per pass) and their cost."""
    pieces: List[Piece]
    start: Fr              # output offset of the first piece start from the reference
    end: Fr                # output offset of the last piece end
    cost: float
    cut_from: Optional[Fr] = None     # the end the heard audio asked for, when this option ends earlier
    head_from: Optional[Fr] = None    # the start the heard audio asked for, when this option starts later
    t_start: float = 0.0              # audio time (file seconds) of the notated start
    t_end: float = 0.0                # audio time of the notated end
    penalty: float = 0.0              # soft-spanner penalty of both cuts


@dataclass
class Fragment:
    lane: int              # material lane index in the map (1..24)
    material: str          # file stem
    t0: float              # piece time (s) of the map bar
    t1: float
    s0: float              # file offsets (s) of the map bar
    s1: float
    pieces: List[Piece] = field(default_factory=list)
    ref_q: Fr = Fr(0)             # source position heard at the entry (the fragment reference)
    options: Dict[str, List[Option]] = field(default_factory=dict)   # per part, cheapest first
    drop_cost: Dict[str, float] = field(default_factory=dict)        # per part: leaving the fragment out

    def span(self) -> Tuple[Fr, Fr]:
        """Output extent relative to the reference."""
        return (min(p.ref for p in self.pieces), max(p.ref + (p.b - p.a) for p in self.pieces))


# costs (seconds of audio, weighted by how much of the part sounds there).  User decision 2026-09-24: entries are
# kept - when two fragments of one part collide, the earlier one is shortened; starting the later one after its heard
# entry is much dearer, and a fragment is left out only when every way to notate it costs more heard sound elsewhere.
W_PRE = 1.0            # notated before the heard entry (the notation starts early)
W_POST = 0.3           # notated after the heard end (the last unit is completed)
W_LOST_TAIL = 1.0      # heard but not notated at the end: the earlier fragment shortened
W_LOST_HEAD = 2.0      # heard but not notated at the start: the later fragment starts after its entry
W_DROP = 1.5           # a fragment missing entirely from a part: its heard seconds plus this, and never cheaper
                       # than its best option plus this
LAM_SOFT = 0.25        # per unit of soft-spanner penalty at a cut


class Activity:
    """Fraction of a source span in which a part has a sounding (non-rest) note."""

    def __init__(self, score: mxl.Score, pid: str):
        iv = sorted((e.pos, e.end) for p in score.parts if p.pid == pid for e in p.elems
                    if e.kind == "note" and not e.rest and not e.grace and e.dur > 0)
        merged = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.iv = merged
        self.starts = [s for s, e in merged]

    def frac(self, x: float, y: float) -> float:
        if y <= x:
            return 0.0
        tot = 0.0
        i = max(0, bisect.bisect_right(self.starts, x) - 1)
        while i < len(self.iv) and float(self.iv[i][0]) < y:
            s, e = float(self.iv[i][0]), float(self.iv[i][1])
            tot += max(0.0, min(e, y) - max(s, x))
            i += 1
        return tot / (y - x)

    def weight(self, x: float, y: float) -> float:
        lo, hi = min(x, y), max(x, y)
        return 0.2 + 0.8 * self.frac(lo, hi) if hi > lo else 1.0


def options(score: mxl.Score, cps: Dict[str, CleanPoints], acts: Dict[str, Activity], al: Alignment,
            s0: float, s1: float, n_out: int = 3) -> Tuple[Dict[str, List[Option]], Dict[str, float], Fr]:
    """All ways to notate the audio span [s0, s1] per part (cut only at the part's clean positions), with costs:
    heard-but-left-out seconds and notated-but-unheard seconds (activity weighted) plus soft-spanner cuts.  Several
    passes when the audio goes through a repeat (inner pass boundaries on the repeat barlines, fixed)."""
    spans = al.spans(s0, s1)
    if not spans:
        return {}, {}, Fr(0)
    qa0 = Fr(round(spans[0][0] * 8), 8)
    # output offsets of every span (inner boundaries snapped to barlines)
    bounds = []
    off = Fr(0)
    for k, (qa, qb, passno) in enumerate(spans):
        first, last = (k == 0), (k == len(spans) - 1)
        qa_f = qa0 if first else _nearest_bar(score, qa)
        qb_f = Fr(round(qb * 8), 8) if last else _nearest_bar(score, qb)
        bounds.append((qa_f, qb_f, passno, off))
        off += qb_f - qa_f
    opts: Dict[str, List[Option]] = {}
    drops: Dict[str, float] = {}
    for pid, cp in cps.items():
        act = acts[pid]
        (qa_f, _, pass0, off0) = bounds[0]
        (qa_l, qb_l, passL, offL) = bounds[-1]
        ta_heard, tb_heard = s0, s1
        q_heard0, q_heardL = spans[0][0], spans[-1][1]
        # candidate starts (first pass) and ends (last pass): every clean point whose audio time lies inside the
        # heard span, plus the three nearest outside it
        def around(passno, lo_t, hi_t, lo_q, hi_q):
            m = al.seg == passno
            q_lo = Fr(float(al.path_q[m].min())).limit_denominator(1 << 16)
            q_hi = Fr(float(al.path_q[m].max())).limit_denominator(1 << 16)
            pts = cp.between(max(Fr(0), q_lo - 16), q_hi + 16)
            return [(p, al.t_at(float(p), passno)) for p in pts]
        heads_all = around(pass0, s0, s1, None, None)
        tails_all = heads_all if passL == pass0 else around(passL, s0, s1, None, None)
        inside_h = [(p, t) for p, t in heads_all if s0 <= t < s1]
        before = [(p, t) for p, t in heads_all if t < s0][-3:]
        heads = before + inside_h
        inside_t = [(p, t) for p, t in tails_all if s0 < t <= s1]
        after = [(p, t) for p, t in tails_all if t > s1][:3]
        tails = inside_t + after
        # multi-pass: fixed inner pieces
        inner = []
        for (qa_f2, qb_f2, passno, off2) in bounds[1:-1]:
            a2 = qa_f2 if qa_f2 in cp.penalty else _nearest(cp, al, passno, al.t_at(float(qa_f2), passno), qa_f2 - 8, qa_f2 + 8)
            b2 = qb_f2 if qb_f2 in cp.penalty else _nearest(cp, al, passno, al.t_at(float(qb_f2), passno), qb_f2 - 8, qb_f2 + 8)
            if a2 is not None and b2 is not None and b2 > a2:
                inner.append(Piece(a2, b2, passno, pid, off2 + (a2 - qa_f2)))
        multi = len(bounds) > 1
        if multi:
            a_last = qa_l if qa_l in cp.penalty else _nearest(cp, al, passL, al.t_at(float(qa_l), passL), qa_l - 8, qa_l + 8)
            b_first = bounds[0][1] if bounds[0][1] in cp.penalty else _nearest(cp, al, pass0, al.t_at(float(bounds[0][1]), pass0), bounds[0][1] - 8, bounds[0][1] + 8)
        out = []
        # the "wanted" ends (cheapest head / tail on their own) for the labels
        def head_cost(p, t):
            if t <= ta_heard:
                return W_PRE * (ta_heard - t) * act.weight(float(p), q_heard0) + LAM_SOFT * cp.penalty.get(p, 0.0)
            return W_LOST_HEAD * (t - ta_heard) * act.weight(q_heard0, float(p)) + LAM_SOFT * cp.penalty.get(p, 0.0)

        def tail_cost(p, t):
            if t >= tb_heard:
                return W_POST * (t - tb_heard) * act.weight(q_heardL, float(p)) + LAM_SOFT * cp.penalty.get(p, 0.0)
            return W_LOST_TAIL * (tb_heard - t) * act.weight(float(p), q_heardL) + LAM_SOFT * cp.penalty.get(p, 0.0)
        hc = [(p, t, head_cost(p, t)) for p, t in heads]
        tc = [(p, t, tail_cost(p, t)) for p, t in tails]
        if not hc or not tc:
            continue
        best_h = min(hc, key=lambda x: x[2])[0]
        best_t = min(tc, key=lambda x: x[2])[0]
        for (a, ta, ca) in hc:
            for (b, tb, cb) in tc:
                if not multi:
                    if b <= a or ta >= s1 or tb <= s0 or tb <= ta:
                        continue
                    pcs = [Piece(a, b, pass0, pid, off0 + (a - qa_f))]
                    st, en = off0 + (a - qa_f), off0 + (b - qa_f)
                else:
                    if b_first is None or a_last is None or b_first <= a or b <= a_last:
                        continue
                    pcs = ([Piece(a, b_first, pass0, pid, off0 + (a - qa_f))] + inner +
                           [Piece(a_last, b, passL, pid, offL + (a_last - qa_l))])
                    st, en = off0 + (a - qa_f), offL + (b - qa_l)
                out.append(Option(pcs, st, en, ca + cb,
                                  cut_from=(best_t if b < best_t else None),
                                  head_from=(best_h if a > best_h else None),
                                  t_start=ta, t_end=tb, penalty=cp.penalty.get(a, 0.0) + cp.penalty.get(b, 0.0)))
        out.sort(key=lambda o: o.cost)
        if out:
            opts[pid] = out
            heard_w = (s1 - s0) * act.weight(q_heard0, q_heardL)
            drops[pid] = max(W_LOST_TAIL * heard_w, out[0].cost) + W_DROP
    return opts, drops, qa0


def _nearest(cp: CleanPoints, al: Alignment, passno: int, target_t: float, lo: Fr, hi: Fr,
             lam: float = 0.25, bias: float = 0.0) -> Optional[Fr]:
    best, bv = None, None
    for p in cp.between(lo, hi):
        t = al.t_at(float(p), passno)
        v = abs(t - target_t) + lam * cp.penalty.get(p, 0.0) + bias * (1 if t > target_t else 0)
        if bv is None or v < bv - 1e-12:
            best, bv = p, v
    return best


def _nearest_bar(score: mxl.Score, q: float) -> Fr:
    starts = [m.start for m in score.parts[0].measures] + [score.length]
    return min(starts, key=lambda x: abs(float(x) - q))


def measure_beat(score: mxl.Score, q: Fr) -> Tuple[str, Fr]:
    """(measure number, beat) of a position; beat is 1-based in quarters from the measure start."""
    ms = score.parts[0].measures
    for m in ms:
        if m.start <= q < m.start + m.length or (q == m.start + m.length and m is ms[-1]):
            return m.number, q - m.start + 1
    return ms[-1].number, q - ms[-1].start + 1

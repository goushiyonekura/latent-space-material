"""Place the extracted pieces on the output timeline and choose barlines.

Output tempo: quarter = 60, so output position (quarters) = piece seconds.  The source position heard at a
fragment's entry sits at its map time (rounded to GRID); where two fragments of one part would overlap, the earlier
one is shortened (or, failing that, the later one's start moved to a later clean position, or a fragment left out),
whichever leaves out the least heard sound (user decision 2026-09-24: entries are kept).  Barlines go only where no copied note,
rest, beam, tuplet, two-note tremolo or glissando is cut on any staff (ties avoided when possible); bar length is
kept near TARGET quarters.
"""
import bisect
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Dict, List, Tuple

from . import mxl
from .extract import Fragment, Piece

GRID = Fr(1, 8)
TARGET = Fr(4)


@dataclass
class Placed:
    frag: Fragment
    piece: Piece
    x: Fr                    # output position of the piece start
    score: mxl.Score         # the material's source score
    first: bool              # first piece of its fragment (gets the label)
    delay: float = 0.0       # how much later than wanted (s)
    cut_from: object = None  # the end the heard audio asked for (source quarters), when the piece ends earlier
    head_from: object = None # the start the heard audio asked for, when the piece starts later
    layer: int = 0           # voice layer (0 = the source's voices; >0 = overlapping an earlier fragment of the part)

    @property
    def end(self) -> Fr:
        return self.x + (self.piece.b - self.piece.a)

    def out(self, q: Fr) -> Fr:
        return self.x + (q - self.piece.a)


def snap(t: float) -> Fr:
    return Fr(round(t / float(GRID))) * GRID


def place(frags_by_lane: Dict[int, List[Fragment]], scores: Dict[int, mxl.Score], policy: str = "overlap") -> Tuple[List[Placed], List[tuple]]:
    """Every fragment keeps its reference at its map time (grid).

    policy "overlap" (user decision 2026-09-24, second revision): every fragment is notated in full with its
    cheapest option (nothing is shortened or left out); where fragments of one part overlap, the later one goes to
    another voice layer (`layer`, see assign_layers).

    policy "keep_entries" (the first revision, kept for comparison): per part, choose for every fragment one of its
    options (where to start and end in the source) or leave it out, so that no two fragments of the part overlap and
    the total cost (heard seconds left out, unheard seconds added, soft cuts, missing fragments) is least: dynamic
    programming over the fragments in time order.  Returns the placed pieces and the left-out (fragment, part)."""
    placed: List[Placed] = []
    dropped: List[tuple] = []
    if policy == "overlap":
        for lane, frags in frags_by_lane.items():
            for f in sorted(frags, key=lambda f: f.t0):
                w = snap(f.t0)
                for pid, opts in f.options.items():
                    o = opts[0]
                    first = True
                    for pc in o.pieces:
                        pl = Placed(f, pc, w + pc.ref, scores[lane], first, 0.0)
                        placed.append(pl)
                        first = False
        assign_layers(placed)
        return placed, dropped
    for lane, frags in frags_by_lane.items():
        frs = sorted(frags, key=lambda f: f.t0)
        want = [snap(f.t0) for f in frs]
        pids = sorted({pid for f in frs for pid in f.options})
        for pid in pids:
            # states: end of the last notated fragment -> (cost, back pointer)
            front = [(Fr(-10 ** 6), 0.0, None)]                     # (end, cost, node)
            for k, f in enumerate(frs):
                opts = f.options.get(pid, [])
                ends = [e for e, c, n in front]
                pre_min = []                                          # best cost among states ending <= x
                best = None
                for e, c, n in front:
                    if best is None or c < best[0]:
                        best = (c, n, e)
                    pre_min.append(best)
                new = []
                for o in opts:
                    st = want[k] + o.start
                    if st < 0:
                        continue
                    i = bisect.bisect_right(ends, st) - 1
                    if i < 0:
                        continue
                    c0, n0, e0 = pre_min[i]
                    new.append((want[k] + o.end, c0 + o.cost, (n0, k, o)))
                drop = f.drop_cost.get(pid, 0.0) if opts else 0.0
                for e, c, n in front:
                    new.append((e, c + drop, (n, k, None) if opts else n))
                # Pareto front: increasing end, strictly decreasing cost
                new.sort(key=lambda x: (x[0], x[1]))
                front = []
                for e, c, n in new:
                    if front and c >= front[-1][1] - 1e-12:
                        continue
                    front.append((e, c, n))
            node = min(front, key=lambda x: x[1])[2]
            chosen = {}
            while node is not None:
                prev, k, o = node
                chosen[k] = o
                node = prev
            for k, f in enumerate(frs):
                if pid not in f.options:
                    continue
                o = chosen.get(k)
                if o is None:
                    dropped.append((f, pid))
                    continue
                first = True
                for pc in o.pieces:
                    pl = Placed(f, pc, want[k] + pc.ref, scores[lane], first, 0.0)
                    pl.cut_from = o.cut_from
                    pl.head_from = o.head_from
                    placed.append(pl)
                    first = False
    return placed, dropped


def assign_layers(placed: List[Placed]) -> int:
    """Voice layer of every piece: per material part, the lowest layer that is free at the piece's start (interval
    colouring in time order).  Layer 0 = the source's own voices.  Returns the number of layers used."""
    by_part: Dict[tuple, List[Placed]] = {}
    for p in placed:
        by_part.setdefault((p.frag.lane, p.piece.pid), []).append(p)
    most = 1
    for key, ps in by_part.items():
        ends: List[Fr] = []                                   # end of the last piece per layer
        for p in sorted(ps, key=lambda p: (p.x, p.end)):
            for L, e in enumerate(ends):
                if e <= p.x:
                    p.layer = L
                    ends[L] = p.end
                    break
            else:
                p.layer = len(ends)
                ends.append(p.end)
        most = max(most, len(ends))
    return most


def forbidden(placed: List[Placed]) -> Tuple[List[Tuple[Fr, Fr]], List[Tuple[Fr, Fr]]]:
    """Open intervals of the output timeline that a barline must not cut (hard) or should not cut (ties)."""
    hard, soft = [], []
    for p in placed:
        a, b = p.piece.a, p.piece.b
        for part in p.score.parts:
            if part.pid != p.piece.pid:
                continue
            for e in part.elems:
                if e.kind == "note" and not e.grace and not e.chord and e.dur > 0 and a <= e.pos < b:
                    hard.append((p.out(e.pos), p.out(min(e.end, b))))
        for g in p.score.groups:
            if g.part != p.piece.pid:
                continue
            s, e = max(g.start, a), min(g.end, b)
            if e <= s:
                continue
            if g.hard:
                hard.append((p.out(s), p.out(e)))
            elif g.kind == "tie":
                soft.append((p.out(s), p.out(e)))
    return hard, soft


def _blocked(intervals: List[Tuple[Fr, Fr]]):
    iv = sorted(intervals)
    merged = []
    for s, e in iv:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    starts = [s for s, e in merged]

    def inside(x: Fr) -> bool:
        i = bisect.bisect_right(starts, x) - 1
        return i >= 0 and merged[i][0] < x < merged[i][1]
    return inside


def barlines(placed: List[Placed], total: Fr) -> List[Fr]:
    hard, soft = forbidden(placed)
    in_hard = _blocked(hard)
    in_tie = _blocked(soft)
    cand = {Fr(0), total}
    for p in placed:
        cand.add(p.x); cand.add(p.end)
        for part in p.score.parts:
            if part.pid != p.piece.pid:
                continue
            for e in part.elems:
                if e.kind == "note" and not e.grace and p.piece.a <= e.pos < p.piece.b:
                    cand.add(p.out(e.pos)); cand.add(p.out(min(e.end, p.piece.b)))
            for m in part.measures:
                if p.piece.a < m.start < p.piece.b:
                    cand.add(p.out(m.start))
    q = Fr(0)
    while q < total:
        cand.add(q); q += Fr(1, 2)
    cands = sorted(c for c in cand if Fr(0) <= c <= total and (c * 64).denominator == 1 and not in_hard(c))
    bars = [Fr(0)]
    while bars[-1] < total:
        x = bars[-1]
        window = [c for c in cands if x + 1 <= c <= x + 2 * TARGET]
        pref = [c for c in window if not in_tie(c)] or window
        if pref:
            # nearest to the target length, whole seconds preferred
            best = min(pref, key=lambda c: (abs(c - (x + TARGET)) + (0 if c.denominator == 1 else Fr(1, 4))))
        else:
            later = [c for c in cands if c > x]
            best = later[0] if later else total
        bars.append(best)
    if bars[-1] != total:
        bars[-1] = total
    if len(bars) > 2 and bars[-1] - bars[-2] < 1:
        del bars[-2]                                          # no tiny last bar: join it to the one before
    return bars

"""Staff reduction (user rule of 2026-09-25): a passage of another part (one piece = one fragment of one part) moves
WHOLE into the lower staff (staff 2) of a luminasity instrument of the same kind (violin / viola / cello), where
that instrument is silent for the whole written extent of the passage.  Luminasity rests there may be removed
(split where the passage covers only part of them), but a passage never sits inside a luminasity phrase (between
the first and the last note of a luminasity passage), and moved passages never overlap one another.  Notes and
beams are never changed; a source staff disappears only when every passage written on it has found a place.

plan() decides which source staves disappear (exact search per instrument kind, the largest number of staves) and
where every passage goes; among the feasible assignments it prefers (1) passages that touch no luminasity rest at
all and (2) consecutive passages of one source staff on the same target staff (few "switches").
"""
import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import mxl
from .layout import Placed

KIND = {"Violin": "violin", "Violin I": "violin", "Violin II": "violin", "Viola": "viola", "Viola I": "viola",
        "Viola II": "viola", "Violoncello": "cello", "Violoncello I": "cello", "Violoncello II": "cello"}
GRACE_ROOM = Fr(1, 8)          # a grace note needs a little room before its note
W_TOUCH = 2.0                  # cost of a moved passage that overlaps a luminasity passage (rests to remove)
W_SWITCH = 1.0                 # cost of two consecutive passages of one source staff on different targets
RESTARTS = 12                  # randomized restarts of the local search (fixed seeds: the result is reproducible)
PERTURB = 0.15                 # share of the passages reassigned at random before each restart


def kind_of(part_name: str) -> str:
    return KIND[part_name.strip()]


def is_luminasity(material: str) -> bool:
    return material.startswith("lumin")


def note_intervals(pl: Placed, staff: Optional[int] = None) -> List[Tuple[Fr, Fr]]:
    """Output-time intervals in which the piece has a sounding note (source staff `staff` only when given)."""
    out = []
    part = pl.score.part(pl.piece.pid)
    a, b = pl.piece.a, pl.piece.b
    for e in part.elems:
        if e.kind != "note" or e.rest or not (a <= e.pos < b) or (staff is not None and e.staff != staff):
            continue
        s = pl.out(e.pos)
        if e.grace:
            out.append((s - GRACE_ROOM, s))
        elif not e.chord:
            out.append((s, pl.out(min(e.end, b))))
    return out


def _overlaps(iv: Tuple[Fr, Fr], ivs: List[Tuple[Fr, Fr]]) -> bool:
    s, e = iv
    return any(s < b and a < e for a, b in ivs)


@dataclass
class Target:
    lane: int
    pid: str
    material: str
    name: str                                     # "luminasity_10 Viola"
    kind: str
    notes: List[Tuple[Fr, Fr]] = field(default_factory=list)      # sounding notes of the instrument (staff 2)
    phrases: List[Tuple[Fr, Fr]] = field(default_factory=list)    # first note .. end of last note, per passage
    extents: List[Tuple[Fr, Fr]] = field(default_factory=list)    # written extents of the luminasity passages


@dataclass
class Source:
    lane: int
    pid: str
    material: str
    name: str
    kind: str
    pieces: List[Placed] = field(default_factory=list)
    units: List[List[Placed]] = field(default_factory=list)     # the pieces of one fragment (repeat passes) move together


def unit_extent(u: List[Placed]) -> Tuple[Fr, Fr]:
    return (min(p.x for p in u), max(p.end for p in u))


@dataclass
class Plan:
    moves: Dict[int, Target]                      # id(Placed) -> target
    removed: List[Source]                         # source parts whose every passage moved
    stay: List[Source]                            # source parts that keep their own staff
    left_out: List[Placed]                        # rest-only passages of removed parts that fit nowhere
    touched: Dict[int, List[Tuple[Fr, Fr]]]       # id(Placed) -> luminasity extents it overlaps
    cost: Tuple[int, int]                         # (passages touching luminasity rests, switches)


def _short(material: str) -> str:
    from .emit import short_name
    return short_name(material)


def collect(placed: List[Placed], names: List[str]) -> Tuple[List[Target], List[Source]]:
    tg: Dict[tuple, Target] = {}
    sr: Dict[tuple, Source] = {}
    for pl in placed:
        mat = names[pl.frag.lane]
        part = pl.score.part(pl.piece.pid)
        key = (pl.frag.lane, pl.piece.pid)
        if is_luminasity(mat):
            t = tg.get(key)
            if t is None:
                t = tg[key] = Target(pl.frag.lane, pl.piece.pid, mat, f"{_short(mat)} {part.name}", kind_of(part.name))
            notes = note_intervals(pl, staff=2)
            t.notes += notes
            t.extents.append((pl.x, pl.end))
            if notes:
                t.phrases.append((min(s for s, e in notes), max(e for s, e in notes)))
        else:
            s = sr.get(key)
            if s is None:
                s = sr[key] = Source(pl.frag.lane, pl.piece.pid, mat, f"{_short(mat)} {part.name}", kind_of(part.name))
            s.pieces.append(pl)
    for s in sr.values():
        s.pieces.sort(key=lambda p: (p.x, p.end))
        by_frag: Dict[int, List[Placed]] = {}
        for p in s.pieces:
            by_frag.setdefault(id(p.frag), []).append(p)
        s.units = sorted(by_frag.values(), key=lambda u: unit_extent(u))
    return [tg[k] for k in sorted(tg)], [sr[k] for k in sorted(sr)]


def fits(t: Target, iv: Tuple[Fr, Fr]) -> bool:
    """The passage (written extent iv) may go to t: no luminasity note sounds in it and it lies inside no
    luminasity phrase (between the first and the last note of a passage)."""
    return not _overlaps(iv, t.notes) and not _overlaps(iv, t.phrases)


def _solve(pieces: List[Tuple[Tuple[Fr, Fr], int]], allowed: List[Tuple[int, ...]], n_t: int) -> Optional[List[int]]:
    """Exact feasibility: assign every piece (sorted by start) to an allowed target so that no two overlap on one
    target.  State: per target, the end of its last passage.  Returns the target index per piece."""
    @lru_cache(maxsize=None)
    def go(i, busy):
        if i == len(pieces):
            return ()
        (s, e), _ = pieces[i]
        for j in allowed[i]:
            if busy[j] <= s:
                r = go(i + 1, tuple((e if jj == j else (b if b > s else Fr(0))) for jj, b in enumerate(busy)))
                if r is not None:
                    return (j,) + r
        return None
    res = go(0, tuple([Fr(0)] * n_t))
    go.cache_clear()
    return None if res is None else list(res)


def _cost(assign: Dict[int, int], sources: List[Source], targets: List[Target]) -> Tuple[int, int]:
    touch = 0
    switches = 0
    for s in sources:
        prev = None
        for u in s.units:
            j = assign.get(id(u[0]))
            if j is None:
                continue
            if _overlaps(unit_extent(u), targets[j].extents):
                touch += 1
            if prev is not None and j != prev:
                switches += 1
            prev = j
    return touch, switches


def _improve(assign: Dict[int, int], sources: List[Source], targets: List[Target], allowed: Dict[int, Tuple[int, ...]],
             max_rounds: int = 100, rng: Optional[random.Random] = None) -> float:
    """Local search in place: move one passage at a time to another feasible target when the cost drops (the
    passages and the candidate targets in random order when `rng` is given).  Returns the final cost."""
    units = [u for s in sources for u in s.units if id(u[0]) in assign]
    on: Dict[int, List[List[Placed]]] = {}
    for u in units:
        on.setdefault(assign[id(u[0])], []).append(u)

    def total(a):
        t, sw = _cost(a, sources, targets)
        return W_TOUCH * t + W_SWITCH * sw
    cur = total(assign)
    for _ in range(max_rounds):
        improved = False
        order = units[:]
        if rng is not None:
            rng.shuffle(order)
        for u in order:
            k = id(u[0])
            j0 = assign[k]
            best, best_j = cur, j0
            cands = list(allowed[k])
            if rng is not None:
                rng.shuffle(cands)
            for j in cands:
                if j == j0 or _overlaps(unit_extent(u), [unit_extent(q) for q in on.get(j, []) if q is not u]):
                    continue
                assign[k] = j
                c = total(assign)
                if c < best - 1e-9:
                    best, best_j = c, j
            assign[k] = best_j
            if best_j != j0:
                on[j0].remove(u)
                on.setdefault(best_j, []).append(u)
                cur = best
                improved = True
        if not improved:
            break
    return cur


def _optimize(assign: Dict[int, int], sources: List[Source], targets: List[Target], allowed: Dict[int, Tuple[int, ...]]) -> Dict[int, int]:
    """The deterministic local search, then RESTARTS randomized ones from perturbed copies; the cheapest wins."""
    units = [u for s in sources for u in s.units if id(u[0]) in assign]
    best_cost = _improve(assign, sources, targets, allowed)
    best = dict(assign)
    for seed in range(RESTARTS):
        rng = random.Random(seed)
        a = dict(best)
        on: Dict[int, List[List[Placed]]] = {}
        for u in units:
            on.setdefault(a[id(u[0])], []).append(u)
        for u in units:
            if rng.random() < PERTURB:
                k = id(u[0])
                j0 = a[k]
                cands = [j for j in allowed[k] if j != j0
                         and not _overlaps(unit_extent(u), [unit_extent(q) for q in on.get(j, []) if q is not u])]
                if cands:
                    j = rng.choice(cands)
                    on[j0].remove(u)
                    on.setdefault(j, []).append(u)
                    a[k] = j
        c = _improve(a, sources, targets, allowed, rng=rng)
        if c < best_cost - 1e-9:
            best_cost, best = c, dict(a)
    assign.clear()
    assign.update(best)
    return assign


def plan(placed: List[Placed], names: List[str], allow=None) -> Plan:
    """allow(source) -> bool restricts which source parts may move (None: all)."""
    targets, sources = collect(placed, names)
    moves: Dict[int, Target] = {}
    removed: List[Source] = []
    stay: List[Source] = []
    left_out: List[Placed] = []
    assign: Dict[int, int] = {}
    for kind in ("violin", "viola", "cello"):
        tl = [j for j, t in enumerate(targets) if t.kind == kind]
        srcs = [s for s in sources if s.kind == kind and (allow is None or allow(s))]
        allowed: Dict[int, Tuple[int, ...]] = {}          # id(unit[0]) -> feasible targets
        for s in srcs:
            for u in s.units:
                allowed[id(u[0])] = tuple(j for j in tl if fits(targets[j], unit_extent(u)))
        # units without a note (rests only) are placed afterwards: they do not decide which staves disappear
        has_notes = {id(u[0]): any(note_intervals(pl) for pl in u) for s in srcs for u in s.units}
        best = None
        for r in range(len(srcs), 0, -1):
            for combo in itertools.combinations(srcs, r):
                us = sorted(((unit_extent(u), id(u[0])) for s in combo for u in s.units if has_notes[id(u[0])]))
                al = [tuple(tl.index(j) for j in allowed[i]) for iv, i in us]
                if any(not a for a in al):
                    continue
                res = _solve(us, al, len(tl))
                if res is not None:
                    best = (list(combo), us, [tl[j] for j in res])
                    break
            if best:
                break
        if not best:
            stay += [s for s in sources if s.kind == kind]
            continue
        combo, us, res = best
        for (iv, i), j in zip(us, res):
            assign[i] = j
        assign = _optimize(assign, combo, targets, allowed)
        # rest-only units of the chosen parts: anywhere they fit without overlapping a moved passage
        on: Dict[int, List[Tuple[Fr, Fr]]] = {}
        for s in combo:
            for u in s.units:
                if id(u[0]) in assign:
                    on.setdefault(assign[id(u[0])], []).append(unit_extent(u))
        for s in combo:
            prev = None
            for u in s.units:
                k = id(u[0])
                if k in assign:
                    prev = assign[k]
                    continue
                iv = unit_extent(u)
                cands = [j for j in allowed[k] if not _overlaps(iv, on.get(j, []))]
                cands.sort(key=lambda j: (0 if j == prev else 1, 1 if _overlaps(iv, targets[j].extents) else 0))
                if cands:
                    assign[k] = cands[0]
                    on.setdefault(cands[0], []).append(iv)
                    prev = cands[0]
                else:
                    left_out += u
        removed += combo
        stay += [s for s in sources if s.kind == kind and s not in combo]
    touched: Dict[int, List[Tuple[Fr, Fr]]] = {}
    for s in removed:
        for u in s.units:
            j = assign.get(id(u[0]))
            if j is None:
                continue
            for pl in u:
                moves[id(pl)] = targets[j]
                touched[id(pl)] = [iv for iv in targets[j].extents if iv[0] < pl.end and pl.x < iv[1]]
    return Plan(moves, removed, stay, left_out, touched, _cost(assign, removed, targets))


# ====================================================================== packing (voices per staff)
# User trial 2026-09-25: every passage (luminasity included) is written on synthetic staves of its instrument kind;
# a staff carries up to `cap` voices at once; passages that sound together on one staff must use the same clef;
# a luminasity passage occupies a two-staff part (its harmonic 0-line staff above the normal staff), other passages
# only a normal staff.  Nothing is cut; every passage keeps its own rests in its own voice.

@dataclass
class OutSpec:
    """One synthetic output part."""
    kind: str
    staves: int                                   # 2 = harmonic + normal (luminasity capable), 1 = normal only
    index: int                                    # number within its kind (1-based)
    units: List[List[Placed]] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (-1, f"S{ {'violin': 'vn', 'viola': 'va', 'cello': 'vc'}[self.kind] }{self.index}")


def _clef_segments(pl: Placed, staff: int) -> List[Tuple[Fr, Fr, tuple]]:
    """(t0, t1, clef key) of the piece on its source staff `staff`, in output time."""
    part = pl.score.part(pl.piece.pid)
    cl = [(e.pos, e.idx, c) for e in part.elems if e.kind == "attributes" for c in e.el.findall("clef")
          if int(c.get("number", "1")) == staff]
    cl.sort(key=lambda t: (t[0], t[1]))
    a, b = pl.piece.a, pl.piece.b
    cur = None
    for (q, i, c) in cl:
        if q <= a:
            cur = c
    from .emit import _clef_key
    segs = []
    t_prev, k_prev = pl.x, (_clef_key(cur) if cur is not None else None)
    for (q, i, c) in cl:
        if a < q < b:
            segs.append((t_prev, pl.out(q), k_prev)); t_prev, k_prev = pl.out(q), _clef_key(c)
    segs.append((t_prev, pl.end, k_prev))
    return segs


def _demand_segments(pl: Placed, staff: int) -> List[Tuple[Fr, Fr, int]]:
    """(t0, t1, number of source voices with a written element) of the piece on source staff `staff`."""
    part = pl.score.part(pl.piece.pid)
    a, b = pl.piece.a, pl.piece.b
    vocc: Dict[str, List[Tuple[Fr, Fr]]] = {}
    for e in part.elems:
        if e.kind == "note" and a <= e.pos < b and not e.chord and not e.grace and e.dur > 0 and e.staff == staff:
            vocc.setdefault(e.voice, []).append((pl.out(e.pos), pl.out(min(e.end, b))))
    pts = sorted({t for ivs in vocc.values() for iv in ivs for t in iv} | {pl.x, pl.end})
    out = []
    for t0, t1 in zip(pts[:-1], pts[1:]):
        n = sum(1 for ivs in vocc.values() if any(s <= t0 and t1 <= e for s, e in ivs))
        out.append((t0, t1, max(1, n)))
    return out


class Unit:
    """A fragment-part: the pieces of one fragment in one source part (repeat passes move together)."""

    def __init__(self, pieces: List[Placed], material: str):
        self.pieces = sorted(pieces, key=lambda p: p.x)
        self.material = material
        self.lum = is_luminasity(material)
        p0 = self.pieces[0]
        part = p0.score.part(p0.piece.pid)
        self.kind = kind_of(part.name)
        self.name = f"{_short(material)} {part.name}"
        self.x0 = min(p.x for p in self.pieces)
        self.x1 = max(p.end for p in self.pieces)
        nstaff = 2 if self.lum else 1                # the normal staff of the source
        self.clefs = [s for p in self.pieces for s in _clef_segments(p, nstaff)]
        self.demand = [s for p in self.pieces for s in _demand_segments(p, nstaff)]

    def clef_at(self, t: Fr):
        for (a, b, k) in self.clefs:
            if a <= t < b:
                return k
        return self.clefs[-1][2] if self.clefs else None

    def dem_at(self, t: Fr) -> int:
        for (a, b, n) in self.demand:
            if a <= t < b:
                return n
        return 1

    @property
    def extent(self) -> Tuple[Fr, Fr]:
        return (self.x0, self.x1)


def units_of(placed: List[Placed], names: List[str]) -> List[Unit]:
    by: Dict[tuple, List[Placed]] = {}
    for p in placed:
        by.setdefault((id(p.frag), p.piece.pid), []).append(p)
    return sorted((Unit(v, names[v[0].frag.lane]) for v in by.values()), key=lambda u: (u.x0, u.x1))


def _staff_ok(staff_units: List[Unit], u: Unit, cap: int, lum_cap: Optional[int], unit_cap: Optional[int] = None) -> bool:
    """u may join the staff: at every instant of its extent the voices stay <= cap, luminasity passages <= lum_cap,
    passages at once <= unit_cap (1 = one passage at a time, its own voices untouched), and every passage active
    there uses the same clef."""
    e0, e1 = u.extent
    act = [v for v in staff_units if v.x0 < e1 and e0 < v.x1]
    if not act:
        return True
    if unit_cap is not None and unit_cap <= 1:
        return False
    pts = {e0} | {t for v in act for t in (v.x0, v.x1) if e0 <= t < e1} | {t for v in act + [u] for s in v.clefs for t in s[:2] if e0 <= t < e1} \
        | {t for v in act + [u] for s in v.demand for t in s[:2] if e0 <= t < e1}
    for t in pts:
        here = [v for v in act if v.x0 <= t < v.x1]
        if not here:
            continue
        if u.dem_at(t) + sum(v.dem_at(t) for v in here) > cap:
            return False
        if lum_cap is not None and (1 if u.lum else 0) + sum(1 for v in here if v.lum) > lum_cap:
            return False
        if unit_cap is not None and 1 + len(here) > unit_cap:
            return False
        ck = u.clef_at(t)
        if any(v.clef_at(t) != ck for v in here):
            return False
    return True


def _pack_kind(units: List[Unit], cap: int, lum_cap: Optional[int], seed: int = 0, unit_cap: Optional[int] = None) -> List[OutSpec]:
    """Greedy packing of one kind: luminasity units into two-staff parts (as few as possible), then the other units
    into the remaining capacity or into one-staff parts; a randomized order with `seed` > 0."""
    rng = random.Random(seed)
    order = sorted(units, key=lambda u: (u.x0 + (Fr(rng.random()).limit_denominator(64) * Fr(1, 4) if seed else 0), u.x1))
    two: List[List[Unit]] = []
    for u in [v for v in order if v.lum]:
        cands = [k for k, st in enumerate(two) if _staff_ok(st, u, cap, lum_cap, unit_cap)]
        if cands:
            k = max(cands, key=lambda k: max(v.x1 for v in two[k]))       # best fit: the most recently busy part
            two[k].append(u)
        else:
            two.append([u])
    one: List[List[Unit]] = []
    for u in [v for v in order if not v.lum]:
        cands2 = [k for k, st in enumerate(two) if _staff_ok(st, u, cap, lum_cap, unit_cap)]
        cands1 = [k for k, st in enumerate(one) if _staff_ok(st, u, cap, lum_cap, unit_cap)]
        if cands2 or cands1:
            best = None
            for k in cands2:
                cand = (max(v.x1 for v in two[k]), 2, k)
                best = cand if best is None or cand > best else best
            for k in cands1:
                cand = (max(v.x1 for v in one[k]), 1, k)
                best = cand if best is None or cand > best else best
            (two if best[1] == 2 else one)[best[2]].append(u)
        else:
            one.append([u])
    # drain: a part whose units all fit elsewhere is dropped (smallest parts first, several rounds)
    for _ in range(4):
        moved_any = False
        for group, other_ok in ((one, lambda u: True), (two, lambda u: not u.lum)):
            for k in sorted(range(len(group)), key=lambda k: len(group[k])):
                src = group[k]
                if not all(other_ok(u) for u in src):
                    continue
                trial = [st[:] for st in two] + [st[:] for st in one]
                me = (0 if group is two else len(two)) + k
                ok = True
                for u in sorted(src, key=lambda u: u.x0):
                    cands = [j for j, st in enumerate(trial) if j != me and (u.lum <= (j < len(two))) and _staff_ok(st, u, cap, lum_cap, unit_cap)]
                    if not cands:
                        ok = False
                        break
                    j = max(cands, key=lambda j: max(v.x1 for v in trial[j]))
                    trial[j].append(u)
                if ok:
                    trial[me] = []
                    two[:] = [st for st in trial[:len(two)] if st]
                    one[:] = [st for st in trial[len(two):] if st]
                    moved_any = True
                    break
            if moved_any:
                break
        if not moved_any:
            break
    kind = units[0].kind if units else "violin"
    specs = [OutSpec(kind, 2, i + 1, [[p for p in u.pieces] for u in st]) for i, st in enumerate(two)]
    specs += [OutSpec(kind, 1, len(two) + i + 1, [[p for p in u.pieces] for u in st]) for i, st in enumerate(one)]
    return specs


def _switches(specs: List[OutSpec]) -> int:
    where: Dict[tuple, List[Tuple[Fr, int]]] = {}
    for k, sp in enumerate(specs):
        for u in sp.units:
            p0 = u[0]
            where.setdefault((p0.frag.lane, p0.piece.pid), []).append((p0.x, k))
    n = 0
    for seq in where.values():
        seq.sort()
        n += sum(1 for a, b in zip(seq, seq[1:]) if a[1] != b[1])
    return n


def pack(placed: List[Placed], names: List[str], cap: int = 3, lum_cap: Optional[int] = None,
         restarts: int = 24, unit_cap: Optional[int] = None) -> Tuple[List[OutSpec], dict]:
    """All units of every kind onto synthetic parts; the packing with the fewest staves (then the fewest
    material switches) over `restarts` randomized greedy runs."""
    us = units_of(placed, names)
    specs_all: List[OutSpec] = []
    stats = {}
    for kind in ("violin", "viola", "cello"):
        ku = [u for u in us if u.kind == kind]
        if not ku:
            continue
        best = None
        for seed in range(restarts):
            specs = _pack_kind(ku, cap, lum_cap, seed, unit_cap)
            score = (sum(sp.staves for sp in specs), _switches(specs))
            if best is None or score < best[0]:
                best = (score, specs)
        specs_all += best[1]
        stats[kind] = {"staves": best[0][0], "parts": len(best[1]), "two_staff_parts": sum(1 for sp in best[1] if sp.staves == 2),
                       "switches": best[0][1], "units": len(ku)}
    return specs_all, stats

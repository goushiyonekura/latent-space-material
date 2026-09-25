"""Write the score mapping as MusicXML (partwise 3.1).

Copied source elements (<note>, <direction>, clef changes) are deep copies of the originals.  The only edits:
<duration> / <offset> rescaled to the common divisions, `default-x` removed (horizontal layout of the source page),
instrument ids renamed to the output part.  Metronome marks of the sources become plain text so that the output
keeps quarter = 60.  Rests are written only where the material does not sound; inside a copied passage gaps stay
invisible (<forward>).  A slur / hairpin / octave line / trill line cut by a passage boundary is closed at the
boundary.  Each passage gets a small label (source, bar/beat, dynamic and technique in effect).
"""
import copy
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Tuple

from . import mxl
from .extract import measure_beat
from .layout import Placed

DIV = 80640
TECH_RE = re.compile(r"^(pizz\.?|arco|sul pont\.?|sul ord\.?|sul tasto|S\.T\.|S\.P\.|Ord\.|N\.?|col legno.*|jet[eé].*|fast bow)$", re.I)


@dataclass
class OutPart:
    pid: str
    lane: int
    material: str
    short: str
    src_pid: str
    score: mxl.Score
    part: mxl.Part
    first_of_material: bool
    staves: int
    main_voice: Dict[int, str] = field(default_factory=dict)
    max_voice: int = 1                                        # highest voice number of the source part
    max_simultaneous: int = 0                                 # most voices sounding at once on one staff
    font_scale: float = 1.0                                   # output staff height / source staff height
    notes_cut: int = 0                                        # notes written as tied notes because of a barline
    beams_cut: int = 0                                        # beam groups divided by a barline
    layout: object = None
    clef_conflicts: List[tuple] = field(default_factory=list)  # (x, staff): a clef change while another layer sounds


# ------------------------------------------------------------------ helpers
def _child(el: ET.Element, tag: str) -> ET.Element:
    c = el.find(tag)
    if c is None:
        c = ET.SubElement(el, tag)
    return c


def _strip_x(el: ET.Element):
    for x in el.iter():
        x.attrib.pop("default-x", None)


def _rescale(el: ET.Element, src_div: int):
    f = Fr(DIV, src_div)
    for tag in ("duration",):
        d = el.find(tag)
        if d is not None and d.text:
            d.text = str(int(Fr(int(d.text)) * f))
    off = el.find("offset")
    if off is not None and off.text:
        off.text = str(int(Fr(int(off.text)) * f))


def _div_at(part: mxl.Part, pos: Fr) -> int:
    d = part.divisions[0][1]
    for p, dd in part.divisions:
        if p <= pos:
            d = dd
    return d


def rest_values(length: Fr, start: Fr) -> List[Tuple[Fr, str, int]]:
    """Split a rest of `length` quarters starting at `start` (bar-relative) into rests that each start at a
    multiple of their own value (so they never straddle a beat of their size): (duration, type, dots)."""
    TYPES = [(Fr(4), "whole"), (Fr(2), "half"), (Fr(1), "quarter"), (Fr(1, 2), "eighth"), (Fr(1, 4), "16th"),
             (Fr(1, 8), "32nd"), (Fr(1, 16), "64th"), (Fr(1, 32), "128th"), (Fr(1, 64), "256th")]
    out = []
    pos, end = start, start + length
    while pos < end:
        for v, name in TYPES:
            if v <= end - pos and (pos / v).denominator == 1:
                out.append((v, name, 0))
                pos += v
                break
        else:
            raise ValueError(f"rest at {pos} (length {end - pos}) not notatable")
    return out


NOTE_VALUES = [(Fr(4) * f, name, d) for (f, d) in ((Fr(7, 4), 2), (Fr(3, 2), 1), (Fr(1), 0))
               for name in ("whole",)] + \
    [(v * f, name, d) for (v, name) in ((Fr(2), "half"), (Fr(1), "quarter"), (Fr(1, 2), "eighth"), (Fr(1, 4), "16th"),
                                        (Fr(1, 8), "32nd"), (Fr(1, 16), "64th"), (Fr(1, 32), "128th"), (Fr(1, 64), "256th"))
     for (f, d) in ((Fr(7, 4), 2), (Fr(3, 2), 1), (Fr(1), 0))]
NOTE_VALUES.sort(key=lambda x: -x[0])


def note_values(length: Fr) -> List[Tuple[Fr, str, int]]:
    """A note length as tied written values, longest first (dotted values allowed): (duration, type, dots)."""
    out, rest = [], length
    while rest > 0:
        for v, name, dots in NOTE_VALUES:
            if v <= rest:
                out.append((v, name, dots)); rest -= v
                break
        else:
            raise ValueError(f"note length {length} not notatable")
    return out


NOTE_ORDER = ["grace", "cue", "chord", "pitch", "unpitched", "rest", "duration", "tie", "instrument", "footnote", "level",
              "voice", "type", "dot", "accidental", "time-modification", "stem", "notehead", "notehead-text", "staff",
              "beam", "notations", "lyric", "play", "listen"]


def _reorder(n: ET.Element):
    kids = list(n)
    for k in kids:
        n.remove(k)
    kids.sort(key=lambda k: NOTE_ORDER.index(k.tag) if k.tag in NOTE_ORDER else len(NOTE_ORDER))
    for k in kids:
        n.append(k)


def split_note(el: ET.Element, parts: List[Tuple[Fr, str, int]], is_rest: bool) -> List[ET.Element]:
    """A copied note or rest written as several notes (a barline falls inside it): the notes are tied (rests
    are not), the first one keeps the attack marks (articulations, slur / line starts, dynamics, accidental), the
    last one keeps what belongs to the end (a tie to the next note, a glissando / slide start, a fermata);
    noteheads, harmonics and single-note tremolo strokes stay on every note.  Pitch and total length are unchanged."""
    out = []
    n = len(parts)
    had_tie_start = any(t.get("type") == "start" for t in el.findall("tie"))
    had_tie_stop = any(t.get("type") == "stop" for t in el.findall("tie"))
    for k, (v, typ, dots) in enumerate(parts):
        first, last = k == 0, k == n - 1
        e = copy.deepcopy(el)
        for tag in ("duration", "type", "dot", "tie"):
            for x in e.findall(tag):
                e.remove(x)
        ET.SubElement(e, "duration").text = str(int(v * DIV))
        ET.SubElement(e, "type").text = typ
        for _ in range(dots):
            ET.SubElement(e, "dot")
        if not is_rest:
            if not first:
                for x in e.findall("accidental"):
                    e.remove(x)
            stop = had_tie_stop if first else True
            start = had_tie_start if last else True
            if stop:
                ET.SubElement(e, "tie", type="stop")
            if start:
                ET.SubElement(e, "tie", type="start")
            for nt in list(e.findall("notations")):
                for x in list(nt):
                    keep = True
                    if x.tag == "tied":
                        keep = False
                    elif x.tag in ("glissando", "slide") and x.get("type") == "start":
                        keep = last
                    elif x.tag in ("glissando", "slide") and x.get("type") == "stop":
                        keep = first
                    elif x.tag == "fermata":
                        keep = last
                    elif x.tag == "technical":
                        if not first:
                            for y in list(x):
                                if y.tag != "harmonic":
                                    x.remove(y)
                            keep = len(x) > 0
                    elif x.tag == "ornaments":
                        if not first:
                            for y in list(x):
                                if not (y.tag == "tremolo" and y.get("type", "single") == "single"):
                                    x.remove(y)
                            keep = len(x) > 0
                    elif not first:
                        keep = False
                    if not keep:
                        nt.remove(x)
                if stop:
                    ET.SubElement(nt, "tied", type="stop")
                if start:
                    ET.SubElement(nt, "tied", type="start")
                if len(nt) == 0:
                    e.remove(nt)
            if not e.findall("notations") and (stop or start):
                nt = ET.SubElement(e, "notations")
                if stop:
                    ET.SubElement(nt, "tied", type="stop")
                if start:
                    ET.SubElement(nt, "tied", type="start")
        else:
            for x in e.findall("notations"):
                if not first:
                    e.remove(x)
        _reorder(e)
        out.append(e)
    return out


def split_at_barlines(evs: List["Ev"], bars: List[Fr]) -> Tuple[List["Ev"], int]:
    """Copied notes / rests that a barline cuts become tied notes / several rests (with their chord members).
    Returns the new events and the number of notes cut."""
    import bisect as _b
    inner = bars[1:-1]
    out: List[Ev] = []
    n_cut = 0
    i = 0
    while i < len(evs):
        e = evs[i]
        is_main = (e.kind == "note" and e.dur > 0 and e.el.find("chord") is None and e.el.find("grace") is None)
        if not is_main:
            out.append(e); i += 1
            continue
        k = _b.bisect_right(inner, e.pos)
        cuts = [b for b in inner[k:] if b < e.pos + e.dur]
        members = []
        j = i + 1
        while j < len(evs) and evs[j].kind == "note" and evs[j].el.find("chord") is not None \
                and evs[j].pos == e.pos and evs[j].staff == e.staff and evs[j].voice == e.voice:
            members.append(evs[j]); j += 1
        if not cuts:
            out.append(e); out.extend(members); i = j
            continue
        is_rest = e.el.find("rest") is not None
        n_cut += 0 if is_rest else 1
        pts = [e.pos] + cuts + [e.pos + e.dur]
        seg_parts = []
        for s0, s1 in zip(pts[:-1], pts[1:]):
            seg_parts += [(s0, v) for v in note_values(s1 - s0)]
        # positions of the written notes
        pos, vals = [], []
        for s0, v in seg_parts:
            pos.append(pos[-1] + vals[-1][0] if vals else e.pos)
            vals.append(v)
        mains = split_note(e.el, vals, is_rest)
        chords = [split_note(m.el, vals, is_rest) for m in members]
        for k2, (p_, v) in enumerate(zip(pos, vals)):
            out.append(Ev(p_, e.staff, e.voice, "note", mains[k2], v[0], order=e.order + (k2,)))
            for m, parts_m in zip(members, chords):
                out.append(Ev(p_, m.staff, m.voice, "note", parts_m[k2], Fr(0), order=m.order + (k2,)))
        i = j
    return out, n_cut


def _drop_tie(el: ET.Element, typ: str):
    for t in list(el.findall("tie")):
        if t.get("type") == typ:
            el.remove(t)
    for nt in list(el.findall("notations")):
        for t in list(nt.findall("tied")):
            if t.get("type") == typ:
                nt.remove(t)
        if len(nt) == 0:
            el.remove(nt)


def rebeam_at_barlines(evs: List["Ev"], bars: List[Fr]) -> int:
    """A beam group that a barline cuts becomes one group on each side (the notes and their values are unchanged;
    a side with a single note loses its beams and shows its flag; a secondary beam cut in two becomes a hook when
    one note is left on a side).  Returns the number of groups cut."""
    inner = bars[1:-1]
    streams: Dict[tuple, List[Ev]] = defaultdict(list)
    for e in evs:
        if e.kind == "note" and e.el.find("chord") is None and e.el.find("grace") is None and e.dur > 0:
            streams[(e.staff, e.voice)].append(e)
    n_cut = 0

    def lv(el, level):
        for b in el.findall("beam"):
            if b.get("number", "1") == str(level):
                return b
        return None
    for key, seq in streams.items():
        seq.sort(key=lambda e: e.pos)
        group: List[Ev] = []
        groups = []
        for e in seq:
            b1 = lv(e.el, 1)
            v = (b1.text or "").strip() if b1 is not None else ""
            if v == "begin":
                group = [e]
            elif v == "continue" and group:
                group.append(e)
            elif v == "end" and group:
                group.append(e); groups.append(group); group = []
        for g in groups:
            cuts = [b for b in inner if g[0].pos < b <= g[-1].pos]
            if not cuts:
                continue
            n_cut += 1
            segs, cur = [], []
            for e in g:
                if cur and any(cur[-1].pos < b <= e.pos for b in cuts):
                    segs.append(cur); cur = []
                cur.append(e)
            segs.append(cur)
            levels = max(len(e.el.findall("beam")) for e in g)
            for si, sg in enumerate(segs):
                for level in range(1, levels + 1):
                    # runs of notes that carried this beam level (not hooks) in the original group
                    runs, run = [], []
                    for e in sg:
                        b = lv(e.el, level)
                        v = (b.text or "").strip() if b is not None else ""
                        if v in ("begin", "continue", "end"):
                            run.append(e)
                            if v == "end":
                                runs.append(run); run = []
                        else:
                            if run:
                                runs.append(run); run = []
                    if run:
                        runs.append(run)
                    for r in runs:
                        if len(r) >= 2:
                            for k2, e in enumerate(r):
                                lv(e.el, level).text = "begin" if k2 == 0 else ("end" if k2 == len(r) - 1 else "continue")
                        else:
                            e = r[0]
                            if level == 1 or len(sg) == 1:
                                for b in list(e.el.findall("beam")):
                                    e.el.remove(b)
                            else:
                                lv(e.el, level).text = "backward hook" if e is sg[-1] else "forward hook"
    return n_cut


def _rest_el(dur: Fr, typ: str, staff: int, voice: str, measure_rest: bool = False) -> ET.Element:
    n = ET.Element("note")
    r = ET.SubElement(n, "rest")
    if measure_rest:
        r.set("measure", "yes")
    ET.SubElement(n, "duration").text = str(int(dur * DIV))
    ET.SubElement(n, "voice").text = voice
    if not measure_rest:
        ET.SubElement(n, "type").text = typ
    ET.SubElement(n, "staff").text = str(staff)
    return n


def _words(text: str, staff: int, voice: str, size: str = "7", placement: str = "above", style: str = "normal",
           weight: str = "normal", enclosure: Optional[str] = None) -> ET.Element:
    d = ET.Element("direction", placement=placement)
    dt = ET.SubElement(d, "direction-type")
    w = ET.SubElement(dt, "words", {"font-size": size, "font-style": style, "font-weight": weight})
    if enclosure:
        w.set("enclosure", enclosure)
    w.text = text
    ET.SubElement(d, "voice").text = voice
    ET.SubElement(d, "staff").text = str(staff)
    return d


NUMBERED = ("slur", "tuplet", "wedge", "octave-shift", "glissando", "slide", "wavy-line", "bracket", "dashes")


def layer_voice(op: "OutPart", v: str, layer: int) -> str:
    """Voice id of source voice v in voice layer `layer` (layer 0 = the source's own ids)."""
    return v if layer == 0 else str(int(v) + layer * op.max_voice)


def relayer(el: ET.Element, op: "OutPart", layer: int, voice: Optional[str] = None):
    """Put a copied element into its output voice (`voice`, from alloc_voices) and, in voice layer `layer` > 0,
    move its spanner numbers on, so that overlapping fragments never share a voice or an open slur / tuplet /
    hairpin number.  Only these encoding ids change; the written notation does not."""
    if voice is not None:
        for v in el.iter("voice"):
            v.text = voice
    if layer == 0:
        return
    for tag in NUMBERED:
        for x in el.iter(tag):
            x.set("number", str(int(x.get("number", "1")) + 3 * layer))


def alloc_voices(op: "OutPart", pieces: List[Placed]) -> Dict[tuple, str]:
    """Output voice id of every (piece, staff, source voice): a piece keeps the source's id when that id is free for
    its whole extent on the staff; otherwise it gets the lowest free id of the part (ids of other staves and of the
    source excluded), so that simultaneous streams on a staff never share a voice and ids are reused over time.
    Returns {(id(piece), staff, source voice): voice id}; op.max_simultaneous records the densest staff."""
    part = op.part
    src_ids = {e.voice for e in part.elems if e.kind == "note"}
    streams = []                                              # (start, end, staff, src voice, piece)
    for pl in pieces:
        ext: Dict[tuple, List[Fr]] = {}
        for e in part.elems:
            if e.kind == "note" and pl.piece.a <= e.pos < pl.piece.b:
                k = (e.staff, e.voice)
                lo, hi = pl.out(e.pos), pl.out(min(e.end, pl.piece.b))
                if k in ext:
                    ext[k][0] = min(ext[k][0], lo); ext[k][1] = max(ext[k][1], hi)
                else:
                    ext[k] = [lo, hi]
        for (st, v), (lo, hi) in ext.items():
            streams.append((lo, max(hi, lo + Fr(1, 64)), st, v, pl))
    streams.sort(key=lambda x: (x[0], getattr(x[4], "layer", 0), x[2], x[3]))
    busy: Dict[str, List[Tuple[Fr, Fr]]] = defaultdict(list)  # voice id -> intervals in use
    staff_of_id: Dict[str, int] = {}
    for e in part.elems:
        if e.kind == "note":
            staff_of_id.setdefault(e.voice, e.staff)
    out: Dict[tuple, str] = {}
    most = 0

    def free(vid, lo, hi):
        return all(hi <= a or lo >= b for a, b in busy[vid])
    for lo, hi, st, v, pl in streams:
        cands = [v] if staff_of_id.get(v, st) == st else []
        # further ids: first the four of this staff in the usual numbering (1-4 on staff 1, 5-8 on staff 2...),
        # then anything free
        pool = [str(n) for n in range(4 * (st - 1) + 1, 4 * st + 1)] + [str(n) for n in range(1, 65)]
        for vid in pool:
            if len(cands) >= 64:
                break
            if vid not in src_ids and staff_of_id.get(vid, st) == st and vid not in cands:
                cands.append(vid)
        for vid in cands:
            if free(vid, lo, hi):
                out[(id(pl), st, v)] = vid
                busy[vid].append((lo, hi))
                staff_of_id.setdefault(vid, st)
                break
        # density check
        same = [vid for vid, ivs in busy.items() if staff_of_id.get(vid) == st and any(a < hi and lo < b for a, b in ivs)]
        most = max(most, len(same))
    op.max_simultaneous = most
    return out


def time_sig(length: Fr) -> Tuple[int, int]:
    for den in (4, 8, 16, 32, 64):
        b = length * den / 4
        if b.denominator == 1:
            return int(b), den
    raise ValueError(f"bar length {length} not notatable")


# ------------------------------------------------------------------ state at a position
def state_at(part: mxl.Part, a: Fr):
    """Clef per staff, latest dynamic and latest technique word in effect at position a (latest by position,
    then document order - elements inside a bar are not in time order because of <backup>)."""
    clefs: Dict[int, tuple] = {}
    dyn = tech = None
    kd = kt = None
    for e in part.elems:
        if e.pos > a:
            continue
        key = (e.pos, e.idx)
        if e.kind == "attributes":
            for c in e.el.findall("clef"):
                st = int(c.get("number", "1"))
                if st not in clefs or clefs[st][0] <= key:
                    clefs[st] = (key, c)
        elif e.kind == "direction" and e.pos < a:
            for d in e.el.iter("dynamics"):
                for c in d:
                    if kd is None or kd <= key:
                        dyn, kd = c.tag, key
            for w in e.el.iter("words"):
                t = (w.text or "").strip()
                if TECH_RE.match(t) and (kt is None or kt <= key):
                    tech, kt = t, key
    return {st: v[1] for st, v in clefs.items()}, dyn, tech


def initial_clefs(part: mxl.Part) -> Dict[int, ET.Element]:
    """Clef per staff in force at the start of a source part (the latest clef element at position 0)."""
    out: Dict[int, ET.Element] = {}
    for e in part.elems:
        if e.kind == "attributes" and e.pos == 0:
            for c in e.el.findall("clef"):
                out[int(c.get("number", "1"))] = c
    if not out:                                               # no clef at 0: the first clef of the part
        for e in part.elems:
            if e.kind == "attributes":
                for c in e.el.findall("clef"):
                    out.setdefault(int(c.get("number", "1")), c)
    return out


def _clef_key(c: ET.Element) -> tuple:
    return (c.findtext("sign"), c.findtext("line"), c.findtext("clef-octave-change"))


def clef_events(op: "OutPart", pieces: List[Placed], valloc: Dict[tuple, str]) -> List["Ev"]:
    """Clef changes of the output staves.  Every copied passage needs the clef that is in force at that point of its
    source (at its start and at the source's own clef changes inside it).  When passages of one part overlap (voice
    layers), the most recently entered passage decides; when it ends, the clef of the passage still sounding comes
    back.  Overlaps whose passages need different clefs are recorded in op.clef_conflicts (the pitches are exact
    either way; only the clef they are shown in differs during the overlap)."""
    part = op.part
    src = defaultdict(list)                                   # staff -> [(pos, idx, clef)]
    for e in part.elems:
        if e.kind == "attributes":
            for c in e.el.findall("clef"):
                src[int(c.get("number", "1"))].append((e.pos, e.idx, c))
    for st in src:
        src[st].sort(key=lambda x: (x[0], x[1]))

    def clef_at(st, q):
        cur = None
        for (p, i, c) in src.get(st, []):
            if p <= q:
                cur = c
            else:
                break
        return cur
    out: List[Ev] = []
    ps = sorted(pieces, key=lambda p: (p.x, getattr(p, "layer", 0)))
    for st in range(1, op.staves + 1):
        c0 = initial_clefs(part).get(st)
        cur = _clef_key(c0) if c0 is not None else None
        times = set()
        for p in ps:
            times.add(p.x); times.add(p.end)
            for (q, i, c) in src.get(st, []):
                if p.piece.a < q < p.piece.b:
                    times.add(p.out(q))
        conflict_open = False
        for t in sorted(times):
            act = [p for p in ps if p.x <= t < p.end]
            if not act:
                conflict_open = False
                continue
            top = max(act, key=lambda p: (p.x, getattr(p, "layer", 0)))
            want = clef_at(st, top.piece.a + (t - top.x))
            if want is None:
                continue
            keys = {_clef_key(k) for k in (clef_at(st, p.piece.a + (t - p.x)) for p in act) if k is not None}
            if len(keys) > 1:
                if not conflict_open:
                    op.clef_conflicts.append((float(t), st))
                conflict_open = True
            else:
                conflict_open = False
            if _clef_key(want) != cur:
                cc = copy.deepcopy(want); _strip_x(cc)
                cc.set("number", str(st))
                v = next((vid for (pid_, st2, v0), vid in valloc.items() if pid_ == id(top) and st2 == st),
                         op.main_voice.get(st, "1"))
                out.append(Ev(t, st, v, "clef", cc, order=(-1, -3, st)))
                cur = _clef_key(want)
    return out


# ------------------------------------------------------------------ events
@dataclass
class Ev:
    pos: Fr
    staff: int
    voice: str
    kind: str                  # note | dir | clef | rest
    el: ET.Element
    dur: Fr = Fr(0)            # advances the voice (notes, rests); 0 for chord members / graces / directions
    order: Tuple = ()


def fragment_labels(lane_placed: List[Placed], lane_dropped: List[tuple], part_abbr: Dict[str, str]) -> List[tuple]:
    """One label per fragment of a material (on its top staff): source name, source span (union over the parts,
    first pass), dynamics / technique in effect, '✂' + the end the heard audio reached when the fragment had to be
    shortened before the next entry, '▸' + the heard start when it had to start later, '−part' for parts left out.
    Returns (x, text, source-barline markers [(x, bar number)])."""
    by_frag: Dict[int, List[Placed]] = defaultdict(list)
    for p in lane_placed:
        by_frag[id(p.frag)].append(p)
    drop_by_frag: Dict[int, List[str]] = defaultdict(list)
    for f, pid in lane_dropped:
        drop_by_frag[id(f)].append(pid)
    out = []
    fmt = lambda m, bt: f"m.{m}" + ("" if bt == 1 else f"/{float(bt):g}")
    for fid, pls in by_frag.items():
        pls = sorted(pls, key=lambda p: (p.x, p.piece.pid))
        sc = pls[0].score
        f = pls[0].frag
        passes = sorted({p.piece.passno for p in pls})
        texts = []
        for k, ps in enumerate(passes):
            pp = [p for p in pls if p.piece.passno == ps]
            a = min(p.piece.a for p in pp); b = max(p.piece.b for p in pp)
            ma, ba = measure_beat(sc, a); mb, bb = measure_beat(sc, b)
            texts.append(f"{fmt(ma, ba)}–{fmt(mb, bb)}")
        first = min(pls, key=lambda p: (p.x, p.piece.pid))
        clefs, dyn, tech = state_at(sc.part(first.piece.pid), first.piece.a)
        ctx = ", ".join(x for x in (dyn, tech) if x)
        txt = f"{short_name(f.material)} " + " ↺ ".join(texts) + (f" [{ctx}]" if ctx else "")
        cuts = [p.cut_from for p in pls if p.cut_from is not None]
        if cuts:
            mc, bc = measure_beat(sc, max(cuts))
            txt += f" ✂{fmt(mc, bc)}"
        heads = [p.head_from for p in pls if p.head_from is not None and p.first]
        if heads:
            mh, bh = measure_beat(sc, min(heads))
            txt += f" ▸{fmt(mh, bh)}"
        if any(getattr(p, "layer", 0) > 0 for p in pls):
            txt += " ⧉"
        entry = Fr(round(f.t0 * 8), 8)                            # map entry (the fragment reference)
        pre = entry - first.x
        if pre >= 1:
            txt += f" (入+{float(pre):.1f}s)"
        if drop_by_frag.get(fid):
            txt += " −" + ",".join(part_abbr.get(pid, pid) for pid in sorted(drop_by_frag[fid]))
        marks = []
        for ps in passes:
            pp = [p for p in pls if p.piece.passno == ps]
            a = min(p.piece.a for p in pp); b = max(p.piece.b for p in pp)
            ref = pp[0]
            for m in sc.parts[0].measures:
                if a < m.start < b:
                    marks.append((ref.out(m.start), m.number))
        if pre >= 1:
            marks.append((entry, "▼"))
        out.append((first.x, txt, marks))
    # fragments left out in every part: a small marker at the map time with the span the audio asked for
    seen = set(by_frag)
    for f, pid in lane_dropped:
        if id(f) in seen:
            continue
        seen.add(id(f))
        o = min((opts[0] for opts in f.options.values() if opts), key=lambda o: o.cost, default=None)
        rng = ""
        if o is not None:
            a = min(pc.a for pc in o.pieces); b = max(pc.b for pc in o.pieces)
            sc = next(iter(lane_placed), None)
            sc = sc.score if sc is not None else None
            if sc is not None:
                ma, ba = measure_beat(sc, a); mb, bb = measure_beat(sc, b)
                rng = f" {fmt(ma, ba)}–{fmt(mb, bb)}"
        out.append((Fr(round(f.t0 * 8), 8), f"✕ {short_name(f.material)}{rng} (省略)", []))
    return out


def short_name(material: str) -> str:
    m = re.match(r"^(.*)_(\d{2})\.(\d{2})\.(\d{3})_(\d{2})\.(\d{2})(?:\.(\d{3}))?$", material)
    return m.group(1) if m else material


def build_events(op: OutPart, pieces: List[Placed], labels: Optional[List[tuple]] = None) -> Tuple[List[Ev], List[Tuple[Fr, Fr]]]:
    """All events of one output part and the output intervals where the material sounds."""
    evs: List[Ev] = []
    valloc = alloc_voices(op, pieces)
    for n, (x, txt, marks) in enumerate(labels or []):
        mv = op.main_voice.get(1, "1")
        lay = getattr(op, "layout", None) or Layout()
        evs.append(Ev(x, 1, mv, "dir", _words(txt, 1, mv, size=f"{lay.label_pt:g}", weight="bold", enclosure="rectangle"),
                      order=(-1, -2, n)))
        for (mx, num) in marks:
            lab = num if num == "▼" else f"|{num}"
            evs.append(Ev(mx, 1, mv, "dir", _words(lab, 1, mv, size=f"{lay.mark_pt:g}", style="italic"), order=(-1, -1, n)))
    active: List[Tuple[Fr, Fr]] = []
    part = op.part
    by_idx = {e.idx: e for e in part.elems}
    evs += clef_events(op, pieces, valloc)
    for n, pl in enumerate(sorted(pieces, key=lambda p: p.x)):
        a, b = pl.piece.a, pl.piece.b
        L = getattr(pl, "layer", 0)
        mine = {(st, v): vid for (pid_, st, v), vid in valloc.items() if pid_ == id(pl)}

        def vmap(v, st=None, mine=mine):
            if st is not None and (st, v) in mine:
                return mine[(st, v)]
            for (st2, v2), vid in mine.items():               # a direction of a voice without notes here
                if v2 == v or st2 == st:
                    return vid
            return v
        active.append((pl.x, pl.end))
        # copied elements
        pending_slur_start: Dict[tuple, List[str]] = defaultdict(list)
        first_note_of_voice: Dict[tuple, ET.Element] = {}
        last_note_of_voice: Dict[tuple, ET.Element] = {}
        first_pos_of_voice: Dict[tuple, Fr] = {}
        last_pos_of_voice: Dict[tuple, Fr] = {}
        copies: Dict[int, ET.Element] = {}
        # a hairpin / octave line that ends exactly where this passage starts belongs to the music before it: its
        # stop mark would dangle here without a start
        skip = {g.members[-1] for g in pl.score.groups
                if g.part == op.src_pid and g.kind in ("wedge", "octave") and g.start < a and g.end == a}
        for e in part.elems:
            if not (a <= e.pos < b) or e.idx in skip:
                continue
            if e.kind == "note":
                el = copy.deepcopy(e.el)
                _strip_x(el)
                _rescale(el, _div_at(part, e.pos))
                r = el.find("rest")
                if r is not None and r.get("measure") == "yes":
                    # a bar rest of the source: the output bars differ, so write rests of the same length
                    vis = el.get("print-object")
                    for k2, (dv, typ, _d) in enumerate(rest_values(e.dur, Fr(0))):
                        rr = _rest_el(dv, typ, e.staff, vmap(e.voice, e.staff))
                        if vis:
                            rr.set("print-object", vis)
                        evs.append(Ev(pl.out(e.pos) + sum((x[0] for x in rest_values(e.dur, Fr(0))[:k2]), Fr(0)),
                                      e.staff, vmap(e.voice, e.staff), "note", rr, dv, order=(n, 0, e.idx + k2 / 100)))
                    continue
                for ins in el.findall("instrument"):
                    ins.set("id", ins.get("id").replace(op.src_pid + "-", op.pid + "-", 1))
                relayer(el, op, L, vmap(e.voice, e.staff))
                scale_fonts(el, getattr(op, "font_scale", 1.0))
                dur = Fr(0) if (e.chord or e.grace) else e.dur
                evs.append(Ev(pl.out(e.pos), e.staff, vmap(e.voice, e.staff), "note", el, dur, order=(n, 0, e.idx)))
                copies[e.idx] = el
                key = (e.staff, e.voice)
                if not e.rest and not e.grace:
                    first_note_of_voice.setdefault(key, el)
                    first_pos_of_voice.setdefault(key, e.pos)
                    if not e.chord:
                        last_note_of_voice[key] = el
                        last_pos_of_voice[key] = e.pos
            elif e.kind == "direction":
                el = copy.deepcopy(e.el)
                _strip_x(el)
                _rescale(el, _div_at(part, e.pos))
                for mt in el.findall("direction-type/metronome"):
                    # a source tempo mark stays visible as text; the output keeps quarter = 60
                    dt = el.find("direction-type")
                    unit = mt.findtext("beat-unit") or "quarter"
                    sym = {"quarter": "♩", "eighth": "♪", "half": "𝅗𝅥"}.get(unit, unit)
                    dt.remove(mt)
                    w = ET.SubElement(dt, "words", {"font-size": mt.get("font-size") or "12"})   # scaled below
                    w.text = f"({sym}={mt.findtext('per-minute')})"
                for snd in el.findall("sound"):
                    el.remove(snd)
                for off in el.findall("offset"):              # its position is taken as the event position
                    el.remove(off)
                if el.find("direction-type/*") is None:
                    continue
                relayer(el, op, L, vmap(e.voice, e.staff))
                scale_fonts(el, getattr(op, "font_scale", 1.0))
                evs.append(Ev(pl.out(e.pos), e.staff, vmap(e.voice, e.staff), "dir", el, order=(n, 0, e.idx)))
        # ties that lead out of the passage (or into it from outside): the note on the other side is not written
        # here, so the tie mark would dangle - drop it (the note itself stays)
        def _pk(el):
            pp = el.find("pitch")
            return None if pp is None else (pp.findtext("step"), pp.findtext("alter") or "0", pp.findtext("octave"))
        tie_notes = [x for x in part.elems if x.kind == "note" and not x.rest and not x.grace and x.el.find("tie") is not None]
        by_start = defaultdict(list); by_end = defaultdict(list)
        for x in tie_notes:
            by_start[(x.staff, x.pos, _pk(x.el))].append(x)
            by_end[(x.staff, x.end, _pk(x.el))].append(x)
        for idx_, el in copies.items():
            src = by_idx[idx_]
            if src.rest or src.grace:
                continue
            k = _pk(src.el)
            if any(t.get("type") == "start" for t in src.el.findall("tie")):
                ok = any(a <= f.pos < b and any(t.get("type") == "stop" for t in f.el.findall("tie"))
                         for f in by_start.get((src.staff, src.end, k), []))
                if not ok:
                    _drop_tie(el, "start")
            if any(t.get("type") == "stop" for t in src.el.findall("tie")):
                ok = any(a <= f.pos < b and any(t.get("type") == "start" for t in f.el.findall("tie"))
                         for f in by_end.get((src.staff, src.pos, k), []))
                if not ok:
                    _drop_tie(el, "stop")
        # soft spanners cut by the passage boundaries: close them at the boundary
        for g in pl.score.groups:
            if g.part != op.src_pid:
                continue
            key = (g.staff, g.voice)
            cross_a = g.start < a < g.end
            cross_b = g.start < b < g.end
            if g.kind in ("wedge", "octave") and g.start < b and g.end == b and g.end > a:
                cross_b = True                        # its stop sits exactly on the passage end (not copied)
            if not (cross_a or cross_b):
                continue
            if g.kind == "slur":
                num = str(int(g.number) + 3 * L)                # the copies carry layer-shifted numbers
                fn_ = first_note_of_voice.get(key); ln_ = last_note_of_voice.get(key)
                if cross_a and cross_b and fn_ is not None and first_pos_of_voice.get(key) == last_pos_of_voice.get(key):
                    continue                                    # one note / chord under a longer slur: nothing to draw
                if cross_a and key in first_note_of_voice:
                    fn = first_note_of_voice[key]
                    has_stop = any(s.get("type") == "stop" for s in fn.iter("slur"))
                    if has_stop:                      # the slur would start and stop on the same note
                        for nt in fn.findall("notations"):
                            for s in list(nt.findall("slur")):
                                if s.get("type") == "stop" and s.get("number", "1") == num:
                                    nt.remove(s)
                    else:
                        nt = _child(fn, "notations")
                        ET.SubElement(nt, "slur", type="start", number=num)
                if cross_b and key in last_note_of_voice:
                    ln = last_note_of_voice[key]
                    has_start = any(s.get("type") == "start" and s.get("number", "1") == num for s in ln.iter("slur"))
                    if has_start:
                        for nt in ln.findall("notations"):
                            for s in list(nt.findall("slur")):
                                if s.get("type") == "start" and s.get("number", "1") == num:
                                    nt.remove(s)
                    else:
                        nt = _child(ln, "notations")
                        ET.SubElement(nt, "slur", type="stop", number=num)
            elif g.kind in ("wedge", "octave"):
                src = by_idx.get(g.members[0])
                if src is None:
                    continue
                tag = "wedge" if g.kind == "wedge" else "octave-shift"
                orig = src.el.find(f"direction-type/{tag}")
                if orig is None:
                    continue
                if cross_a:
                    d = copy.deepcopy(src.el); _strip_x(d)
                    for snd in d.findall("sound"):
                        d.remove(snd)
                    off = d.find("offset")
                    if off is not None:
                        d.remove(off)
                    relayer(d, op, L, vmap(src.voice, g.staff))
                    scale_fonts(d, getattr(op, "font_scale", 1.0))
                    evs.append(Ev(pl.x, g.staff, vmap(src.voice, g.staff), "dir", d, order=(n, -1, g.members[0])))
                if cross_b:
                    d = ET.Element("direction")
                    dt = ET.SubElement(d, "direction-type")
                    ET.SubElement(dt, tag, type="stop", number=str(int(orig.get("number", "1")) + 3 * L),
                                  **({"size": orig.get("size")} if tag == "octave-shift" and orig.get("size") else {}))
                    ET.SubElement(d, "voice").text = vmap(src.voice, g.staff)
                    ET.SubElement(d, "staff").text = str(g.staff)
                    evs.append(Ev(pl.end, g.staff, vmap(src.voice, g.staff), "dir", d, order=(n, 9, 0)))
            elif g.kind == "trill":
                wl = {} if (L == 0 and g.number == "1") else {"number": str(int(g.number) + 3 * L)}

                def has(el, t):
                    return any(w.get("type") == t for w in el.iter("wavy-line"))

                def drop(el, t):
                    for orn in el.iter("ornaments"):
                        for w in list(orn.findall("wavy-line")):
                            if w.get("type") == t:
                                orn.remove(w)
                if cross_a and key in first_note_of_voice:
                    fn = first_note_of_voice[key]
                    if has(fn, "stop"):                  # the line would start and stop on the same note
                        drop(fn, "stop")
                    else:
                        orn = _child(_child(fn, "notations"), "ornaments")
                        ET.SubElement(orn, "wavy-line", type="start", **wl)
                if cross_b and key in last_note_of_voice:
                    ln = last_note_of_voice[key]
                    if has(ln, "start"):
                        drop(ln, "start")
                    else:
                        orn = _child(_child(ln, "notations"), "ornaments")
                        ET.SubElement(orn, "wavy-line", type="stop", **wl)
    return evs, active


# ------------------------------------------------------------------ writer
@dataclass
class Layout:
    """Page layout of the score map: a tall page, one system of all staves per page, small staves (like a large
    orchestral score).  Distances in tenths (40 tenths = one staff height = staff_mm)."""
    page_w_mm: float = 420.0          # A2 portrait
    page_h_mm: float = 594.0
    staff_mm: float = 2.9             # height of a five-line staff
    margin: int = 150                 # page margins
    top_system: int = 150             # above the first staff: tempo, time labels, the first fragment labels
    gap_group: int = 85               # between two materials (a label may sit above a material's first staff)
    gap_part: int = 65                # between the parts of one material
    gap_staff: int = 55               # between the two staves of one luminasity instrument
    system_quarters: float = 36.0     # bars are gathered into systems of at most this many quarters (a longer bar
                                      # is a system of its own); every system starts a new page
    label_pt: float = 5.5             # fragment labels
    mark_pt: float = 4.5              # source bar numbers, entry marks
    time_pt: float = 5.0              # time labels above the first staff
    word_pt: float = 5.0              # default text size

    @property
    def tenth_mm(self) -> float:
        return self.staff_mm / 40.0


def staff_mm_of(score: mxl.Score) -> float:
    """Staff height (mm) of a source score, from its <defaults><scaling> (7 mm when missing)."""
    sc = score.root.find("defaults/scaling")
    try:
        return float(sc.findtext("millimeters")) * 40.0 / float(sc.findtext("tenths"))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return 7.0


def scale_fonts(el: ET.Element, factor: float):
    """Copied text keeps its size relative to the staff: font-size (points) times new / source staff height."""
    if abs(factor - 1.0) < 1e-6:
        return
    for x in el.iter():
        fs = x.get("font-size")
        if fs:
            try:
                x.set("font-size", f"{float(fs) * factor:.2f}")
            except ValueError:
                pass


def make_defaults(lay: Layout, src_defaults: Optional[ET.Element]) -> ET.Element:
    t = lay.tenth_mm
    d = ET.Element("defaults")
    sc = ET.SubElement(d, "scaling")
    ET.SubElement(sc, "millimeters").text = f"{lay.staff_mm:.4f}"
    ET.SubElement(sc, "tenths").text = "40"
    pl = ET.SubElement(d, "page-layout")
    ET.SubElement(pl, "page-height").text = f"{lay.page_h_mm / t:.2f}"
    ET.SubElement(pl, "page-width").text = f"{lay.page_w_mm / t:.2f}"
    pm = ET.SubElement(pl, "page-margins", type="both")
    for side in ("left", "right", "top", "bottom"):
        ET.SubElement(pm, f"{side}-margin").text = str(lay.margin)
    sl = ET.SubElement(d, "system-layout")
    sm = ET.SubElement(sl, "system-margins")
    ET.SubElement(sm, "left-margin").text = "0"
    ET.SubElement(sm, "right-margin").text = "0"
    ET.SubElement(sl, "system-distance").text = str(lay.gap_group * 2)
    ET.SubElement(sl, "top-system-distance").text = str(lay.top_system)
    stl = ET.SubElement(d, "staff-layout")
    ET.SubElement(stl, "staff-distance").text = str(lay.gap_part)
    if src_defaults is not None and src_defaults.find("appearance") is not None:
        d.append(copy.deepcopy(src_defaults.find("appearance")))      # line widths in tenths: they scale along
    mf = src_defaults.find("music-font") if src_defaults is not None else None
    ET.SubElement(d, "music-font", {"font-family": mf.get("font-family") if mf is not None else "Opus Std",
                                    "font-size": f"{lay.staff_mm * 72 / 25.4:.2f}"})
    wf = src_defaults.find("word-font") if src_defaults is not None else None
    ET.SubElement(d, "word-font", {"font-family": wf.get("font-family") if wf is not None else "Palatino",
                                   "font-size": f"{lay.word_pt:.2f}"})
    return d


def systems_of(bars: List[Fr], lay: Layout, forced: Optional[set] = None) -> List[int]:
    """Indices of the bars that start a system (greedy: at most lay.system_quarters per system; a bar that starts at
    one of the `forced` positions always starts a new system / page)."""
    starts, acc = [0], Fr(0)
    for bi in range(len(bars) - 1):
        L = bars[bi + 1] - bars[bi]
        if bi > 0 and (acc + L > lay.system_quarters or (forced and bars[bi] in forced)):
            starts.append(bi)
            acc = Fr(0)
        acc += L
    return starts


def write(path: str, title: str, parts: List[OutPart], placed: List[Placed], bars: List[Fr], defaults: Optional[ET.Element],
          time_labels: bool = True, dropped: Optional[List[tuple]] = None, layout: Optional[Layout] = None,
          page_breaks: Optional[List[Fr]] = None):
    root = ET.Element("score-partwise", version="3.1")
    wk = ET.SubElement(root, "work")
    ET.SubElement(wk, "work-title").text = title
    ident = ET.SubElement(root, "identification")
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "latent-space-material scoremap"
    lay = layout or Layout()
    root.append(make_defaults(lay, defaults))
    for op in parts:
        op.font_scale = lay.staff_mm / staff_mm_of(op.score)
        op.layout = lay
    sys_starts = set(systems_of(bars, lay, set(page_breaks or [])))
    pl = ET.SubElement(root, "part-list")
    by_lane: Dict[tuple, List[Placed]] = defaultdict(list)
    for p in placed:
        by_lane[(p.frag.lane, p.piece.pid)].append(p)
    # part groups: one bracket per material
    open_group = None
    gnum = 0
    for op in parts:
        if op.first_of_material:
            if open_group is not None:
                ET.SubElement(pl, "part-group", type="stop", number="1")
            gnum += 1
            pg = ET.SubElement(pl, "part-group", type="start", number="1")
            ET.SubElement(pg, "group-name").text = op.short
            ET.SubElement(pg, "group-symbol").text = "bracket"
            ET.SubElement(pg, "group-barline").text = "yes"
            open_group = gnum
        sp = copy.deepcopy(op.part.score_part)
        sp.set("id", op.pid)
        for pn in sp.findall("part-name"):
            pn.text = f"{op.short} {op.part.name}"
        for pnd in sp.findall("part-name-display"):
            sp.remove(pnd)
        for x in sp.iter():
            if "id" in x.attrib and x is not sp:
                x.set("id", x.get("id").replace(op.src_pid + "-", op.pid + "-", 1))
        pl.append(sp)
    if open_group is not None:
        ET.SubElement(pl, "part-group", type="stop", number="1")

    lane_placed: Dict[int, List[Placed]] = defaultdict(list)
    for p in placed:
        lane_placed[p.frag.lane].append(p)
    lane_dropped: Dict[int, List[tuple]] = defaultdict(list)
    for f, pid in (dropped or []):
        lane_dropped[f.lane].append((f, pid))
    for idx, op in enumerate(parts):
        labels = None
        if op.first_of_material:
            abbr = {p.pid: (p.score_part.findtext("part-abbreviation") or p.name).strip() for p in op.score.parts}
            labels = [lb for lb in fragment_labels(lane_placed[op.lane], lane_dropped[op.lane], abbr) if lb[0] is not None]
        evs, active = build_events(op, by_lane[(op.lane, op.src_pid)], labels)
        evs, n_cut = split_at_barlines(evs, bars)
        op.notes_cut = n_cut
        op.beams_cut = rebeam_at_barlines(evs, bars)
        part_el = ET.SubElement(root, "part", id=op.pid)
        # static attributes from the source part's first attributes
        fa = op.part.first_attributes
        for bi in range(len(bars) - 1):
            b0, b1 = bars[bi], bars[bi + 1]
            L = b1 - b0
            m = ET.SubElement(part_el, "measure", number=str(bi + 1))
            if bi in sys_starts:
                pr = ET.SubElement(m, "print", **({"new-page": "yes"} if bi > 0 else {}))
                if idx == 0:
                    sl = ET.SubElement(pr, "system-layout")
                    sm = ET.SubElement(sl, "system-margins")
                    ET.SubElement(sm, "left-margin").text = "0"
                    ET.SubElement(sm, "right-margin").text = "0"
                    ET.SubElement(sl, "top-system-distance").text = str(lay.top_system)
                else:
                    stl = ET.SubElement(pr, "staff-layout", number="1")
                    ET.SubElement(stl, "staff-distance").text = str(lay.gap_group if op.first_of_material else lay.gap_part)
                for st in range(2, op.staves + 1):
                    stl = ET.SubElement(pr, "staff-layout", number=str(st))
                    ET.SubElement(stl, "staff-distance").text = str(lay.gap_staff)
            at = ET.SubElement(m, "attributes")
            if bi == 0:
                ET.SubElement(at, "divisions").text = str(DIV)
                if fa is not None and fa.find("key") is not None:
                    k = copy.deepcopy(fa.find("key")); _strip_x(k); at.append(k)
            beats, den = time_sig(L)
            t = ET.SubElement(at, "time", {"print-object": "no"})
            ET.SubElement(t, "beats").text = str(beats)
            ET.SubElement(t, "beat-type").text = str(den)
            if bi == 0:
                ET.SubElement(at, "staves").text = str(op.staves)
                if fa is not None:
                    ps = fa.find("part-symbol")
                    if ps is not None:
                        at.append(copy.deepcopy(ps))
                    # the clef of every staff in force at the start of the source (it may sit in a later
                    # <attributes> of the first bar)
                    for st, c in sorted(initial_clefs(op.part).items()):
                        cc = copy.deepcopy(c); _strip_x(cc); cc.set("number", str(st)); at.append(cc)
                    # staff details (e.g. the zero-line upper staff of luminasity) from the whole source part
                    sd_seen = set()
                    for e in op.part.elems:
                        if e.kind == "attributes":
                            for sd in e.el.findall("staff-details"):
                                num = sd.get("number", "1")
                                if num in sd_seen or sd.find("staff-lines") is None:
                                    continue
                                sd_seen.add(num)
                                s2 = ET.Element("staff-details", number=num)
                                s2.append(copy.deepcopy(sd.find("staff-lines")))
                                at.append(s2)
                if idx == 0:
                    pass
            # tempo and time labels on the very first part
            if idx == 0:
                if bi == 0:
                    d = ET.SubElement(m, "direction", placement="above")
                    dt = ET.SubElement(d, "direction-type")
                    mt = ET.SubElement(dt, "metronome")
                    ET.SubElement(mt, "beat-unit").text = "quarter"
                    ET.SubElement(mt, "per-minute").text = "60"
                    ET.SubElement(d, "staff").text = "1"
                    ET.SubElement(d, "sound", tempo="60")
                if time_labels:
                    sec = float(b0)
                    lab = f"{int(sec // 60)}:{sec % 60:04.1f}"
                    m.append(_words(lab, 1, op.main_voice.get(1, "1"), size=f"{lay.time_pt:g}", style="italic"))
            _emit_measure(m, op, evs, active, b0, b1)
    tree = ET.ElementTree(root)
    ET.indent(tree, space=" ")
    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
                b'"http://www.musicxml.org/dtds/partwise.dtd">\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


def _emit_measure(m: ET.Element, op: OutPart, evs: List[Ev], active: List[Tuple[Fr, Fr]], b0: Fr, b1: Fr):
    L = b1 - b0
    mine = [e for e in evs if b0 <= e.pos < b1]
    streams: Dict[Tuple[int, str], List[Ev]] = defaultdict(list)
    for e in mine:
        v = e.voice
        if e.kind in ("clef", "dir"):
            # attach to the main voice of its staff unless that voice has notes of its own here
            has_notes = any(x.kind == "note" and x.staff == e.staff and x.voice == v for x in mine)
            if not has_notes:
                v = op.main_voice.get(e.staff, v)
        streams[(e.staff, v)].append(e)
    # filler rests: per staff, where the material does not sound
    for st in range(1, op.staves + 1):
        mv = op.main_voice.get(st, "1")
        gaps = [(b0, b1)]
        for (x0, x1) in active:
            nxt = []
            for g0, g1 in gaps:
                if x1 <= g0 or x0 >= g1:
                    nxt.append((g0, g1))
                else:
                    if g0 < x0:
                        nxt.append((g0, x0))
                    if x1 < g1:
                        nxt.append((x1, g1))
            gaps = nxt
        cuts_at = sorted({e.pos for e in streams.get((st, mv), []) if e.kind in ("dir", "clef") and b0 < e.pos < b1})
        if gaps == [(b0, b1)] and not cuts_at:
            streams[(st, mv)].append(Ev(b0, st, mv, "rest", _rest_el(L, "whole", st, mv, measure_rest=True), L, order=(-1, 0, 0)))
            continue
        split = []
        for g0, g1 in gaps:
            pts = [g0] + [c for c in cuts_at if g0 < c < g1] + [g1]
            split += list(zip(pts[:-1], pts[1:]))
        gaps = split
        for g0, g1 in gaps:
            pos = g0
            for dur, typ, dots in rest_values(g1 - g0, g0 - b0):
                streams[(st, mv)].append(Ev(pos, st, mv, "rest", _rest_el(dur, typ, st, mv), dur, order=(-1, 0, 0)))
                pos += dur
    # emit: staff by staff, main voice first
    keys = sorted(streams, key=lambda k: (k[0], 0 if k[1] == op.main_voice.get(k[0]) else 1, k[1]))
    first = True
    for key in keys:
        seq = sorted(streams[key], key=lambda e: (e.pos, 0 if e.kind == "clef" else (1 if e.kind == "dir" else 2), e.order))
        if not first:
            bk = ET.SubElement(m, "backup")
            ET.SubElement(bk, "duration").text = str(int(cursor_used * DIV))
        first = False
        cursor = b0
        for e in seq:
            if e.pos > cursor:
                fw = ET.SubElement(m, "forward")
                ET.SubElement(fw, "duration").text = str(int((e.pos - cursor) * DIV))
                ET.SubElement(fw, "voice").text = key[1]
                ET.SubElement(fw, "staff").text = str(key[0])
                cursor = e.pos
            if e.kind == "clef":
                at = ET.SubElement(m, "attributes")
                at.append(e.el)
            else:
                if e.kind == "dir" and e.pos < cursor:
                    # a direction inside an already written note of this stream: keep its time with <offset>
                    # (added to an offset the source already had)
                    off = e.el.find("offset")
                    if off is None:
                        off = ET.Element("offset")
                        off.text = "0"
                        kids = list(e.el)
                        i = max(k for k, c in enumerate(kids) if c.tag == "direction-type") + 1
                        e.el.insert(i, off)
                    off.text = str(int(off.text) + int((e.pos - cursor) * DIV))
                m.append(e.el)
                cursor += e.dur
        if key[1] == op.main_voice.get(key[0]) and cursor < b1:
            fw = ET.SubElement(m, "forward")
            ET.SubElement(fw, "duration").text = str(int((b1 - cursor) * DIV))
            ET.SubElement(fw, "voice").text = key[1]
            ET.SubElement(fw, "staff").text = str(key[0])
            cursor = b1
        cursor_used = cursor - b0

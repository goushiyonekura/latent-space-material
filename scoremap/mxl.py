"""Read a source MusicXML (Sibelius export) into timed elements.

Every <note>, <direction>, <attributes>, <barline> of every part gets an absolute position in quarters from the
start of the file (Fraction), its staff and voice, and a reference to the original element (never modified here).
Chord members share the position and duration of the note they attach to; grace notes sit at the position of the
note they precede (duration 0).  Spanner groups (beams, tuplets, two-note tremolos, glissandi/slides: hard; ties,
slurs, wedges, trill lines, octave shifts: soft) are collected as position intervals so that callers can tell
whether a cut at a given position would break them.
"""
import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Tuple


@dataclass
class Elem:
    kind: str                  # note | direction | attributes | barline
    el: ET.Element             # original element (read only)
    part: str
    pos: Fr                    # absolute position (quarters from the file start)
    dur: Fr = Fr(0)            # sounding duration (0 for grace notes and non-notes)
    staff: int = 1
    voice: str = "1"
    measure: str = ""
    mpos: Fr = Fr(0)           # position inside its measure
    chord: bool = False
    grace: bool = False
    rest: bool = False
    idx: int = 0               # document order inside the part
    anchor: int = -1           # for chord members: idx of the note they attach to

    @property
    def end(self) -> Fr:
        return self.pos + self.dur


@dataclass
class Measure:
    number: str
    start: Fr
    length: Fr
    implicit: bool = False


@dataclass
class Part:
    pid: str
    name: str
    score_part: ET.Element                 # <score-part> from the part-list
    elems: List[Elem] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)
    divisions: List[Tuple[Fr, int]] = field(default_factory=list)   # (position, divisions) changes
    staves: int = 1
    first_attributes: Optional[ET.Element] = None

    @property
    def length(self) -> Fr:
        return self.measures[-1].start + self.measures[-1].length if self.measures else Fr(0)


@dataclass
class Group:
    kind: str                  # beam | tuplet | tremolo | gliss | tie | slur | wedge | trill | octave
    part: str
    staff: int
    voice: str
    start: Fr                  # position of the first element
    end: Fr                    # end of the last element (end of the last note)
    hard: bool
    members: List[int] = field(default_factory=list)   # element indices
    open_end: bool = False                               # no closing marker was found
    number: str = "1"                                    # the spanner's number attribute (slurs, wedges, lines)


@dataclass
class Score:
    path: str
    root: ET.Element
    parts: List[Part]
    groups: List[Group]

    @property
    def length(self) -> Fr:
        return max(p.length for p in self.parts)

    def part(self, pid: str) -> Part:
        return next(p for p in self.parts if p.pid == pid)


def _int(el, tag, default=None):
    t = el.findtext(tag)
    return int(t) if t is not None and t.strip() != "" else default


def parse(path: str) -> Score:
    root = ET.parse(path).getroot()
    sp = {s.get("id"): s for s in root.find("part-list").findall("score-part")}
    parts = []
    for part_el in root.findall("part"):
        pid = part_el.get("id")
        part = Part(pid=pid, name=(sp[pid].findtext("part-name") or "").strip(), score_part=sp[pid])
        div = None
        mstart = Fr(0)
        idx = 0
        for m in part_el.findall("measure"):
            local = Fr(0)
            extent = Fr(0)
            last_note: Optional[Elem] = None
            pending_right: List[Elem] = []
            for el in m:
                tag = el.tag
                if tag == "attributes":
                    d = _int(el, "divisions")
                    if d:
                        div = d
                        part.divisions.append((mstart + local, d))
                    st = _int(el, "staves")
                    if st:
                        part.staves = max(part.staves, st)
                    if part.first_attributes is None:
                        part.first_attributes = el
                    part.elems.append(Elem("attributes", el, pid, mstart + local, measure=m.get("number"),
                                           mpos=local, idx=idx))
                elif tag == "note":
                    if div is None:
                        raise ValueError(f"{path}: note before <divisions>")
                    staff = _int(el, "staff", 1)
                    is_chord = el.find("chord") is not None
                    # a chord member without its own <voice> belongs to the note it attaches to
                    voice = (el.findtext("voice") or (last_note.voice if is_chord and last_note is not None else "1")).strip()
                    is_grace = el.find("grace") is not None
                    is_rest = el.find("rest") is not None
                    if is_chord and last_note is not None:
                        pos, dur = last_note.pos, last_note.dur
                        anchor = last_note.idx if last_note.anchor < 0 else last_note.anchor
                    elif is_grace:
                        pos, dur, anchor = mstart + local, Fr(0), -1
                    else:
                        dd = _int(el, "duration", 0)
                        pos, dur, anchor = mstart + local, Fr(dd, div), -1
                        local += dur
                        extent = max(extent, local)
                    e = Elem("note", el, pid, pos, dur, staff, voice, m.get("number"), pos - mstart,
                             chord=is_chord, grace=is_grace, rest=is_rest, idx=idx, anchor=anchor)
                    part.elems.append(e)
                    last_note = e
                elif tag == "backup":
                    local -= Fr(_int(el, "duration", 0), div)
                elif tag == "forward":
                    local += Fr(_int(el, "duration", 0), div)
                    extent = max(extent, local)
                elif tag == "direction":
                    staff = _int(el, "staff", 1)
                    voice = (el.findtext("voice") or "1").strip()
                    # <offset> moves a direction off the current position (Sibelius uses it for hairpin starts,
                    # text placed between notes...): the direction belongs where it sounds / is read
                    shift = Fr(_int(el, "offset", 0), div) if div else Fr(0)
                    part.elems.append(Elem("direction", el, pid, mstart + local + shift, staff=staff, voice=voice,
                                           measure=m.get("number"), mpos=local + shift, idx=idx))
                elif tag == "barline":
                    loc = el.get("location", "right")
                    e = Elem("barline", el, pid, mstart if loc == "left" else mstart + local,
                             measure=m.get("number"), mpos=Fr(0) if loc == "left" else local, idx=idx)
                    part.elems.append(e)
                    if loc == "right":
                        pending_right.append(e)
                # <print>, <sound> and anything else: layout / playback only, not copied
                idx += 1
            length = extent
            for e in pending_right:          # a right barline sits at the end of its measure
                e.pos, e.mpos = mstart + length, length
            part.measures.append(Measure(m.get("number"), mstart, length, m.get("implicit") == "yes"))
            mstart += length
        parts.append(part)
    groups = collect_groups(parts)
    return Score(path, root, parts, groups)


def _notations(el: ET.Element):
    return el.findall("notations")


def collect_groups(parts: List[Part]) -> List[Group]:
    """Spanner groups as position intervals.  Hard groups must never be cut: beams, tuplets, two-note tremolos,
    glissandi and slides.  Soft groups should not be cut if avoidable: ties, slurs, wedges, trill lines, octave
    shifts (the copier can re-open or close soft ones at a boundary)."""
    groups: List[Group] = []
    for part in parts:
        open_: Dict[tuple, Group] = {}
        by_idx = {e.idx: e for e in part.elems}

        def start(key, kind, e, hard):
            prev = open_.pop(key, None)
            if prev is not None:                      # a new start while one is open: close the old one here
                prev.open_end = True
                groups.append(prev)
            g = Group(kind, part.pid, e.staff, e.voice, e.pos, e.end if e.kind == "note" else e.pos, hard, [e.idx],
                      number=str(key[-1]) if kind in ("slur", "wedge", "octave", "trill", "tuplet", "gliss") else "1")
            open_[key] = g

        def stop(key, e):
            g = open_.pop(key, None)
            if g is None:
                return
            g.members.append(e.idx)
            g.end = max(g.end, e.end if e.kind == "note" else e.pos)
            groups.append(g)

        def cont(key, e):
            g = open_.get(key)
            if g is not None:
                g.members.append(e.idx)
                g.end = max(g.end, e.end if e.kind == "note" else e.pos)

        for e in part.elems:
            if e.kind == "note":
                vkey = (e.staff, e.voice)
                # beams: only the primary level decides the group (secondary levels live inside it)
                for b in e.el.findall("beam"):
                    if b.get("number", "1") != "1":
                        continue
                    v = (b.text or "").strip()
                    key = ("beam",) + vkey
                    if v == "begin":
                        start(key, "beam", e, True)
                    elif v == "continue":
                        cont(key, e)
                    elif v == "end":
                        stop(key, e)
                for tie in e.el.findall("tie"):
                    key = ("tie", e.staff, e.voice, _pitch_key(e.el))
                    if tie.get("type") == "stop":
                        stop(key, e)
                for tie in e.el.findall("tie"):
                    key = ("tie", e.staff, e.voice, _pitch_key(e.el))
                    if tie.get("type") == "start":
                        start(key, "tie", e, False)
                for nt in _notations(e.el):
                    for t in nt.findall("tuplet"):
                        key = ("tuplet", e.staff, e.voice, t.get("number", "1"))
                        if t.get("type") == "start":
                            start(key, "tuplet", e, True)
                        elif t.get("type") == "stop":
                            stop(key, e)
                    for s in nt.findall("slur"):
                        key = ("slur", e.staff, e.voice, s.get("number", "1"))
                        if s.get("type") == "start":
                            start(key, "slur", e, False)
                        elif s.get("type") == "stop":
                            stop(key, e)
                        elif s.get("type") == "continue":
                            cont(key, e)
                    for tag in ("glissando", "slide"):
                        for s in nt.findall(tag):
                            key = (tag, e.staff, s.get("number", "1"))
                            if s.get("type") == "start":
                                start(key, "gliss", e, True)
                            elif s.get("type") == "stop":
                                stop(key, e)
                    for orn in nt.findall("ornaments"):
                        for tr in orn.findall("tremolo"):
                            key = ("tremolo", e.staff, e.voice)
                            if tr.get("type") == "start":
                                start(key, "tremolo", e, True)
                            elif tr.get("type") == "stop":
                                stop(key, e)
                        for w in orn.findall("wavy-line"):
                            key = ("trill", e.staff, w.get("number", "1"))
                            if w.get("type") == "start":
                                start(key, "trill", e, False)
                            elif w.get("type") == "stop":
                                stop(key, e)
                            elif w.get("type") == "continue":
                                cont(key, e)
                # tuplets whose start/stop markers are missing: keep the notes of one time-modification run
                # together (handled by the time-modification check in extract)
            elif e.kind == "direction":
                for dt in e.el.findall("direction-type"):
                    for w in dt.findall("wedge"):
                        key = ("wedge", e.staff, w.get("number", "1"))
                        if w.get("type") in ("crescendo", "diminuendo"):
                            start(key, "wedge", e, False)
                        elif w.get("type") == "stop":
                            stop(key, e)
                    for o in dt.findall("octave-shift"):
                        key = ("octave", e.staff, o.get("number", "1"))
                        if o.get("type") in ("up", "down"):
                            start(key, "octave", e, False)
                        elif o.get("type") == "stop":
                            stop(key, e)
        # tuplets from the note values: a run of notes carrying the same <time-modification> fills whole tuplet
        # groups of (normal-notes x normal-type); positions inside a group are inside a tuplet even when the export
        # wrote no <tuplet> start / stop marks.  Two-note tremolos (written 2:1) are tremolo groups instead.  A run
        # that ends before its group is full is kept together as it is.
        TYPE_Q = {"breve": Fr(8), "whole": Fr(4), "half": Fr(2), "quarter": Fr(1), "eighth": Fr(1, 2), "16th": Fr(1, 4),
                  "32nd": Fr(1, 8), "64th": Fr(1, 16), "128th": Fr(1, 32), "256th": Fr(1, 64)}
        pending: Dict[tuple, tuple] = {}                  # (staff, voice) -> (signature, [notes], group length)

        def flush(key):
            item = pending.pop(key, None)
            if item and len(item[1]) > 1:
                grp = item[1]
                groups.append(Group("tuplet", part.pid, grp[0].staff, grp[0].voice, grp[0].pos, grp[-1].end, True,
                                    [x.idx for x in grp]))
        for e in part.elems:
            if e.kind != "note" or e.chord or e.grace:
                continue
            key = (e.staff, e.voice)
            tm = e.el.find("time-modification")
            if tm is None or any(True for _ in e.el.iter("tremolo")):
                flush(key)
                continue
            sig = (tm.findtext("actual-notes"), tm.findtext("normal-notes"), tm.findtext("normal-type"),
                   len(tm.findall("normal-dot")))
            item = pending.get(key)
            if item is not None and (item[0] != sig or item[1][-1].end != e.pos):
                flush(key)
                item = None
            if item is None:
                nt = sig[2] or e.el.findtext("type")
                try:
                    T = int(sig[1]) * TYPE_Q[nt] * (Fr(3, 2) if sig[3] else 1)
                except (KeyError, TypeError, ValueError):
                    T = None
                item = (sig, [], T)
                pending[key] = item
            item[1].append(e)
            if item[2] is not None and e.end - item[1][0].pos >= item[2]:
                flush(key)                                  # the group is full
        for key in list(pending):
            flush(key)
        # groups left open: a glissando/slide ends with the next note of its voice; other groups end with their
        # last member (a group really cut by the end of the excerpt ends there anyway)
        notes_by_voice: Dict[tuple, List[Elem]] = {}
        for e in part.elems:
            if e.kind == "note" and not e.chord and not e.grace:
                notes_by_voice.setdefault((e.staff, e.voice), []).append(e)
        for g in open_.values():
            g.open_end = True
            if g.kind == "gliss":
                seq = notes_by_voice.get((g.staff, g.voice), [])
                nxt = [n for n in seq if n.pos > g.start]
                g.end = max(g.end, nxt[0].end if nxt else g.end)
            else:
                last = max((by_idx[i] for i in g.members if i in by_idx), key=lambda x: x.pos + x.dur)
                g.end = max(g.end, last.pos + last.dur)
            groups.append(g)
    return groups


def _pitch_key(el: ET.Element) -> tuple:
    p = el.find("pitch")
    if p is None:
        return ("unpitched",)
    return (p.findtext("step"), p.findtext("alter") or "0", p.findtext("octave"))


def midi_of(el: ET.Element) -> Optional[float]:
    p = el.find("pitch")
    if p is None:
        return None
    step = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[p.findtext("step")]
    alter = float(p.findtext("alter") or 0)
    return 12 * (int(p.findtext("octave")) + 1) + step + alter

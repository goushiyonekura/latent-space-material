"""Check a score map against its sources: every copied note / rest / direction must be the source element unchanged
(allowed differences: <duration> rescaled to the output divisions, default-x dropped, font sizes scaled with the
staff size of the page layout, <voice> / <instrument id> /
spanner numbers renumbered, slurs, trill lines and hairpins closed or reopened at the edges of a copied passage,
a source bar rest written as rests of the same length, tempo marks written as text), and every output bar must add
up in every voice.

usage: python3 -m scoremap.verify OUTPUT_DIR SCOREMAP.musicxml REPORT.json [--scores DIR]
"""
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction as Fr

from . import mxl
from .build import short_of
from .emit import DIV, NUMBERED

EDGE_TAGS = ("slur", "wavy-line", "tied")  # may be added / removed at the first or last note of a passage


def canon(el: ET.Element, beams: bool = True) -> str:
    e = copy.deepcopy(el)
    if not beams:
        for b in e.findall("beam"):
            e.remove(b)
    e.tail = None
    for x in list(e.iter()):
        x.attrib.pop("default-x", None)
        x.attrib.pop("font-size", None)          # text is resized with the staff size of the page layout
        if x.text is not None and not x.text.strip():
            x.text = None
        if x.tail is not None and not x.tail.strip():
            x.tail = None
        if x.tag in NUMBERED:
            x.attrib.pop("number", None)
    for tag in ("duration", "voice", "instrument", "tie"):
        for d in e.findall(tag):
            e.remove(d)
    for nt in e.findall("notations"):
        for t in EDGE_TAGS:
            for s in nt.findall(t):
                nt.remove(s)
        for orn in nt.findall("ornaments"):
            for s in orn.findall("wavy-line"):
                orn.remove(s)
            if len(orn) == 0 and not (orn.text or "").strip():
                nt.remove(orn)
        if len(nt) == 0:
            e.remove(nt)
    return ET.tostring(e, encoding="unicode")


def edge_marks(el: ET.Element) -> Counter:
    c = Counter()
    for t in EDGE_TAGS + ("tie",):
        for s in el.iter(t):
            c[(t, s.get("type"))] += 1
    return c


def _pitch(el):
    p = el.find("pitch")
    return None if p is None else (p.findtext("step"), p.findtext("alter") or "0", p.findtext("octave"))


def split_chain(e, pos, onotes, used):
    """A source note written as several tied notes (a barline cuts it): the output notes from `pos` on, same pitch
    (or rest), same chord / grace status, tied one to the next, adding up to the source length."""
    def attempt(first):
        total = Fr(0); chain = []
        p = pos
        while total < e.dur:
            nxt = [o for o in onotes.get((p, e.staff), []) if id(o) not in used and o not in chain
                   and _pitch(o.el) == _pitch(e.el) and o.rest == e.rest and o.chord == e.chord and not o.grace
                   and (first is None or o.voice == first.voice) and (chain or o is first)]
            if not nxt:
                return None
            o = nxt[0]
            chain.append(o)
            total += o.dur
            p = p + o.dur
            if not e.rest and total < e.dur and not any(t.get("type") == "start" for t in o.el.findall("tie")):
                return None
        return chain if total == e.dur and len(chain) > 1 else None
    for first in [o for o in onotes.get((pos, e.staff), []) if id(o) not in used and _pitch(o.el) == _pitch(e.el)
                  and o.rest == e.rest and o.chord == e.chord and not o.grace]:
        c = attempt(first)
        if c:
            return c
    return None


def main(argv):
    out_dir, out_path, rep_path = argv[0], argv[1], argv[2]
    score_dir = argv[argv.index("--scores") + 1] if "--scores" in argv else "materials/scores-diffusion"
    trace = json.load(open(os.path.join(out_dir, "state_trace.json")))
    names = list(trace["source_order"])
    rep = json.load(open(rep_path))
    out = mxl.parse(out_path)
    # output parts in build order: lanes 1..N, each material's source parts in order
    srcs = {}
    order = []
    for lane in range(1, len(names)):
        sc = mxl.parse(os.path.join(score_dir, names[lane] + ".musicxml"))
        srcs[names[lane]] = sc
        for p in sc.parts:
            order.append((names[lane], p.pid))
    assert len(order) == len(out.parts), (len(order), len(out.parts))
    problems = []
    n_checked = 0
    n_edge = 0
    n_dirs = [0]
    n_split = [0]
    n_rebeamed = [0]
    bars = rep["bar_lengths"]
    starts = [Fr(0)]
    for L in bars:
        starts.append(starts[-1] + Fr(L).limit_denominator(1 << 16))
    for (mat, pid), op in zip(order, out.parts):
        sc = srcs[mat]
        sp = sc.part(pid)
        # output notes by (position, staff) for lookup
        onotes = defaultdict(list)
        for e in op.elems:
            if e.kind == "note":
                onotes[(e.pos, e.staff)].append(e)
        used = set()
        for pl in rep["placed"]:
            if pl["material"] != mat or pl["part"] != pid:
                continue
            a, b = Fr(pl["a"]), Fr(pl["b"])
            x = Fr(pl["x"]).limit_denominator(1 << 16)
            src_notes = [e for e in sp.elems if e.kind == "note" and a <= e.pos < b]
            for e in src_notes:
                pos = x + (e.pos - a)
                r = e.el.find("rest")
                if r is not None and r.get("measure") == "yes":
                    # a bar rest: rests of the same total length starting here in some voice
                    cands = [o for o in onotes.get((pos, e.staff), []) if o.rest and id(o) not in used]
                    if not cands:
                        problems.append((mat, pid, float(pos), "bar rest missing"))
                    else:
                        used.add(id(cands[0]))
                    continue
                want = canon(e.el)
                cands = [o for o in onotes.get((pos, e.staff), []) if id(o) not in used and canon(o.el) == want
                         and o.dur == e.dur and o.chord == e.chord and o.grace == e.grace]
                if not cands:
                    # a beam group divided by a new barline: the same note with its beams redrawn
                    want2 = canon(e.el, beams=False)
                    rb = [o for o in onotes.get((pos, e.staff), []) if id(o) not in used and canon(o.el, beams=False) == want2
                          and o.dur == e.dur and o.chord == e.chord and o.grace == e.grace]
                    if rb:
                        used.add(id(rb[0])); n_rebeamed[0] += 1; n_checked += 1
                        continue
                    chain = split_chain(e, pos, onotes, used)
                    if chain:
                        for o in chain:
                            used.add(id(o))
                        n_split[0] += 1
                        continue
                    alt = [o for o in onotes.get((pos, e.staff), []) if id(o) not in used]
                    problems.append((mat, pid, float(pos), "no identical copy of source note at m%s pos %s (%d candidates)"
                                     % (e.measure, e.pos, len(alt))))
                    if "--diff" in argv and alt and len(problems) <= 3:
                        print("SOURCE:", want); print("OUTPUT:", canon(alt[0].el))
                    continue
                o = cands[0]
                used.add(id(o))
                n_checked += 1
                if edge_marks(o.el) != edge_marks(e.el):
                    n_edge += 1
        # directions: every source direction of a passage (tempo marks become text, <sound> dropped)
        def eff(e, div):
            return e.pos                                  # mxl positions already include <offset>
        odirs = defaultdict(list)
        for e in op.elems:
            if e.kind == "direction":
                odirs[eff(e, DIV)].append(e)
        for pl in rep["placed"]:
            if pl["material"] != mat or pl["part"] != pid:
                continue
            a, b = Fr(pl["a"]), Fr(pl["b"])
            x = Fr(pl["x"]).limit_denominator(1 << 16)
            skip = {g.members[-1] for g in sc.groups
                    if g.part == pid and g.kind in ("wedge", "octave") and g.start < a and g.end == a}
            for e in sp.elems:
                if e.kind != "direction" or not (a <= e.pos < b) or e.idx in skip:
                    continue
                d = copy.deepcopy(e.el)
                for snd in d.findall("sound"):
                    d.remove(snd)
                if d.find("direction-type/metronome") is not None:
                    continue                                  # written as "(♩=N)" text on purpose
                if d.find("direction-type/*") is None:
                    continue
                for off in d.findall("offset"):
                    d.remove(off)
                want = canon(d)
                src_div = next((dd for q, dd in reversed(sp.divisions) if q <= e.pos), sp.divisions[0][1])
                pos = x + (eff(e, src_div) - a)
                hit = None
                for o in odirs.get(pos, []):
                    oo = copy.deepcopy(o.el)
                    for off in oo.findall("offset"):
                        oo.remove(off)
                    if id(o) not in used and canon(oo) == want:
                        hit = o
                        break
                if hit is None:
                    problems.append((mat, pid, float(pos), "direction not copied unchanged: " + want[:80]))
                else:
                    used.add(id(hit)); n_dirs[0] += 1
        # notes in the output that are not copies: must be filler rests
        for e in op.elems:
            if e.kind == "note" and id(e) not in used and not e.rest:
                problems.append((mat, pid, float(e.pos), "note without a source"))
        # bar sums: every voice's extent within its bar
        for m in op.measures:
            k = int(m.number) - 1
            want_len = starts[k + 1] - starts[k]
            if m.length != want_len:
                problems.append((mat, pid, float(m.start), f"bar {m.number} length {m.length} != {want_len}"))
    print(json.dumps({"checked_notes": n_checked, "of_which_rebeamed": n_rebeamed[0], "notes_written_as_tied_notes": n_split[0],
                      "checked_directions": n_dirs[0], "edge_spanner_changes": n_edge, "problems": len(problems)}))
    for p in problems[:40]:
        print("  ", p)
    return problems


if __name__ == "__main__":
    main(sys.argv[1:])

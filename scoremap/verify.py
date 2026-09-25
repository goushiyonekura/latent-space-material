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
from .emit import DIV, NUMBERED, key_alter, key_fifths

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
    for tag in ("duration", "voice", "instrument", "tie", "staff"):   # staff: a moved passage sits on another staff
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


def split_chain(e, pos, onotes, used, staff=None):
    """A source note written as several tied notes (a barline cuts it): the output notes from `pos` on, same pitch
    (or rest), same chord / grace status, tied one to the next, adding up to the source length."""
    st = e.staff if staff is None else staff

    def attempt(first):
        total = Fr(0); chain = []
        p = pos
        while total < e.dur:
            nxt = [o for o in onotes.get((p, st), []) if id(o) not in used and o not in chain
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
    for first in [o for o in onotes.get((pos, st), []) if id(o) not in used and _pitch(o.el) == _pitch(e.el)
                  and o.rest == e.rest and o.chord == e.chord and not o.grace]:
        c = attempt(first)
        if c:
            return c
    return None


def _strip_value(el: ET.Element) -> ET.Element:
    """A note without its written value, tuplet marks and accidental (for approximated tuplet notes)."""
    e = copy.deepcopy(el)
    for tag in ("type", "dot", "duration", "time-modification", "accidental"):
        for x in e.findall(tag):
            e.remove(x)
    for nt in list(e.findall("notations")):
        for t in list(nt.findall("tuplet")):
            nt.remove(t)
        if len(nt) == 0:
            e.remove(nt)
    return e


def _strip_acc(el: ET.Element) -> ET.Element:
    e = copy.deepcopy(el)
    for x in e.findall("accidental"):
        e.remove(x)
    return e


def _subtract(iv, cuts):
    segs = [iv]
    for (c0, c1) in sorted(cuts):
        nxt = []
        for (s0, s1) in segs:
            if c1 <= s0 or c0 >= s1:
                nxt.append((s0, s1))
            else:
                if s0 < c0:
                    nxt.append((s0, c0))
                if c1 < s1:
                    nxt.append((c1, s1))
        segs = nxt
    return segs


def main(argv):
    out_dir, out_path, rep_path = argv[0], argv[1], argv[2]
    score_dir = argv[argv.index("--scores") + 1] if "--scores" in argv else "materials/scores-diffusion"
    trace = json.load(open(os.path.join(out_dir, "state_trace.json")))
    names = list(trace["source_order"])
    rep = json.load(open(rep_path))
    out = mxl.parse(out_path)
    srcs = {}
    for lane in range(1, len(names)):
        srcs[names[lane]] = mxl.parse(os.path.join(score_dir, names[lane] + ".musicxml"))
    # output parts: from the report (staff reduction drops parts), else lanes 1..N with each material's parts
    if "out_parts" in rep:
        order = [(o["material"], o["part"], o["pid"]) for o in rep["out_parts"]]
    else:
        order = []
        for lane in range(1, len(names)):
            for p in srcs[names[lane]].parts:
                order.append((names[lane], p.pid, f"P{len(order) + 1}"))
    assert len(order) == len(out.parts), (len(order), len(out.parts))
    natural = {(mat, pid): opid for (mat, pid, opid) in order}
    by_out = defaultdict(list)                                # output part -> placed pieces written in it
    for pl in rep["placed"]:
        by_out[pl.get("out_part") or natural.get((pl["material"], pl["part"]))].append(pl)
    moves = rep.get("merge", {}).get("moves", [])
    moved_ext = defaultdict(list)                             # (output part, staff) -> extents of moved passages
    for mv in moves:
        moved_ext[(mv["out_part"], mv["target_staff"])].append((Fr(mv["x"]).limit_denominator(1 << 16),
                                                                Fr(mv["end"]).limit_denominator(1 << 16)))
    problems = []
    edges = {}                                                # (material, part, x, source idx) -> edge marks written
    n_checked = 0
    n_edge = 0
    n_dirs = [0]
    n_split = [0]
    n_rebeamed = [0]
    n_acc = [0]
    n_rest_removed = [0]
    n_moved = 0
    rule = {"moved_overlaps_note": 0, "moved_overlaps_moved": 0, "moved_inside_phrase": 0, "moved_wrong_kind": 0}
    bars = rep["bar_lengths"]
    starts = [Fr(0)]
    for L in bars:
        starts.append(starts[-1] + Fr(L).limit_denominator(1 << 16))
    synthetic = {o["pid"] for o in rep.get("out_parts", []) if o.get("synthetic")}
    policy = rep.get("merge", {}).get("policy", "move" if moves else None)
    cap = rep.get("merge", {}).get("voices")
    n_approx = [0]
    for (mat, pid, opid), op in zip(order, out.parts):
        assert op.pid == opid, (op.pid, opid)
        tgt_part = srcs[mat].part(pid) if opid not in synthetic else None
        dst_fifths = key_fifths(tgt_part) if tgt_part is not None else 0
        onotes = defaultdict(list)                            # (position, staff) -> output notes
        for e in op.elems:
            if e.kind == "note":
                onotes[(e.pos, e.staff)].append(e)
        used = set()
        owner = {}                                            # id(output note) -> index of its placed piece
        pieces = by_out[opid]
        for pk, pl in enumerate(pieces):
            pmat, ppid = pl["material"], pl["part"]
            sc = srcs[pmat]
            sp = sc.part(ppid)
            shift = int(pl.get("staff_shift", 0))
            moved = shift > 0 or (pmat, ppid) != (mat, pid) or opid in synthetic
            src_fifths = key_fifths(sp)
            if moved:
                n_moved += 1
            a, b = Fr(pl["a"]), Fr(pl["b"])
            x = Fr(pl["x"]).limit_denominator(1 << 16)
            open_tuplets = [g for g in sc.groups if g.part == ppid and g.kind == "tuplet" and (g.start < a < g.end or g.start < b < g.end)] \
                if rep.get("cut_rules", {}).get("cut_tuplets") else []
            src_notes = [e for e in sp.elems if e.kind == "note" and a <= e.pos < b]
            for e in src_notes:
                pos = x + (e.pos - a)
                st = e.staff + shift
                r = e.el.find("rest")
                in_open_tuplet = e.el.find("time-modification") is not None and any(g.start <= e.pos < g.end for g in open_tuplets)
                cut = moved_ext.get((opid, st), []) if not moved else []
                r_iv = (pos, x + (min(e.end, b) - a))
                under_moved = r is not None and any(c0 < r_iv[1] and r_iv[0] < c1 for c0, c1 in cut)
                if r is not None and r.get("measure") == "yes":
                    # a bar rest: rests of the same total length starting here in some voice
                    cands = [o for o in onotes.get((pos, st), []) if o.rest and id(o) not in used]
                    if cands:
                        used.add(id(cands[0]))
                    elif under_moved:
                        n_rest_removed[0] += 1
                    else:
                        problems.append((pmat, ppid, float(pos), "bar rest missing"))
                    continue
                want = canon(e.el)
                cands = [o for o in onotes.get((pos, st), []) if id(o) not in used and canon(o.el) == want
                         and o.dur == e.dur and o.chord == e.chord and o.grace == e.grace]
                if not cands and in_open_tuplet:
                    want_a = canon(_strip_value(e.el), beams=False)
                    ap = [o for o in onotes.get((pos, st), []) if id(o) not in used and canon(_strip_value(o.el), beams=False) == want_a
                          and o.chord == e.chord and o.grace == e.grace and (o.grace or o.chord or Fr(0) < o.dur <= e.dur)]
                    if ap:
                        used.add(id(ap[0])); owner[id(ap[0])] = pk; n_approx[0] += 1; n_checked += 1
                        continue
                if not cands:
                    # a beam group divided by a new barline: the same note with its beams redrawn
                    want2 = canon(e.el, beams=False)
                    rb = [o for o in onotes.get((pos, st), []) if id(o) not in used and canon(o.el, beams=False) == want2
                          and o.dur == e.dur and o.chord == e.chord and o.grace == e.grace]
                    if rb:
                        used.add(id(rb[0])); owner[id(rb[0])] = pk; n_rebeamed[0] += 1; n_checked += 1
                        continue
                    chain = split_chain(e, pos, onotes, used, st)
                    if chain:
                        for o in chain:
                            used.add(id(o)); owner[id(o)] = pk
                        n_split[0] += 1
                        if not e.rest and e.el.find("accidental") is None and chain[0].el.find("accidental") is not None:
                            n_acc[0] += 1                     # the first tied note carries an added accidental
                            if not moved:
                                problems.append((pmat, ppid, float(pos), "accidental added to an unmoved note"))
                        edges[(pmat, ppid, float(x), e.idx)] = tuple(sorted(edge_marks(chain[0].el).items()))
                        continue
                    # a moved note with a key-signature accidental made explicit: identical apart from that
                    if moved and not e.rest and src_fifths != dst_fifths and e.el.find("accidental") is None:
                        pt = e.el.find("pitch")
                        step = pt.findtext("step"); alter = int(float(pt.findtext("alter") or 0))
                        exp = {2: "double-sharp", 1: "sharp", 0: "natural", -1: "flat", -2: "flat-flat"}.get(alter)
                        ok_key = alter == key_alter(src_fifths, step) and key_alter(src_fifths, step) != key_alter(dst_fifths, step)
                        ac = [o for o in onotes.get((pos, st), []) if id(o) not in used and o.el.findtext("accidental") == exp
                              and (canon(_strip_acc(o.el)) == want or canon(_strip_acc(o.el), beams=False) == canon(e.el, beams=False))
                              and o.dur == e.dur and o.chord == e.chord and o.grace == e.grace]
                        if ac and ok_key:
                            used.add(id(ac[0])); owner[id(ac[0])] = pk; n_acc[0] += 1; n_checked += 1
                            edges[(pmat, ppid, float(x), e.idx)] = tuple(sorted(edge_marks(ac[0].el).items()))
                            continue
                        if ac and not ok_key:
                            problems.append((pmat, ppid, float(pos), f"accidental {exp} added to a note the key did not alter"))
                            continue
                    if under_moved:
                        # a rest of this part under a moved passage: removed, or split around it
                        kept = _subtract(r_iv, cut)
                        for (s0, s1) in kept:
                            ks = [o for o in onotes.get((s0, st), []) if o.rest and id(o) not in used]
                            if not ks:
                                problems.append((pmat, ppid, float(s0), "kept part of a split rest missing"))
                            else:
                                used.add(id(ks[0]))
                        n_rest_removed[0] += 1
                        continue
                    alt = [o for o in onotes.get((pos, st), []) if id(o) not in used]
                    problems.append((pmat, ppid, float(pos), "no identical copy of source note at m%s pos %s (%d candidates)"
                                     % (e.measure, e.pos, len(alt))))
                    if "--diff" in argv and alt and len(problems) <= 3:
                        print("SOURCE:", want); print("OUTPUT:", canon(alt[0].el))
                    continue
                o = cands[0]
                used.add(id(o)); owner[id(o)] = pk
                n_checked += 1
                edges[(pmat, ppid, float(x), e.idx)] = tuple(sorted(edge_marks(o.el).items()))
                if edge_marks(o.el) != edge_marks(e.el):
                    n_edge += 1
        # directions: every source direction of a passage (tempo marks become text, <sound> dropped)
        odirs = defaultdict(list)
        for e in op.elems:
            if e.kind == "direction":
                odirs[e.pos].append(e)
        for pl in pieces:
            pmat, ppid = pl["material"], pl["part"]
            sc = srcs[pmat]
            sp = sc.part(ppid)
            a, b = Fr(pl["a"]), Fr(pl["b"])
            x = Fr(pl["x"]).limit_denominator(1 << 16)
            skip = {g.members[-1] for g in sc.groups
                    if g.part == ppid and g.kind in ("wedge", "octave") and g.start < a and g.end == a}
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
                for stf in d.findall("staff"):                # a moved direction sits on the output staff
                    d.remove(stf)
                want = canon(d)
                pos = x + (e.pos - a)
                hit = None
                for o in odirs.get(pos, []):
                    oo = copy.deepcopy(o.el)
                    for off in oo.findall("offset"):
                        oo.remove(off)
                    for stf in oo.findall("staff"):
                        oo.remove(stf)
                    if id(o) not in used and canon(oo) == want:
                        hit = o
                        break
                if hit is None:
                    problems.append((pmat, ppid, float(pos), "direction not copied unchanged: " + want[:80]))
                else:
                    used.add(id(hit)); n_dirs[0] += 1
        # notes in the output that are not copies: must be filler rests
        for e in op.elems:
            if e.kind == "note" and id(e) not in used and not e.rest:
                pt_ = e.el.find("pitch")
                desc = f"{opid} st{e.staff} v{e.voice} {pt_.findtext('step') if pt_ is not None else '?'}{pt_.findtext('alter') or '' if pt_ is not None else ''}{pt_.findtext('octave') if pt_ is not None else ''} {e.el.findtext('type')} dur={e.dur}" \
                    + (" chord" if e.chord else "") + (" grace" if e.grace else "") + (" tm" if e.el.find("time-modification") is not None else "")
                problems.append((mat, pid, float(e.pos), "note without a source: " + desc))
        # packed parts (voices per staff): at most `cap` voices with an element at any instant per staff, the
        # passages that overlap on a staff use one clef, and every passage is of the part's instrument kind
        if opid in synthetic:
            kind_of = {"Violin": "vn", "Viola": "va", "Violoncello": "vc"}
            my_kind = kind_of[op.score_part.findtext("part-name").split()[0].replace("Vn", "Violin").replace("Va", "Viola").replace("Vc", "Violoncello")]
            for st in range(1, op.staves + 1):
                ivs = defaultdict(list)                           # voice -> element intervals (notes and rests)
                own = defaultdict(list)                           # piece -> element intervals
                for e in op.elems:
                    if e.kind == "note" and e.staff == st and not e.chord and not e.grace and e.dur > 0 and id(e) in used:
                        ivs[e.voice].append((e.pos, e.end))
                        own[owner.get(id(e), -1)].append((e.pos, e.end))
                pts = sorted({t for v in ivs.values() for iv in v for t in iv})
                worst = 0; worst_shared = 0
                for t in pts:
                    n = sum(1 for v in ivs.values() if any(s0 <= t < e0 for s0, e0 in v))
                    npc = sum(1 for v in own.values() if any(s0 <= t < e0 for s0, e0 in v))
                    worst = max(worst, n)
                    if npc >= 2:                                  # the cap binds only where passages share the staff
                        worst_shared = max(worst_shared, n)
                if cap and worst_shared > cap:
                    problems.append((mat, pid, 0.0, f"{opid} staff {st}: {worst_shared} voices at once from several passages (cap {cap})"))
                rule["max_voices"] = max(rule.get("max_voices", 0), worst)
            unit_cap = rep.get("merge", {}).get("unit_cap")
            if unit_cap:
                # at most unit_cap passages at once per staff (from the report's placement)
                by_st = defaultdict(list)
                for pl in pieces:
                    st_ = 1 + int(pl.get("staff_shift", 0)) + (1 if pl["material"].startswith("lumin") else 0)
                    by_st[st_].append((Fr(pl["x"]).limit_denominator(1 << 16), Fr(pl["end"]).limit_denominator(1 << 16), id(pl["material"] + pl["part"] + str(pl["x"]) + str(pl["pass"]))))
                for st_, ivs_ in by_st.items():
                    frag_ivs = defaultdict(list)                  # pieces of one fragment-part count once
                    for (x_, e_, k_) in ivs_:
                        frag_ivs[k_].append((x_, e_))
                    pts_ = sorted({t for (x_, e_, k_) in ivs_ for t in (x_, e_)})
                    worst_ = max((sum(1 for (x_, e_, k_) in ivs_ if x_ <= t < e_) for t in pts_), default=0)
                    if worst_ > unit_cap:
                        problems.append((mat, pid, 0.0, f"{opid} staff {st_}: {worst_} passages at once (unit cap {unit_cap})"))
                    rule["max_passages"] = max(rule.get("max_passages", 0), worst_)
            for pk, pl in enumerate(pieces):
                sname = srcs[pl["material"]].part(pl["part"]).name.split()[0]
                if kind_of.get(sname) != my_kind:
                    rule["moved_wrong_kind"] += 1
                    if rule["moved_wrong_kind"] <= 3:
                        problems.append((pl["material"], pl["part"], float(pl["x"]), f"wrong kind: {sname} on {opid} ({my_kind})"))
            # clef agreement of overlapping passages (from the sources)
            def clef_segs(pl):
                sp_ = srcs[pl["material"]].part(pl["part"]); a_, b_ = Fr(pl["a"]), Fr(pl["b"]); x_ = Fr(pl["x"]).limit_denominator(1 << 16)
                nst = 2 if pl["material"].startswith("lumin") else 1
                cl = sorted(((e.pos, e.idx, (c.findtext("sign"), c.findtext("line"))) for e in sp_.elems if e.kind == "attributes"
                             for c in e.el.findall("clef") if int(c.get("number", "1")) == nst), key=lambda t: (t[0], t[1]))
                cur = None
                for (q, i, c) in cl:
                    if q <= a_:
                        cur = c
                segs = []; tp, kp = x_, cur
                for (q, i, c) in cl:
                    if a_ < q < b_:
                        segs.append((tp, x_ + (q - a_), kp)); tp, kp = x_ + (q - a_), c
                segs.append((tp, x_ + (b_ - a_), kp))
                return segs
            by_staff = defaultdict(list)
            for pl in pieces:
                by_staff[1 + int(pl.get("staff_shift", 0)) + (0 if not pl["material"].startswith("lumin") else 1)].append(pl)
            for st, pls in by_staff.items():
                segs = [(pl, clef_segs(pl)) for pl in pls]
                for i in range(len(segs)):
                    for j in range(i + 1, len(segs)):
                        for (a1, b1, k1) in segs[i][1]:
                            for (a2, b2, k2) in segs[j][1]:
                                if a1 < b2 and a2 < b1 and k1 != k2 and k1 is not None and k2 is not None:
                                    rule["clef_clash"] = rule.get("clef_clash", 0) + 1
                                    problems.append((segs[i][0]["material"], segs[i][0]["part"], float(max(a1, a2)),
                                                     f"{opid} staff {st}: overlapping passages with different clefs"))
        # the staff-reduction rule, checked on the output itself: a moved passage shares its staff with no note of
        # another passage, lies inside no phrase of this part, overlaps no other moved passage, same instrument kind
        kind = {"Violin": "vn", "Viola": "va", "Violoncello": "vc"}
        for pk, pl in enumerate(pieces) if opid not in synthetic else []:
            shift = int(pl.get("staff_shift", 0))
            if not (shift > 0 or (pl["material"], pl["part"]) != (mat, pid)):
                continue
            x = Fr(pl["x"]).limit_denominator(1 << 16)
            end = Fr(pl["end"]).limit_denominator(1 << 16)
            st = 1 + shift
            sname = srcs[pl["material"]].part(pl["part"]).name.split()[0]
            if kind.get(sname) != kind.get(tgt_part.name.split()[0]):
                rule["moved_wrong_kind"] += 1
            for e in op.elems:
                if e.kind == "note" and not e.rest and e.staff == st and e.dur > 0 and owner.get(id(e), pk) != pk \
                        and e.pos < end and x < e.end:
                    rule["moved_overlaps_note"] += 1
                    problems.append((pl["material"], pl["part"], float(e.pos), "moved passage overlaps a note of another passage"))
            for qk, ql in enumerate(pieces):
                if qk == pk:
                    continue
                qx = Fr(ql["x"]).limit_denominator(1 << 16); qend = Fr(ql["end"]).limit_denominator(1 << 16)
                q_shift = int(ql.get("staff_shift", 0))
                q_moved = q_shift > 0 or (ql["material"], ql["part"]) != (mat, pid)
                if q_moved and 1 + q_shift == st and qx < end and x < qend:
                    rule["moved_overlaps_moved"] += 1
                    problems.append((pl["material"], pl["part"], float(x), "two moved passages overlap"))
                if not q_moved:
                    qa, qb = Fr(ql["a"]), Fr(ql["b"])
                    sp = srcs[ql["material"]].part(ql["part"])
                    ns = [(qx + (e.pos - qa), qx + (min(e.end, qb) - qa)) for e in sp.elems
                          if e.kind == "note" and not e.rest and not e.grace and e.dur > 0 and e.staff + q_shift == st and qa <= e.pos < qb]
                    if ns:
                        f0, l1 = min(s0 for s0, e0 in ns), max(e0 for s0, e0 in ns)
                        if f0 < end and x < l1:
                            rule["moved_inside_phrase"] += 1
                            problems.append((pl["material"], pl["part"], float(x), "moved passage inside a phrase of this part"))
        # bar sums: every voice's extent within its bar
        for m in op.measures:
            k = int(m.number) - 1
            want_len = starts[k + 1] - starts[k]
            if m.length != want_len:
                problems.append((mat, pid, float(m.start), f"bar {m.number} length {m.length} != {want_len}"))
    res = {"checked_notes": n_checked, "of_which_rebeamed": n_rebeamed[0], "notes_written_as_tied_notes": n_split[0],
           "checked_directions": n_dirs[0], "edge_spanner_changes": n_edge, "problems": len(problems)}
    if moves or n_moved:
        res.update({"moved_passages": n_moved, "accidentals_added": n_acc[0], "rests_removed_or_split": n_rest_removed[0],
                    "tuplet_notes_approximated": n_approx[0], "rule_violations": rule})
    print(json.dumps(res, ensure_ascii=False))
    for p in problems[:40]:
        print("  ", p)
    if "--dump-edges" in argv:
        import pickle
        pickle.dump(edges, open(argv[argv.index("--dump-edges") + 1], "wb"))
    return problems


if __name__ == "__main__":
    main(sys.argv[1:])

"""Build the score mapping of one generated output.

usage: python3 -m scoremap.build OUTPUT_DIR OUT.musicxml [--scores DIR] [--align DIR] [--report OUT.json]
                                 [--cuts q,q,...] [--pages q,q,...] [--merge] [--staff-mm MM]

OUTPUT_DIR: a generated run (state_trace.json).  Source scores: DIR/<material>.musicxml.  Alignments:
DIR/<section>.pkl (audio seconds <-> score position per section; see dev/align_sections.py).
--merge: staff reduction (scoremap.merge): passages of the papillon / prism parts move whole into the lower staff
of a luminasity instrument of the same kind where it is silent; source staves that empty out are dropped.
--voices N [--lum-cap M] [--unit-cap U]: packing (scoremap.merge.pack): every passage on synthetic staves of its
instrument kind, up to N voices at once per staff (luminasity passages at most M per staff; at most U passages at
once per staff, U=1: one passage at a time with its own voices); nothing is cut unless the cut rules say so.
--staff-mm: staff height of the page layout (default 2.9 mm; with --merge the largest height that fits A2; with
--voices 4.5 mm).
--tight: notate each fragment from the clean position nearest to its heard start to the one nearest to its heard end
(default: the cheapest option, which completes phrases).  --cut-beams / --cut-tuplets: a cut may fall inside a
beam group (re-beamed) / a tuplet (written with approximate plain values).  User rules of 2026-09-25.
"""
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction as Fr

import numpy as np

from latent_space.curves import TrackCurve
from . import emit, extract, layout, merge, mxl

SECTION_OF = [(r"^luminusity_7_", "lum7"), (r"^luminasity_8_", "lum8"), (r"^luminasity_9_", "lum9"),
              (r"^luminasity_10_", "lum10"), (r"^luminasity_11_", "lum11"), (r"^luminasity_14_", "lum14"),
              (r"^papillon(_viola|_violin)?_1_", "pap1"), (r"^papillon(_viola|_violin)?_2_", "pap2"),
              (r"^papillon(_viola|_violin)?_4_", "pap4"), (r"^papillon(_viola|_violin)?_5_", "pap5"),
              (r"^prism(_cello|_viola)?_1_", "prism1"), (r"^prism(_cello|_viola)?_2_", "prism2")]
NAME_RE = re.compile(r"^(.*)_(\d{2})\.(\d{2})\.(\d{3})_(\d{2})\.(\d{2})(?:\.(\d{3}))?$")


def section_of(material: str) -> str:
    for rx, sec in SECTION_OF:
        if re.match(rx, material):
            return sec
    raise KeyError(material)


def short_of(material: str) -> str:
    m = NAME_RE.match(material)
    return m.group(1) if m else material


def map_bars(trace: dict, min_peak: float = 1e-3):
    """(lane, t0, t1, s0, s1) seconds of every placement of every material (goal excluded): the same bars as the
    material map page (exact gain segments + position clips, split at jumps and loop wraps)."""
    fs = int(trace["sample_rate"])
    inputs = {int(x["track"]): x for x in trace["input_paths_and_hashes"]}
    out = []
    for lane, entries in enumerate(trace["curve_segments_per_track"]):
        if lane == 0:
            continue
        length = int(inputs[lane]["frames"])
        curve = TrackCurve.from_list(entries)
        segs = sorted((e for e in entries if e.get("type") not in ("BUMPS", "CLIPS")), key=lambda x: x["start_frame"])
        cuts = sorted({int(c.out_start) for c in curve.clips})
        iv = []
        for s in segs:
            if max(float(s["start_gain"]), float(s["end_gain"])) <= 1e-9:
                continue
            a, b = int(s["start_frame"]), int(s["end_frame"])
            if iv and a <= iv[-1][1]:
                iv[-1][1] = max(iv[-1][1], b)
            else:
                iv.append([a, b])
        for a, b in iv:
            pts = [a] + [c for c in cuts if a < c < b] + [b]
            for x0, x1 in zip(pts[:-1], pts[1:]):
                y = x0
                while y < x1:
                    q = int(curve.positions(np.array([y]), length)[0])
                    y1 = min(x1, y + (length - q))
                    if y1 - y >= int(0.01 * fs):
                        grid = np.linspace(y, y1 - 1, num=min(64, max(2, (y1 - y) // 441))).astype(np.int64)
                        if float(curve.values(grid).max()) >= min_peak:
                            out.append((lane, y / fs, y1 / fs, q / fs, (q + y1 - y) / fs))
                    y = y1
    return out


def merge_continuous(bars, tol=0.05, max_gap=0.5):
    """Join consecutive bars of one material that continue the same playback (the gap in piece time equals the gap in
    the file, at most max_gap s): short dropouts of the gain, not jumps."""
    by = {}
    for b in sorted(bars, key=lambda x: (x[0], x[1])):
        by.setdefault(b[0], []).append(list(b))
    out = []
    for lane, bs in by.items():
        cur = bs[0]
        for nb in bs[1:]:
            gt, gs = nb[1] - cur[2], nb[3] - cur[4]
            if 0 <= gt <= max_gap and abs(gs - gt) <= tol:
                cur[2], cur[4] = nb[2], nb[4]
            else:
                out.append(tuple(cur)); cur = nb
        out.append(tuple(cur))
    return sorted(out, key=lambda x: (x[1], x[0]))


def main(argv):
    out_dir, out_path = argv[0], argv[1]
    get = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
    score_dir = get("--scores", "materials/scores-diffusion")
    align_dir = get("--align", "dev/out/align")
    report_path = get("--report", None)
    trace = json.load(open(os.path.join(out_dir, "state_trace.json")))
    names = list(trace["source_order"])
    total_s = float(trace["form"]["total_frames"]) / int(trace["sample_rate"])

    scores, cps, acts, aligns = {}, {}, {}, {}
    frags = defaultdict(list)
    missing = []
    cut_beams, cut_tuplets, tight = "--cut-beams" in argv, "--cut-tuplets" in argv, "--tight" in argv
    for (lane, t0, t1, s0, s1) in merge_continuous(map_bars(trace)):
        mat = names[lane]
        if lane not in scores:
            scores[lane] = mxl.parse(os.path.join(score_dir, mat + ".musicxml"))
            cps[lane] = {p.pid: extract.clean_points(scores[lane], p.pid, cut_beams=cut_beams, cut_tuplets=cut_tuplets)
                         for p in scores[lane].parts}
            acts[lane] = {p.pid: extract.Activity(scores[lane], p.pid) for p in scores[lane].parts}
        sec = section_of(mat)
        if sec not in aligns:
            aligns[sec] = pickle.load(open(os.path.join(align_dir, sec + ".pkl"), "rb"))
        al = aligns[sec]
        opts, drops, ref_q = extract.options(scores[lane], cps[lane], acts[lane], al, s0, s1)
        if not opts:
            missing.append((mat, s0, s1))
            continue
        frags[lane].append(extract.Fragment(lane, mat, t0, t1, s0, s1, [], ref_q, opts, drops))

    layout.LOST_WEIGHT = float(get("--tight-lost-weight", "1.0"))
    placed, dropped = layout.place(frags, scores, select="tight" if tight else "cheapest")
    do_merge = "--merge" in argv
    voices = int(get("--voices", "0"))
    lum_cap = int(get("--lum-cap", "0")) or None
    unit_cap = int(get("--unit-cap", "0")) or None            # 1: one passage at a time per staff
    if unit_cap and not voices:
        voices = 4
    plan = None
    specs, pack_stats = [], {}
    if voices:
        specs, pack_stats = merge.pack(placed, names, cap=voices, lum_cap=lum_cap, unit_cap=unit_cap)
        for sp in specs:
            for u in sp.units:
                for pl in u:
                    pl.dst = sp.key
                    pl.staff_shift = 1 if (sp.staves == 2 and not merge.is_luminasity(pl.frag.material)) else 0
    if do_merge:
        plan = merge.plan(placed, names)
        for pl in placed:
            t = plan.moves.get(id(pl))
            if t is not None:
                pl.dst = (t.lane, t.pid)
                pl.staff_shift = 1                            # a one-staff source part into staff 2
        for pl in plan.left_out:
            placed.remove(pl)
    removed = {(s.lane, s.pid) for s in plan.removed} if plan else set()
    end = max([total_s] + [float(p.end) for p in placed])
    total = Fr(int(np.ceil(end)))
    bars = layout.barlines(placed, total)
    # extra barlines chosen by the user (page breaks inside long bars; notes they cut are tied)
    cuts = [Fr(c).limit_denominator(1 << 10) for c in get("--cuts", "").split(",") if c.strip()]
    moved_cuts = []
    if cuts:
        # a note cut by a barline is written as tied notes (a glissando keeps its start on the last of them); a beam,
        # tuplet or two-note tremolo must not be cut (user criterion for the page breaks, HANDOFF §6.7)
        groups = [(p.out(max(g.start, p.piece.a)), p.out(min(g.end, p.piece.b)), g.kind) for p in placed
                  for g in p.score.groups if g.kind in ("beam", "tuplet", "tremolo") and not g.open_end
                  and g.part == p.piece.pid and g.start < p.piece.b and g.end > p.piece.a]
        bad = sorted({(float(c), k) for c in cuts for (s_, e_, k) in groups if s_ < c < e_})
        if bad and "--snap-cuts" in argv:
            # move each offending cut to the nearest position (1/8 grid, within 3 s) that cuts nothing hard
            moved_cuts = []
            for i, c in enumerate(cuts):
                if not any(s_ < c < e_ for (s_, e_, k) in groups):
                    continue
                best = None
                for step in range(1, 25):
                    for cand in (c - Fr(step, 8), c + Fr(step, 8)):
                        if Fr(0) < cand < total and not any(s_ < cand < e_ for (s_, e_, k) in groups):
                            best = cand
                            break
                    if best is not None:
                        break
                if best is None:
                    raise SystemExit(f"--snap-cuts: no clean position within 3 s of {float(c)}")
                moved_cuts.append((float(c), float(best)))
                cuts[i] = best
            print("snap-cuts:", moved_cuts)
            bad = []
        if bad:
            raise SystemExit(f"--cuts inside a beam / tuplet / two-note tremolo (alignment changed?): {bad}")
        bars = sorted(set(bars) | set(cuts))
        if voices:
            # packed staves leave few clean positions: an automatic barline that makes a bar shorter than one
            # quarter is dropped (the user's cuts stay)
            keep = []
            for i, b_ in enumerate(bars):
                if b_ in set(cuts) or i == 0 or i == len(bars) - 1:
                    keep.append(b_); continue
                prev_ = keep[-1]
                nxt_ = next((c for c in bars[i + 1:] if c in set(cuts) or c == bars[-1]), bars[-1])
                if b_ - prev_ < 1 or nxt_ - b_ < 1:
                    continue
                keep.append(b_)
            bars = keep

    parts = []
    n = 0
    KIND_LABEL = {"violin": ("Violin", "Vn"), "viola": ("Viola", "Va"), "cello": ("Violoncello", "Vc")}
    LUM_CLEF = {"violin": "G2", "viola": "C3", "cello": "F4"}
    for lane in range(1, len(names)):
        if lane not in scores:
            scores[lane] = mxl.parse(os.path.join(score_dir, names[lane] + ".musicxml"))
    if voices:
        first_seen = set()
        for sp in specs:
            n += 1
            label, ab = KIND_LABEL[sp.kind]
            if sp.staves == 2:
                clefs = {1: "G2", 2: LUM_CLEF[sp.kind]}
            else:
                u0 = min(sp.units, key=lambda u: u[0].x)
                ck = merge.Unit(u0, u0[0].frag.material).clef_at(u0[0].x)
                clefs = {1: f"{ck[0]}{ck[1]}" if ck and f"{ck[0]}{ck[1]}" in emit.CLEF_OF else LUM_CLEF[sp.kind]}
            part = emit.synthetic_part(f"P{n}", f"{label} {sp.index}", f"{ab} {sp.index}", label, sp.staves, clefs,
                                       zero_line_staff=1 if sp.staves == 2 else None)
            mv = {st: str(4 * (st - 1) + 1) for st in range(1, sp.staves + 1)}
            op = emit.OutPart(f"P{n}", -1, label, label, sp.key[1], scores[min(scores)], part, sp.kind not in first_seen,
                              sp.staves, mv, 1)
            op.synthetic = True
            first_seen.add(sp.kind)
            parts.append(op)
    for lane in range(1, len(names)) if not voices else []:
        sc = scores[lane]
        first_left = True
        for k, p in enumerate(sc.parts):
            if (lane, p.pid) in removed:
                continue                                      # every passage of this part moved to a luminasity staff
            n += 1
            mv = {}
            for st in range(1, p.staves + 1):
                c = Counter(e.voice for e in p.elems if e.kind == "note" and e.staff == st)
                mv[st] = c.most_common(1)[0][0] if c else "1"
            vmax = max([int(e.voice) for e in p.elems if e.kind == "note" and e.voice.isdigit()] + [1])
            parts.append(emit.OutPart(f"P{n}", lane, names[lane], short_of(names[lane]), p.pid, sc, p, first_left,
                                      p.staves, mv, vmax))
            first_left = False
    defaults = None
    first = scores[min(scores)]
    defaults = first.root.find("defaults")
    title = f"Score map — {os.path.basename(out_dir.rstrip('/'))}"
    # page starts: the extra barlines, plus any further positions asked for (they must be barlines)
    extra_pages = [Fr(c).limit_denominator(1 << 10) for c in get("--pages", "").split(",") if c.strip()]
    if "--snap-cuts" in argv and cuts:
        snapped = {Fr(a).limit_denominator(1 << 10): Fr(b).limit_denominator(1 << 10) for a, b in moved_cuts}
        extra_pages = [snapped.get(c, c) for c in extra_pages]
    missing_bar = [float(c) for c in extra_pages if c not in set(bars)]
    if missing_bar:
        raise SystemExit(f"--pages positions are not barlines: {missing_bar}")
    if "--staff-mm" in argv:
        lay = emit.Layout.for_staff(float(get("--staff-mm", "2.9")), packed=bool(voices))
    elif voices:
        lay = emit.Layout.for_staff(min(4.5, emit.fit_staff_mm(parts, emit.Layout.for_staff(4.5, packed=True))), packed=True)
    elif do_merge:
        lay = emit.Layout.for_staff(emit.fit_staff_mm(parts))
    else:
        lay = emit.Layout()
    emit.write(out_path, title, parts, placed, bars, defaults, dropped=dropped, layout=lay,
               page_breaks=cuts + extra_pages, pages_only=bool(voices))

    conflicts = [(op.material, x, st) for op in parts for (x, st) in op.clef_conflicts]
    kept = {(id(p.frag)) for p in placed}
    lost_whole = [f for v in frags.values() for f in v if id(f) not in kept]
    rep = {"fragments": sum(len(v) for v in frags.values()), "pieces": len(placed), "bars": len(bars) - 1,
           "total_quarters": float(total), "parts": len(parts), "staves": sum(p.staves for p in parts),
           "fragments_left_out": len(lost_whole), "part_pieces_left_out": len(dropped),
           "shortened": int(sum(1 for p in placed if p.cut_from is not None)),
           "late_start": int(sum(1 for p in placed if p.head_from is not None and p.first)),
           "bar_lengths": [float(b - a) for a, b in zip(bars[:-1], bars[1:])], "missing": missing,
           "left_out": [(f.material, f.t0, f.t1, f.s0, f.s1, pid) for f, pid in dropped],
           "layered_pieces": int(sum(1 for p in placed if p.layer > 0)), "max_layer": max([p.layer for p in placed] + [0]),
           "clef_conflicts": conflicts,
           "max_voices_at_once": max([op.max_simultaneous for op in parts] + [0]),
           "cuts": [float(c) for c in cuts], "notes_cut_by_barlines": int(sum(op.notes_cut for op in parts)),
           "beam_groups_cut_by_barlines": int(sum(op.beams_cut for op in parts)),
           "dense_parts": [(op.material, op.src_pid, op.max_simultaneous) for op in parts if op.max_simultaneous > 4],
           "staff_mm": lay.staff_mm,
           "cut_rules": {"tight": tight, "cut_beams": cut_beams, "cut_tuplets": cut_tuplets},
           "beams_cut_at_edges": int(sum(op.beams_cut_at_edges for op in parts)),
           "tuplets_dissolved": int(sum(op.tuplets_dissolved for op in parts)),
           "tuplet_notes_approximated": int(sum(op.tuplet_notes_approximated for op in parts))}
    # how the notation matches the heard spans (audio seconds): per fragment-part, first / last piece
    heard_out, unheard_in = 0.0, 0.0
    by_fp = defaultdict(list)
    for p in placed:
        by_fp[(id(p.frag), p.piece.pid)].append(p)
    for (fid, pid), ps in by_fp.items():
        ps.sort(key=lambda p: p.x)
        f = ps[0].frag
        al = aligns[section_of(f.material)]
        ta = al.t_at(float(ps[0].piece.a), ps[0].piece.passno)
        tb = al.t_at(float(ps[-1].piece.b), ps[-1].piece.passno)
        if ta == ta and tb == tb:
            heard_out += max(0.0, ta - f.s0) + max(0.0, f.s1 - tb)
            unheard_in += max(0.0, f.s0 - ta) + max(0.0, tb - f.s1)
    rep["heard_left_out_s"] = round(heard_out, 1)
    rep["unheard_written_s"] = round(unheard_in, 1)
    rep["heard_total_s"] = round(sum(f.s1 - f.s0 for v in frags.values() for f in v), 1)
    out_pid = {(op.lane, op.src_pid): op.pid for op in parts}
    if voices:
        by_part = []
        for sp, op in zip(specs, parts):
            items = []
            for u in sp.units:
                pl = u[0]
                ma, ba = extract.measure_beat(pl.score, pl.piece.a)
                mb, bb = extract.measure_beat(u[-1].score, u[-1].piece.b)
                items.append({"material": pl.frag.material, "part": pl.piece.pid, "from": f"m{ma}/{float(ba):g}",
                              "to": f"m{mb}/{float(bb):g}", "x": float(min(p.x for p in u)), "end": float(max(p.end for p in u)),
                              "staff": 1 + pl.staff_shift})
            by_part.append({"pid": op.pid, "kind": sp.kind, "staves": sp.staves, "passages": items})
        rep["merge"] = {"policy": "pack", "voices": voices, "lum_cap": lum_cap, "unit_cap": unit_cap, "stats": pack_stats,
                        "rule": merge.pack.__doc__.strip().split("\n")[0], "parts": by_part,
                        "accidentals_added": int(sum(op.accidentals_added for op in parts)),
                        "moved_pieces": int(sum(op.moved_pieces for op in parts))}
        rep["staves_removed"] = 72 - sum(op.staves for op in parts)
        print("pack:", json.dumps({"voices": voices, "lum_cap": lum_cap, "unit_cap": unit_cap, "staves": sum(op.staves for op in parts), "parts": len(parts),
                                   "stats": pack_stats, "accidentals_added": rep["merge"]["accidentals_added"]}, ensure_ascii=False))
    if plan is not None:
        tname = {(op.lane, op.src_pid): f"{op.short} {op.part.name}" for op in parts}
        moves = []
        for s_ in plan.removed:
            for pl in s_.pieces:
                t = plan.moves[id(pl)]
                ma, ba = extract.measure_beat(pl.score, pl.piece.a)
                mb, bb = extract.measure_beat(pl.score, pl.piece.b)
                moves.append({"source_material": pl.frag.material, "source_part": pl.piece.pid, "source_name": s_.name,
                              "from": f"m{ma}/{float(ba):g}", "to": f"m{mb}/{float(bb):g}", "pass": pl.piece.passno,
                              "x": float(pl.x), "end": float(pl.end), "target_material": t.material, "target_part": t.pid,
                              "target_name": t.name, "out_part": out_pid[(t.lane, t.pid)], "target_staff": 1 + pl.staff_shift,
                              "luminasity_passages_touched": [(float(a_), float(b_)) for a_, b_ in plan.touched[id(pl)]]})
        rep["merge"] = {"rule": merge.__doc__.strip().split("\n\n")[0],
                        "removed_parts": [{"material": s_.material, "part": s_.pid, "name": s_.name} for s_ in plan.removed],
                        "kept_parts": [{"material": s_.material, "part": s_.pid, "name": s_.name} for s_ in plan.stay],
                        "moves": moves, "passages_touching_luminasity": plan.cost[0], "target_switches": plan.cost[1],
                        "left_out": [(pl.frag.material, pl.piece.pid, float(pl.x), float(pl.end)) for pl in plan.left_out],
                        "deleted_rests": [d for op in parts for d in op.deleted_rests],
                        "accidentals_added": int(sum(op.accidentals_added for op in parts)),
                        "moved_pieces": int(sum(op.moved_pieces for op in parts))}
        rep["staves_removed"] = len(plan.removed)
        summary = {k: v for k, v in rep["merge"].items() if k in ("passages_touching_luminasity", "target_switches",
                                                                   "accidentals_added", "moved_pieces")}
        summary["deleted_rests"] = len(rep["merge"]["deleted_rests"])
        summary["removed_parts"] = len(plan.removed)
        summary["kept_parts"] = [s_.name for s_ in plan.stay]
        print("merge:", json.dumps(summary, ensure_ascii=False))
    rep["out_parts"] = [{"pid": op.pid, "material": (None if op.synthetic else op.material), "part": (None if op.synthetic else op.src_pid),
                         "name": (op.part.name if op.synthetic else f"{op.short} {op.part.name}"), "staves": op.staves,
                         "first_of_material": op.first_of_material, "synthetic": op.synthetic} for op in parts]
    print(json.dumps({k: v for k, v in rep.items() if k not in ("bar_lengths", "left_out", "missing", "clef_conflicts", "dense_parts",
                                                                "merge", "out_parts")}, ensure_ascii=False))
    if report_path:
        detail = []
        for p in placed:
            ma, ba = extract.measure_beat(p.score, p.piece.a)
            mb, bb = extract.measure_beat(p.score, p.piece.b)
            d = {"material": p.frag.material, "part": p.piece.pid, "t0": p.frag.t0, "t1": p.frag.t1, "s0": p.frag.s0, "s1": p.frag.s1,
                 "a": str(p.piece.a), "b": str(p.piece.b), "pass": p.piece.passno, "from": f"m{ma}/{float(ba):g}",
                 "to": f"m{mb}/{float(bb):g}", "x": float(p.x), "end": float(p.end), "delay": p.delay,
                 "cut_from": (str(p.cut_from) if p.cut_from is not None else None),
                 "head_from": (str(p.head_from) if p.head_from is not None else None), "layer": p.layer}
            if plan is not None or voices:
                d["out_part"] = out_pid[p.dst or (p.frag.lane, p.piece.pid)]
                d["staff_shift"] = p.staff_shift
            detail.append(d)
        rep["placed"] = detail
        json.dump(rep, open(report_path, "w"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])

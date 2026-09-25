"""Build the score mapping of one generated output.

usage: python3 -m scoremap.build OUTPUT_DIR OUT.musicxml [--scores DIR] [--align DIR] [--report OUT.json]

OUTPUT_DIR: a generated run (state_trace.json).  Source scores: DIR/<material>.musicxml.  Alignments:
DIR/<section>.pkl (audio seconds <-> score position per section; see dev/align_sections.py).
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
from . import emit, extract, layout, mxl

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
    for (lane, t0, t1, s0, s1) in merge_continuous(map_bars(trace)):
        mat = names[lane]
        if lane not in scores:
            scores[lane] = mxl.parse(os.path.join(score_dir, mat + ".musicxml"))
            cps[lane] = {p.pid: extract.clean_points(scores[lane], p.pid) for p in scores[lane].parts}
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

    placed, dropped = layout.place(frags, scores)
    end = max([total_s] + [float(p.end) for p in placed])
    total = Fr(int(np.ceil(end)))
    bars = layout.barlines(placed, total)
    # extra barlines chosen by the user (page breaks inside long bars; notes they cut are tied)
    cuts = [Fr(c).limit_denominator(1 << 10) for c in get("--cuts", "").split(",") if c.strip()]
    if cuts:
        bars = sorted(set(bars) | set(cuts))

    parts = []
    n = 0
    for lane in range(1, len(names)):
        if lane not in scores:
            scores[lane] = mxl.parse(os.path.join(score_dir, names[lane] + ".musicxml"))
        sc = scores[lane]
        for k, p in enumerate(sc.parts):
            n += 1
            mv = {}
            for st in range(1, p.staves + 1):
                c = Counter(e.voice for e in p.elems if e.kind == "note" and e.staff == st)
                mv[st] = c.most_common(1)[0][0] if c else "1"
            vmax = max([int(e.voice) for e in p.elems if e.kind == "note" and e.voice.isdigit()] + [1])
            parts.append(emit.OutPart(f"P{n}", lane, names[lane], short_of(names[lane]), p.pid, sc, p, k == 0,
                                      p.staves, mv, vmax))
    defaults = None
    first = scores[min(scores)]
    defaults = first.root.find("defaults")
    title = f"Score map — {os.path.basename(out_dir.rstrip('/'))}"
    # page starts: the extra barlines, plus any further positions asked for (they must be barlines)
    extra_pages = [Fr(c).limit_denominator(1 << 10) for c in get("--pages", "").split(",") if c.strip()]
    missing_bar = [float(c) for c in extra_pages if c not in set(bars)]
    if missing_bar:
        raise SystemExit(f"--pages positions are not barlines: {missing_bar}")
    emit.write(out_path, title, parts, placed, bars, defaults, dropped=dropped, layout=emit.Layout(),
               page_breaks=cuts + extra_pages)

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
           "dense_parts": [(op.material, op.src_pid, op.max_simultaneous) for op in parts if op.max_simultaneous > 4]}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("bar_lengths", "left_out", "missing", "clef_conflicts", "dense_parts")}, ensure_ascii=False))
    if report_path:
        detail = []
        for p in placed:
            ma, ba = extract.measure_beat(p.score, p.piece.a)
            mb, bb = extract.measure_beat(p.score, p.piece.b)
            detail.append({"material": p.frag.material, "part": p.piece.pid, "t0": p.frag.t0, "t1": p.frag.t1, "s0": p.frag.s0, "s1": p.frag.s1,
                           "a": str(p.piece.a), "b": str(p.piece.b), "pass": p.piece.passno, "from": f"m{ma}/{float(ba):g}",
                           "to": f"m{mb}/{float(bb):g}", "x": float(p.x), "end": float(p.end), "delay": p.delay,
                           "cut_from": (str(p.cut_from) if p.cut_from is not None else None),
                           "head_from": (str(p.head_from) if p.head_from is not None else None), "layer": p.layer})
        rep["placed"] = detail
        json.dump(rep, open(report_path, "w"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])

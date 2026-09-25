"""Sounding-pitch sequence of a score (as scoremap.align models it): q, measure, sounding pitch name(s), duration.
usage: python3 dev/soundseq.py SECTION [stem] [q0 q1]"""
import sys
from collections import defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import align, mxl
import align_sections as AS
from f0segments import nm

sec = sys.argv[1]; stem = int(sys.argv[2]) if len(sys.argv) > 2 else 0
q0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0; q1 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
sc = mxl.parse(AS.S + AS.SECTIONS[sec][stem] + ".musicxml")
ev = align.score_events(sc)
by = defaultdict(list)
for st, en, md, w, att in zip(ev.starts, ev.ends, ev.midis, ev.weights, ev.attack):
    if q0 <= st < q1 and w >= 0.5:
        by[round(st, 3)].append((md, en - st, att))
meas = sc.parts[0].measures
line = []
for q in sorted(by):
    m = next((mm.number for mm in meas if float(mm.start) <= q < float(mm.start + mm.length)), "?")
    names = "+".join(sorted({nm(md) + ("" if att else "~") for md, d, att in by[q]}))
    d = max(d for md, d, att in by[q])
    line.append(f"q{q:g}(m{m}):{names}/{d:.2g}")
print("  ".join(line))

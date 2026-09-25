"""Compact per-bar summary of a source score: start, length, notes (pitch/duration/marks), words, dynamics.
usage: python3 dev/barsummary.py SECTION [stem] [from_bar to_bar]"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl
import align_sections as AS

sec = sys.argv[1]; stem = int(sys.argv[2]) if len(sys.argv) > 2 else 0
b0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; b1 = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 6
sc = mxl.parse(AS.S + AS.SECTIONS[sec][stem] + ".musicxml")
for part in sc.parts:
    by_m = {}
    for e in part.elems:
        by_m.setdefault(e.measure, []).append(e)
    for m in part.measures:
        try:
            if not (b0 <= int(m.number) <= b1):
                continue
        except ValueError:
            pass
        items = []
        for e in sorted(by_m.get(m.number, []), key=lambda x: (x.pos, x.idx)):
            if e.kind == "note":
                if e.rest:
                    if e.el.get("print-object") != "no":
                        items.append(f"R{float(e.dur):g}")
                    continue
                p = e.el.find("pitch")
                ps = p.findtext("step") + {"1": "#", "-1": "b", "0": ""}.get(p.findtext("alter") or "0", p.findtext("alter") or "") + p.findtext("octave") if p is not None else "?"
                nh = e.el.findtext("notehead") or ""
                mk = ("◇" if nh == "diamond" else "") + ("~" if e.el.find("notations/ornaments/tremolo") is not None else "") + \
                     ("g" if e.grace else "")
                items.append(("+" if e.chord else "") + ps + mk + ("" if e.chord or e.grace else f"/{float(e.dur):g}"))
            elif e.kind == "direction":
                for w in e.el.iter("words"):
                    if (w.text or "").strip():
                        items.append(f"[{w.text.strip()[:14]}]")
                for d in e.el.iter("dynamics"):
                    items.append("<" + "".join(c.tag for c in d) + ">")
            elif e.kind == "barline" and e.el.find("repeat") is not None:
                items.append("||:" if e.el.find("repeat").get("direction") == "forward" else ":||")
        print(f"{part.pid} m{m.number:>3s} q{float(m.start):6.2f} L{float(m.length):g}: " + " ".join(items))

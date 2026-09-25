"""List notes per part for given bars: python3 dev/barnotes.py SECTION bar [bar ...]"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl
import align_sections as A
sec = sys.argv[1]; bars = sys.argv[2:]
s = mxl.parse(A.S + A.SECTIONS[sec][0] + ".musicxml")
for part in s.parts:
    rows = []
    for e in part.elems:
        if e.kind == "note" and e.measure in bars and not e.chord:
            p = e.el.find("pitch")
            ps = (p.findtext("step") + (p.findtext("alter") or "") + p.findtext("octave")) if p is not None else "R"
            if e.rest and e.el.get("print-object") == "no":
                continue
            rows.append(f"m{e.measure}s{e.staff} q{float(e.pos):.2f}{'g' if e.grace else ''}:{ps}")
        if e.kind == "direction" and e.measure in bars:
            for w in e.el.iter("words"):
                if w.text and w.text.strip():
                    rows.append(f"[{w.text.strip()[:12]}@q{float(e.pos):.2f}]")
            for d in e.el.iter("dynamics"):
                rows.append(f"[{''.join(c.tag for c in d)}@q{float(e.pos):.2f}]")
    print(part.name + ": " + " ".join(rows))
print("bar starts:", [(m.number, float(m.start), float(m.length)) for m in s.parts[0].measures])

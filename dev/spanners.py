"""Open/close balance of spanners per part, in time order (positions from scoremap.mxl, stops before starts at one
moment).  usage: python3 dev/spanners.py FILE [-v]"""
import sys
from collections import Counter
sys.path.insert(0, ".")
from scoremap import mxl

SPEC = (("wavy-line", ("start",), ("stop",), "staff"), ("octave-shift", ("up", "down"), ("stop",), "staff"),
        ("wedge", ("crescendo", "diminuendo"), ("stop",), "staff"), ("slur", ("start",), ("stop",), "voice"),
        ("tuplet", ("start",), ("stop",), "voice"), ("glissando", ("start",), ("stop",), "voice"),
        ("slide", ("start",), ("stop",), "voice"), ("bracket", ("start",), ("stop",), "staff"),
        ("dashes", ("start",), ("stop",), "staff"))


def check(path):
    sc = mxl.parse(path)
    tot = Counter(); per = {}
    for part in sc.parts:
        evs = []
        for e in part.elems:
            if e.kind not in ("note", "direction"):
                continue
            off = e.el.find("offset")
            pos = e.pos
            for tag, st, sp, scope in SPEC:
                for x in e.el.iter(tag):
                    t = x.get("type")
                    if t not in st and t not in sp:
                        continue
                    key = (tag, x.get("number", "1"), e.staff if scope == "staff" else (e.staff, e.voice))
                    # stops first at the same moment; a note's own stop and start: stop first
                    evs.append((pos, 0 if t in sp else 1, e.idx, key, "stop" if t in sp else "start"))
        evs.sort(key=lambda v: (v[0], v[1], v[2]))
        open_ = {}; issues = []
        for pos, _, idx, key, t in evs:
            if t == "start":
                if key in open_:
                    issues.append((float(pos), "restart", key))
                open_[key] = pos
            else:
                if key not in open_:
                    issues.append((float(pos), "stop without start", key))
                else:
                    del open_[key]
        for key, pos in open_.items():
            issues.append((float(pos), "never closed", key))
        c = Counter((i[1], i[2][0]) for i in issues)
        tot.update(c); per[part.pid] = (c, issues)
    return tot, per


if __name__ == "__main__":
    tot, per = check(sys.argv[1])
    if "-v" in sys.argv:
        for pid, (c, issues) in per.items():
            if c:
                print(pid, dict(c), issues[:6])
    print("TOTAL", dict(tot))

"""Ties: every tie start must be followed, right at its end, by a note of the same pitch with a tie stop (same part,
staff, voice), and every tie stop must be preceded by such a start.  usage: python3 dev/tiecheck.py FILE"""
import sys
from collections import defaultdict, Counter
sys.path.insert(0, ".")
from scoremap import mxl


def check(path):
    sc = mxl.parse(path)
    bad = Counter(); ex = []
    for part in sc.parts:
        by = defaultdict(list)
        for e in part.elems:
            if e.kind == "note" and not e.rest and not e.grace:
                p = e.el.find("pitch")
                if p is None:
                    continue
                key = (p.findtext("step"), p.findtext("alter") or "0", p.findtext("octave"))
                by[(e.staff, e.voice, key)].append(e)
        for k, notes in by.items():
            starts = {e.pos + e.dur for e in notes if any(t.get("type") == "start" for t in e.el.findall("tie"))}
            stops = {e.pos for e in notes if any(t.get("type") == "stop" for t in e.el.findall("tie"))}
            for q in starts - stops:
                bad["start without stop"] += 1; ex.append((part.pid, float(q), k, "start"))
            for q in stops - starts:
                bad["stop without start"] += 1; ex.append((part.pid, float(q), k, "stop"))
    return bad, ex


if __name__ == "__main__":
    bad, ex = check(sys.argv[1])
    print(dict(bad)); print(ex[:10])

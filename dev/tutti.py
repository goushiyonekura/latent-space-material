"""Score attack strength per position: how many parts attack there and how loud (current dynamic, accents)."""
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
from scoremap import mxl

DYN = {"pppp": 0.1, "ppp": 0.15, "pp": 0.25, "p": 0.4, "mp": 0.55, "mf": 0.7, "f": 0.9, "ff": 1.1, "fff": 1.3,
       "fp": 0.9, "sfz": 1.2, "sf": 1.1, "sfp": 1.1, "fz": 1.1, "rfz": 1.1}


def attack_strength(sc):
    out = defaultdict(float)
    for part in sc.parts:
        dyn_ev = []
        for e in part.elems:
            if e.kind == "direction":
                for d in e.el.iter("dynamics"):
                    for c in d:
                        if c.tag in DYN:
                            dyn_ev.append((e.pos, e.idx, c.tag))
        dyn_ev.sort()
        seen = set()
        for e in part.elems:
            if e.kind != "note" or e.rest or e.chord:
                continue
            if any(t.get("type") == "stop" for t in e.el.findall("tie")):
                continue
            # current dynamic
            lev = 0.5
            for (p, i, tag) in dyn_ev:
                if p <= e.pos:
                    lev = DYN[tag]
                else:
                    break
            acc = 0.0
            for art in e.el.iter("articulations"):
                for c in art:
                    if c.tag in ("accent", "strong-accent"):
                        acc = 0.3
            for d in e.el.iter("dynamics"):
                for c in d:
                    if c.tag in DYN:
                        lev = max(lev, DYN[c.tag])
            pizz = False
            key = (round(float(e.pos), 4), part.pid)
            if key in seen:
                continue
            seen.add(key)
            out[round(float(e.pos), 4)] += lev + acc
    qs = np.array(sorted(out)); w = np.array([out[q] for q in qs])
    return qs, w

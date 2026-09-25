"""Per-bar timing check: shift delta (s) that best aligns the mapped score attacks with audio onset strength.
usage: python3 dev/snapcheck.py SECTION"""
import sys, pickle, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import mxl, align
import align_sections as A, onsets as O
sec = sys.argv[1]
names, scores, al = A.run(sec)
f = O.feats(sec, 0)
fl = np.convolve(f["flux"], np.ones(3) / 3, mode="same")
t_a = f["t"]
sc = scores[0]
DYN = {"f": 2.0, "ff": 2.5, "fff": 3, "sfz": 2.5, "sf": 2.2, "fp": 2.0, "mf": 1.5, "mp": 1.2, "p": 1.0, "pp": 0.8, "ppp": 0.7, "fz": 2.2}
# attack list with weights (dynamic markings near the attack raise the weight)
dyn_at = []
for part in sc.parts:
    for e in part.elems:
        if e.kind == "direction":
            for d in e.el.iter("dynamics"):
                for c in d:
                    dyn_at.append((float(e.pos), DYN.get(c.tag, 1.0)))
att = []
for part in sc.parts:
    for e in part.elems:
        if e.kind == "note" and not e.rest and not e.chord and not e.grace and not any(t.get("type") == "stop" for t in e.el.findall("tie")):
            w = 1.0 + max([dw for (q, dw) in dyn_at if abs(q - float(e.pos)) < 1e-6] + [0.0])
            att.append((float(e.pos), w, e.measure))
deltas = np.arange(-0.8, 0.81, 0.02)
for m in sc.parts[0].measures:
    sel = [(q, w) for (q, w, mm) in att if mm == m.number]
    if not sel:
        continue
    ts = np.array([al.t_at(q, 0) for q, w in sel]); ws = np.array([w for q, w in sel])
    score = []
    for d in deltas:
        idx = np.clip(np.searchsorted(t_a, ts + d), 0, len(fl) - 1)
        score.append(float((fl[idx] * ws).sum() / ws.sum()))
    score = np.array(score); k = int(np.argmax(score))
    print(f"m{m.number:>3} start {al.t_at(float(m.start),0):6.2f}s attacks {len(sel):3d}  best shift {deltas[k]:+.2f}s  "
          f"(fit {score[k]:.2f} vs at 0: {score[len(deltas)//2]:.2f})")

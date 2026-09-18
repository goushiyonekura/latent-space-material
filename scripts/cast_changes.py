"""Cast statistics of a capped run: changes of the sounding set per minute, dwell times."""
import csv, json, sys
import numpy as np
for d in sys.argv[1:]:
    rows = list(csv.reader(open(d + "/gain_curves.csv")))
    names = rows[0][2:]
    g = np.array([[float(x) for x in r] for r in rows[1:]])
    t, goal, lv = g[:, 0], g[:, 1], g[:, 2:]
    mats = sorted({n.split("#")[0] for n in names})
    on = np.stack([np.any(lv[:, [k for k, n in enumerate(names) if n.split("#")[0] == m]] > 1e-6, axis=1) for m in mats], axis=1)
    step = max(1, int(round(0.5 / (t[1] - t[0]))))
    idx = np.arange(step - 1, len(t), step)              # end of every 0.5 s commit block
    free = goal[idx] < 0.01
    sets = [frozenset(np.where(on[i])[0]) for i in idx]
    ch = sum(1 for a, b, f in zip(sets[:-1], sets[1:], free[1:]) if f and a != b)
    minutes = free.sum() * 0.5 / 60.0
    dwell = []
    for m in range(len(mats)):
        run = 0
        for i, f in zip(idx, free):
            if on[i, m] and f:
                run += 1
            elif run:
                dwell.append(run * 0.5); run = 0
    print(f"{d:58s} cast changes/min {ch / max(minutes, 1e-9):5.1f} | median dwell {np.median(dwell) if dwell else 0:4.1f} s mean {np.mean(dwell) if dwell else 0:4.1f} s | mean sounding {np.mean([len(s) for s, f in zip(sets, free) if f]):.2f}")

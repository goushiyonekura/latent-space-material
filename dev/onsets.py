"""Strongest onsets near a time: python3 dev/onsets.py SECTION t_center [radius] [stem]"""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "dev")
from scoremap import align
import align_sections as A
_cache = {}
def feats(sec, stem=0):
    k = (sec, stem)
    if k not in _cache:
        _cache[k] = align.audio_features(A.A + A.SECTIONS[sec][stem] + ".wav")
    return _cache[k]
def near(sec, t, r=0.6, stem=0, n=4):
    f = feats(sec, stem)
    m = (f["t"] >= t - r) & (f["t"] <= t + r)
    idx = np.where(m)[0]
    fl = f["flux"]
    pk = [i for i in idx if 0 < i < len(fl) - 1 and fl[i] >= fl[i - 1] and fl[i] >= fl[i + 1]]
    pk = sorted(pk, key=lambda i: -fl[i])[:n]
    return [(round(float(f["t"][i]), 3), round(float(fl[i]), 2), round(float(f["db"][i]), 1)) for i in sorted(pk)]
if __name__ == "__main__":
    sec = sys.argv[1]; t = float(sys.argv[2]); r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    stem = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    for x in near(sec, t, r, stem): print(x)

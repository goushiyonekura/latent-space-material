"""Material map page for one generated output: which material sounds where in the piece, and from which
position of its file.  Bars come from the trace's exact gain curves and position clips (the renderer's own
rules: from out_start the source plays from src_start 1:1, modulo its length); one bar = one stretch of a
material with a continuous source position (split at jumps and loop wraps).  Times in the original
recording = file-name start (MM.SS.mmm) + offset inside the file.

usage: python3 dev/material_map.py OUTPUT_DIR PAGE_DIR [--title T] [--kbps 128] [--mix-kbps 192] [--no-audio]
Writes PAGE_DIR/index.html (page fragment for publishing) and PAGE_DIR/audio/*.mp4 (afconvert AAC in an MPEG-4 container).
"""
import json, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_space.curves import TrackCurve

NAME_RE = re.compile(r"^(.*)_(\d{2})\.(\d{2})\.(\d{3})_(\d{2})\.(\d{2})(?:\.(\d{3}))?$")


def parse_name(name):
    """'luminasity_8_02.29.381_02.47.541' -> ('luminasity_8', start_ms, end_ms); no times -> (name, 0, None)."""
    m = NAME_RE.match(name)
    if not m:
        return name, 0, None
    s = (int(m.group(2)) * 60 + int(m.group(3))) * 1000 + int(m.group(4))
    e = (int(m.group(5)) * 60 + int(m.group(6))) * 1000 + int(m.group(7) or 0)
    return m.group(1), s, e


def family(short):
    for f in ("lumin", "papillon", "prism"):
        if short.startswith(f):
            return {"lumin": "luminasity"}.get(f, f)
    return "goal" if short.lower().startswith("goal") or "GOAL" in short else "other"


def sounding_intervals(segs, eps=1e-9):
    """Merged [a, b) frame intervals where the gain curve is above zero (HOLD/Q5 segments)."""
    out = []
    for s in sorted(segs, key=lambda x: x["start_frame"]):
        if max(float(s["start_gain"]), float(s["end_gain"])) <= eps:
            continue
        a, b = int(s["start_frame"]), int(s["end_frame"])
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def bars_for_track(entries, length, fs, min_peak=1e-3):
    curve = TrackCurve.from_list(entries)
    segs = [e for e in entries if e.get("type") not in ("BUMPS", "CLIPS")]
    cuts = sorted({int(c.out_start) for c in curve.clips})
    bars = []
    for a, b in sounding_intervals(segs):
        pts = [a] + [c for c in cuts if a < c < b] + [b]
        for x0, x1 in zip(pts[:-1], pts[1:]):
            pieces, y = [], x0
            while y < x1:                                   # split at every loop wrap (file end -> 0)
                q = int(curve.positions(np.array([y]), length)[0])
                y_end = min(x1, y + (length - q))
                pieces.append((y, y_end, q))
                y = y_end
            for y0, y1, q0 in pieces:
                if y1 - y0 < int(0.01 * fs):                # < 10 ms: ramp slivers at a cut
                    continue
                grid = np.linspace(y0, y1 - 1, num=min(64, max(2, (y1 - y0) // 441))).astype(np.int64)
                g = curve.values(grid)
                if float(g.max()) < min_peak:                # inaudible placements (e.g. 1e-7 holds) are not mapped
                    continue
                bars.append({"a": y0, "b": y1, "src": q0, "peak": float(g.max()), "mean": float(g.mean())})
    return bars


def encode(src, dst, kbps):
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", str(int(kbps) * 1000), src, dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv):
    out_dir, page_dir = argv[0].rstrip("/"), argv[1].rstrip("/")
    kbps = int(argv[argv.index("--kbps") + 1]) if "--kbps" in argv else 128
    mix_kbps = int(argv[argv.index("--mix-kbps") + 1]) if "--mix-kbps" in argv else 192
    tr = json.load(open(os.path.join(out_dir, "state_trace.json")))
    fs = int(tr["sample_rate"])
    names = list(tr["source_order"])
    inputs = {int(x["track"]): x for x in tr["input_paths_and_hashes"]}
    cs = tr["curve_segments_per_track"]
    lanes, bars = [], []
    for i, name in enumerate(names):
        short, s_ms, e_ms = parse_name(name)
        fam = "goal" if i == 0 else family(short)
        if i == 0:
            short = os.path.splitext(os.path.basename(inputs[0]["path"]))[0].replace("material_", "")
        length = int(inputs[i]["frames"])
        lanes.append({"id": f"t{i:02d}", "name": name if i else short, "short": short, "family": fam,
                      "file_start_ms": s_ms, "file_end_ms": e_ms, "file_ms": int(round(length * 1000 / fs))})
        for bb in bars_for_track(cs[i], length, fs):
            bars.append([i, int(round(bb["a"] * 1000 / fs)), int(round(bb["b"] * 1000 / fs)),
                         int(round(bb["src"] * 1000 / fs)), int(round((bb["src"] + bb["b"] - bb["a"]) * 1000 / fs)),
                         int(round(100 * bb["peak"])), int(round(100 * bb["mean"]))])
    phases = [{"name": p["name"], "a": int(round(p["start_frame"] * 1000 / fs)), "b": int(round(p["end_frame"] * 1000 / fs))}
              for p in tr["form"]["phase_intervals_in_integer_frames"] if p["end_frame"] > p["start_frame"]]
    data = {"run": os.path.basename(out_dir), "total_ms": int(round(tr["form"]["total_frames"] * 1000 / fs)),
            "seed": tr["seed"], "status": tr["run_status"], "lanes": lanes, "bars": bars, "phases": phases}
    os.makedirs(os.path.join(page_dir, "audio"), exist_ok=True)
    if "--no-audio" not in argv:
        encode(os.path.join(out_dir, "result.wav"), os.path.join(page_dir, "audio", "mix.mp4"), mix_kbps)
        for i in range(len(names)):
            encode(inputs[i]["path"], os.path.join(page_dir, "audio", f"t{i:02d}.mp4"), kbps)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "material_map_template.html"), encoding="utf-8").read()
    title = argv[argv.index("--title") + 1] if "--title" in argv else f"素材マップ seed {tr['seed']}"
    html = tpl.replace("__TITLE__", title, 1).replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open(os.path.join(page_dir, "index.html"), "w", encoding="utf-8").write(html)
    per = {}
    for b in bars:
        per[b[0]] = per.get(b[0], 0) + 1
    print(f"{len(bars)} bars over {len(lanes)} lanes; per lane {[per.get(i, 0) for i in range(len(lanes))]}")
    print("page:", os.path.join(page_dir, "index.html"))


if __name__ == "__main__":
    main(sys.argv[1:])

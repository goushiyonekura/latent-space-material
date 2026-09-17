"""How much of each mode's ideal reaches the sound: per-hop variation of the frozen references
vs. the realized composition, fast-component correlation, fixed-reference residual and gain
activity.  Works for baseline, hires and fragment outputs (reads the trace's resolved config).

  python3 scripts/ideal_vs_realized.py output_frag [output_hires ...]
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from latent_space.config import deep_merge, DEFAULTS, apply_motion_profile, validate_config  # noqa: E402
from latent_space.engine import Job  # noqa: E402
from latent_space.curves import TrackCurve  # noqa: E402


def diag(outdir, mode):
    tp = os.path.join(outdir, mode, "state_trace.json")
    if not os.path.exists(tp):
        print(f"{outdir}/{mode}: no trace")
        return
    t = json.load(open(tp))
    cfg = t["resolved_config"]
    cfg["motion"] = apply_motion_profile({k: v for k, v in cfg["motion"].items() if k != "profile_scale_k"})
    validate_config(cfg)
    job = Job(cfg, "dev/out/tmp_diag", config_path=t.get("config_path") or "project.json")
    job.load_inputs()
    job.setup()
    an = job.analyzer
    curves = [TrackCurve.from_list(l) for l in t["curve_segments_per_track"]]
    hires = bool(t.get("hires_enabled"))
    res = []
    for uk, ref in t["fixed_reference_rows"].items():
        rows = np.array(ref["grid_rows"])
        xi_hat = np.array(ref["xi_hat"])
        gains = np.stack([cv.values(an.centers[rows]) for cv in curves], axis=1)
        if hires:
            starts = an.centers[rows] - an.W // 2
            pos = np.stack([curves[i].positions(starts, job.sources[i].shape[0]) for i in range(an.M)], axis=1)
            _f, S, _c = an.material_features_at(pos)
            G0, Gb = an.grams_at_positions(job.sources, starts, pos)
            xi, _ = an.composition_from_grams(gains, G0, Gb, S)
        else:
            xi, _ = an.composition(gains, rows)
        free = job.form.phase_names_at(an.centers[rows]) != "GOAL_HOLD"
        W = an.weight_vector()
        d_ref = an.dist2(xi_hat[1:], xi_hat[:-1])[free[1:]].mean()
        d_real = an.dist2(xi[1:], xi[:-1])[free[1:]].mean()
        k = 20
        hp = lambda x: x - np.stack([np.convolve(x[:, j], np.ones(k) / k, mode="same") for j in range(x.shape[1])], axis=1)  # noqa: E731
        hr, hx = hp(xi_hat), hp(xi)
        corr = (hr * hx * W).sum() / np.sqrt((hr * hr * W).sum() * (hx * hx * W).sum())
        gvar = np.abs(np.diff(gains[:, 1:], axis=0)).mean() * 100
        res.append((d_ref, d_real, corr, an.dist2(xi, xi_hat)[free].mean(), gvar))
    r = np.array(res).mean(axis=0)
    tag = "frag " if cfg.get("hires", {}).get("fragment_vocabulary") else ("hires" if hires else "base ")
    print(f"{tag} {mode:11s} bands={an.nb:2d} ideal var/hop {r[0]:.3f} realized var/hop {r[1]:.3f} "
          f"fast(2s)-corr {r[2]:+.2f} fit residual {r[3]:.3f} gain change/hop {r[4]:.2f} pts")


for outdir in (sys.argv[1:] or ["output_frag"]):
    for mode in ("diffusion", "vae", "transformer", "gan"):
        try:
            diag(outdir, mode)
        except Exception as e:  # noqa: BLE001
            print(f"{outdir}/{mode}: error {e}")

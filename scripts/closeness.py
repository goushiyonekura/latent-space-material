"""How close is the realized sound to each mode's ideal — on an interpretable scale.

The trace residual (mean d_xi^2 between the realized composition and the frozen ideal) has no
scale by itself, and part of any closeness is automatic because a window reference is anchored
to the realized current composition.  This read-only diagnostic recomputes the realized
composition from the trace (curves + position maps) and reports, per unit, on the rows the
realizer actually searched (committed rows; goal rise / goal hold excluded):

  now        mean d2(realized, frozen ideal)            == the trace's full-grid residual
  random     mean d2(random fragment mix, ideal)        random positions + random levels
  static     mean d2(uncontrolled mix, ideal)           continuous clock, one constant level
  block05/2s mean d2 between 0.5 s / 2 s block means    residual at the modes' own time scale
  flutter    mean d2(realized row, its 0.5 s block mean)   the material's own fast motion

and how well the sound FOLLOWS the requested moves, per reference hold (the span during which
one frozen ideal was chased; legacy traces re-anchor at every commit, so their hold is one
commit block):

  requested  d2(anchor, ideal at the end of the hold)   the move the mode asked for
  remaining  d2(realized, ideal) at the end of the hold
  achieved   1 - remaining / requested                  (<= 0: no closer than not moving at all)
  cos        direction cosine of (ideal_end - anchor) and (realized_end - anchor), d_xi metric
  cos_chance the same cosine against the ideal of an unrelated hold (shared-anchor artefacts)
  change_corr_within_hold   correlation of 0.5 s block-to-block changes, pairs inside one hold
  intervention   d2(ideal end, do-nothing end of the same hold)   what the plan ADDS to the material's own drift
  achieved_net   1 - remaining / intervention            the honest figure: `achieved` also counts the drift that
  cos_net        direction cosine relative to the do-nothing end     ideal and sound share automatically

Counterfactual baseline ("the realizer ignores the mode"): write the config with
  python3 scripts/closeness.py --ignore-config project.hold.diffusion.json dev/ignore_diffusion.json
run it with the normal CLI into <dir>/<mode>/ and pass <dir> to this script as well.

  python3 scripts/closeness.py output_hold [output_frag ...] [--modes=diffusion,vae] [--json=out.json]
"""
import json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from latent_space.config import apply_motion_profile, validate_config  # noqa: E402
from latent_space.engine import Job  # noqa: E402
from latent_space.curves import TrackCurve  # noqa: E402

MODES = ("diffusion", "vae", "transformer", "gan")


def write_ignore_config(src: str, dst: str) -> None:
    """Counterfactual: identical job, but the realizer's objective does not see the mode."""
    cfg = json.load(open(src))
    cfg.setdefault("objective", {})["w_mode"] = 0.0
    cfg.setdefault("hires", {})["w_relation"] = 0.0
    for k in ("materials",):
        cfg[k] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(src)), p)) for p in cfg[k]]
    cfg["goal"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(src)), cfg["goal"]))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    json.dump(cfg, open(dst, "w"), indent=1)
    print("wrote", dst)


def grams_chunked(an, sources, starts, pos, chunk=256):
    G0s, Gbs = [], []
    for k in range(0, len(starts), chunk):
        G0, Gb = an.grams_at_positions(sources, starts[k:k + chunk], pos[k:k + chunk])
        G0s.append(G0)
        Gbs.append(Gb)
    return np.concatenate(G0s), np.concatenate(Gbs)


def wcos(a, b, W):
    num = (a * b * W).sum(axis=-1)
    den = np.sqrt((a * a * W).sum(axis=-1) * (b * b * W).sum(axis=-1)) + 1e-30
    return num / den


def diag(outdir, mode, n_random=400, seed=20260916):
    tp = os.path.join(outdir, mode, "state_trace.json")
    if not os.path.exists(tp):
        print(f"{outdir}/{mode}: no trace")
        return []
    t = json.load(open(tp))
    cfg = t["resolved_config"]
    cfg["motion"] = apply_motion_profile({k: v for k, v in cfg["motion"].items() if k != "profile_scale_k"})
    validate_config(cfg)
    job = Job(cfg, "dev/out/tmp_diag", config_path=t.get("config_path") or "project.json")
    job.load_inputs()
    job.setup()
    an, fs, M = job.analyzer, job.fs, job.analyzer.M
    if not bool(t.get("hires_enabled")):
        print(f"{outdir}/{mode}: not a hires trace (use scripts/ideal_vs_realized.py)")
        return []
    Wv = an.weight_vector()
    curves = [TrackCurve.from_list(l) for l in t["curve_segments_per_track"]]
    hz = cfg["hires"]
    commit = int(round(float(hz["commit_seconds"]) * fs))
    ramp = max(1, min(commit, int(round(float(hz["ramp_seconds"]) * fs))))
    rise = int(round(float(hz["goal_rise_seconds"]) * fs))
    rng = np.random.default_rng(seed)
    Ls = [s.shape[0] for s in job.sources]
    out = []
    for u in t["form"]["units"]:
        ui = int(u["index"])
        ref = t["fixed_reference_rows"][str(ui)]
        rows = np.array(ref["grid_rows"])
        xi_hat = np.array(ref["xi_hat"])
        centers = an.centers[rows]
        starts = centers - an.W // 2
        ga = u["goal_arrival_frame"]
        search_end = max(u["start_frame"], ga - rise) if ga is not None else u["end_frame"]
        used = (centers >= u["start_frame"]) & (centers < search_end)
        if used.sum() < 10:
            continue
        # ---- realized composition (actual positions and gains)
        gains = np.stack([cv.values(centers) for cv in curves], axis=1)
        pos = np.stack([curves[i].positions(starts, Ls[i]) for i in range(M)], axis=1)
        _f, S, _c = an.material_features_at(pos)
        G0, Gb = grams_chunked(an, job.sources, starts, pos)
        xi, _ = an.composition_from_grams(gains, G0, Gb, S)
        now = float(an.dist2(xi, xi_hat)[used].mean())
        # ---- static baseline: continuous clock, all materials at one constant level
        lvl = float(gains[used][:, 1:].mean())
        g_st = gains.copy()
        g_st[:, 1:] = lvl
        pos_st = np.stack([starts % Ls[i] for i in range(M)], axis=1)
        _f, S_st, _c = an.material_features_at(pos_st)
        G0s, Gbs = grams_chunked(an, job.sources, starts, pos_st)
        xi_st, _ = an.composition_from_grams(g_st, G0s, Gbs, S_st)
        static = float(an.dist2(xi_st, xi_hat)[used].mean())
        # ---- random fragment baseline (blocks of 5 rows, one leading row for the flux)
        uidx = np.where(used)[0]
        okr = uidx[uidx >= 1]
        okr = okr[np.isin(okr + 4, uidx)]
        d_rand = []
        for _ in range(n_random):
            r0 = int(rng.choice(okr))
            rr = np.arange(r0 - 1, r0 + 5)
            st = starts[rr]
            p = np.zeros((6, M), dtype=np.int64)
            p[:, 0] = pos[rr, 0]
            for i in range(1, M):
                p[:, i] = (int(rng.integers(0, Ls[i])) + (st - st[0])) % Ls[i]
            g = np.zeros((6, M))
            g[:, 0] = gains[rr, 0]
            g[:, 1:] = rng.uniform(0.0, 1.0, M - 1)[None, :]
            _f, S_r, _c = an.material_features_at(p)
            G0r, Gbr = an.grams_at_positions(job.sources, st, p)
            xi_r, _ = an.composition_from_grams(g, G0r, Gbr, S_r)
            d_rand.append(an.dist2(xi_r[1:], xi_hat[rr[1:]]).mean())
        random_ = float(np.mean(d_rand))
        # ---- commit blocks (0.5 s)
        bidx = ((centers - u["start_frame"]) // commit).astype(int)
        bids = [b for b in np.unique(bidx[used]) if (used & (bidx == b)).sum() >= 3]
        brow = {int(b): np.where(used & (bidx == b))[0] for b in bids}
        xb = {b: xi[r].mean(axis=0) for b, r in brow.items()}
        hb = {b: xi_hat[r].mean(axis=0) for b, r in brow.items()}
        bl = sorted(brow)
        XB = np.array([xb[b] for b in bl])
        HB = np.array([hb[b] for b in bl])
        block05 = float(an.dist2(XB, HB).mean())
        # the material's own fast motion: Bessel-corrected variance of the rows AFTER the gain ramp
        after = ((centers - u["start_frame"]) % commit) >= ramp
        fl = []
        for b in bl:
            r_ = brow[b][after[brow[b]]]
            if len(r_) >= 2:
                fl.append(float(an.dist2(xi[r_], xi[r_].mean(axis=0)[None, :]).mean()) * len(r_) / (len(r_) - 1.0))
        flutter = float(np.mean(fl)) if fl else float("nan")
        n4 = (len(bl) // 4) * 4
        block2s = float(an.dist2(XB[:n4].reshape(-1, 4, XB.shape[1]).mean(axis=1),
                                 HB[:n4].reshape(-1, 4, HB.shape[1]).mean(axis=1)).mean()) if n4 >= 4 else float("nan")
        # ---- reference holds: recorded by the engine, or one commit block each for legacy traces
        rep = t["units"][ui]
        holds = []
        if rep.get("hold_segments"):
            for h in rep["hold_segments"]:
                a, b = h["committed_rows"]
                bs = sorted({int(x) for x in bidx[a:b + 1]} & set(bl))
                if bs:
                    holds.append({"blocks": bs, "anchor": np.array(h["anchor"]),
                                  "stay": (np.array(h["stay_end"]) if h.get("stay_end") is not None else None)})
        else:
            for k in range(1, len(bl)):
                if bl[k] == bl[k - 1] + 1:
                    holds.append({"blocks": [bl[k]], "anchor": xi[brow[bl[k - 1]][-1]]})
        hold_blocks = float(np.mean([len(h["blocks"]) for h in holds])) if holds else float("nan")
        A = np.array([h["anchor"] for h in holds])
        He = np.array([hb[h["blocks"][-1]] for h in holds])
        Xe = np.array([xb[h["blocks"][-1]] for h in holds])
        requested = an.dist2(He, A)
        remaining = an.dist2(Xe, He)
        moved = an.dist2(Xe, A)
        big = requested > 1e-9
        achieved = float(1.0 - remaining[big].sum() / requested[big].sum())
        cos = float(wcos(He - A, Xe - A, Wv)[big].mean())
        lag = max(1, int(round(10.0 * fs / commit / max(1.0, hold_blocks))))     # >= 10 s away
        cos_chance = float(wcos(np.roll(He, -lag, axis=0) - A, Xe - A, Wv)[big].mean()) if len(holds) > 2 * lag else float("nan")
        # net of the material's own drift: the plan's INTERVENTION = ideal end vs the do-nothing end of the same
        # hold (recorded by the engine); achieved_net = share of that intervention found in the sound
        achieved_net, cos_net, intervention = float("nan"), float("nan"), float("nan")
        hs = [k for k, h in enumerate(holds) if h.get("stay") is not None]
        if len(hs) > 3:
            Se = np.array([holds[k]["stay"] for k in hs])
            itv = an.dist2(He[hs], Se)
            okk = itv > 1e-6
            if okk.any():
                intervention = float(itv.mean())
                achieved_net = float(1.0 - remaining[hs][okk].sum() / itv[okk].sum())
                cos_net = float(wcos(He[hs] - Se, Xe[hs] - Se, Wv)[okk].mean())
        # ---- change correlation of consecutive 0.5 s blocks inside one hold / over all pairs
        da, dh = [], []
        for h in holds:
            for k in range(1, len(h["blocks"])):
                b0, b1 = h["blocks"][k - 1], h["blocks"][k]
                if b1 == b0 + 1:
                    da.append(xb[b1] - xb[b0])
                    dh.append(hb[b1] - hb[b0])
        within = float(wcos(np.array(dh).ravel(), np.array(da).ravel(), np.tile(Wv, len(da)))) if len(da) > 5 else float("nan")
        allp = float(wcos((HB[1:] - HB[:-1]).ravel(), (XB[1:] - XB[:-1]).ravel(), np.tile(Wv, len(HB) - 1)))
        res = dict(dir=outdir, mode=mode, unit=ui, rows=int(used.sum()),
                   trace_residual=rep["realization_summary"]["full_grid_fixed_target_residual"], now=now,
                   random=random_, static=static, block05=block05, block2s=block2s, flutter=flutter,
                   hold_blocks=hold_blocks, holds=len(holds), requested=float(requested.mean()),
                   remaining=float(remaining.mean()), moved=float(moved.mean()), achieved=achieved,
                   cos=cos, cos_chance=cos_chance, intervention=intervention, achieved_net=achieved_net, cos_net=cos_net,
                   change_corr_within_hold=within, change_corr_all_pairs=allp,
                   mean_level=lvl)
        out.append(res)
    return out


def summarize(res):
    def wavg(rs, k):
        rs = [r for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        n = sum(r["rows"] for r in rs)
        return sum(r[k] * r["rows"] for r in rs) / n if n else float("nan")
    print(f"{'dir':24s} {'mode':11s} | random  now   gauge | block05 block2s flutter | hold  requested remaining achieved   cos (chance) | change corr in-hold / all | intervention achieved_net cos_net")
    for key in sorted({(r["dir"], r["mode"]) for r in res}, key=lambda x: (x[0], MODES.index(x[1]))):
        rs = [r for r in res if (r["dir"], r["mode"]) == key]
        rnd, nw = wavg(rs, "random"), wavg(rs, "now")
        print(f"{key[0][-24:]:24s} {key[1]:11s} | {rnd:.3f} {nw:.3f} {100 * math.sqrt(nw / rnd):5.1f} | "
              f"{wavg(rs, 'block05'):.3f}   {wavg(rs, 'block2s'):.3f}   {wavg(rs, 'flutter'):.3f}   | "
              f"{wavg(rs, 'hold_blocks') * 0.5:3.1f}s  {wavg(rs, 'requested'):.3f}     {wavg(rs, 'remaining'):.3f}    "
              f"{wavg(rs, 'achieved'):+.2f}   {wavg(rs, 'cos'):+.2f} ({wavg(rs, 'cos_chance'):+.2f}) | "
              f"{wavg(rs, 'change_corr_within_hold'):+.2f} / {wavg(rs, 'change_corr_all_pairs'):+.2f}            | "
              f"{wavg(rs, 'intervention'):.3f}        {wavg(rs, 'achieved_net'):+.2f}        {wavg(rs, 'cos_net'):+.2f}")
    print("gauge: linear distance to the ideal, random fragment mix = 100, perfect = 0 (row-weighted over units)")
    print("intervention / achieved_net / cos_net: the same, but relative to the do-nothing end of each hold (the material's own drift removed)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--ignore-config":
        write_ignore_config(argv[1], argv[2])
        sys.exit(0)
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    dirs = [a for a in argv if not a.startswith("--")] or ["output_frag"]
    modes = opts.get("modes", ",".join(MODES)).split(",")
    allres = []
    for d in dirs:
        for m in modes:
            try:
                r = diag(d, m)
            except Exception as e:  # noqa: BLE001
                print(f"{d}/{m}: error {e!r}")
                r = []
            for x in r:
                print(json.dumps(x), flush=True)
            allres += r
    if allres:
        summarize(allres)
    if "json" in opts:
        json.dump(allres, open(opts["json"], "w"), indent=1)

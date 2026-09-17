# Fragment-vocabulary revision — mode contract (2026-09-16, user-authorised)

Goal: make each mode's latent-space computation audible.  Diagnostics showed the bottleneck is
the *ideal side*: references move slowly (0.03–0.3 per 0.1 s in d_xi units) and are tethered to
continuous-clock probes and xi_G, while the realizer (hires) can now switch levels every 0.5 s
and jump playback positions every >= 2 s.  The fix: build every mode's ideal on the **fragment
vocabulary** — the set of positions of each source with the features of what sounds after a
jump — so that ideals live inside the enlarged reachable set, and re-tune the mode's time scales
to the commit step.

Read: `docs/MODE_CONTRACT.md` (hooks), `latent_space/hires.py` (`run_unit_hires`: how references
are used, what `observe_committed` receives), `latent_space/analysis.py` (fragment helpers),
`latent_space/history.py`.  Run the fixture with `dev/fixture_frag.json` (24 bands, d_xi = 41,
model_step 0.5 s, fragment mode) — the baseline configs must keep working too (8 bands, no hires).

## Fragment helpers (Analyzer, available when `hires.enabled` and `hires.fragment_vocabulary`)

```
an.solo_pos[i]   (P_i,)        positions (frames) of the fragment grid of source i (hop 0.1 s)
an.solo_f[i]     (P_i, d_phi)  normalized features of the 46 ms window at each position
an.frag_f[i]     (P_i, d_phi)  features averaged over the clip_feature_seconds that FOLLOW each
                               position (what actually sounds after a jump) — use this one
an.frag_E[i]     (P_i,)        mean energy of that clip (silence: < an.silence_energy)
an.fragment_composition(sources, positions (M,), levels (M,), offsets (n,) frames)
                 -> (xi (n, d_xi), parts): EXACT mixture rows for "track i plays from positions[i]
                    at level levels[i]", rows at offsets after the positions (use e.g.
                    offsets = arange(0, clip_frames, hop_frames)[:k])
an.random_fragment_composition(sources, rng, offsets, goal_level=0.0, levels=None)
                 -> (xi, parts, meta{"positions","levels"})
an.fragment_candidates(track, target_ratios (nb,), n, exclude_near=None, exclude_frames=0)
                 -> top-n positions by clip-averaged band profile distance
an.material_features_at(positions (n, M)) -> (f (n,M,d_phi), S (n,M,M), chi (n,M))
an.composition_from_grams(gains (n,M), G0, Gb, S) / an.grams_at_positions(sources, starts, positions)
```
`sources` = `job.sources` (list of (L_i, C) float32 arrays); the mode can keep a reference to them
from the engine: they are not on the analyzer.  Add `self.sources = None` and read
`job.sources` through the unit context is not possible — instead the engine sets
`mode.sources = job.sources` before `begin_unit` (see engine.setup; use `getattr(self, "sources", None)`).

Layout of xi: `[phi (d_phi = 2 + nb) | c (M) | upper(R)]`; band ratios are `xi[:, 1:1+nb]`
(normalized).  `an.nb` = 24 in the fragment configs, 8 in the baseline configs — never hard-code.

## What `run_unit_hires` does with your hooks (fragment mode)

- Before `prepare_reference(unit, history, rows, xi_current, n)`: `unit.f_mat/S/chi[rows]` are the
  features at the **played** positions.  `xi_current` = last committed composition.
- The reference is frozen; the realizer picks, per material, fragment candidates whose
  clip-averaged band profile is closest to the reference's mean band profile, runs a beam over
  tracks on exact mixtures, then a level search.  Therefore: a reference that is itself an exact
  fragment composition (built with `fragment_composition`) is reachable up to level/timing
  quantisation; a reference far outside the fragment space is not.
- `observe_committed(unit, history, rows_c, xi_c, parts_c, reference, stats)`: `stats["positions"]`
  (M,) played positions at the block end, `stats["jumps"]` {track: position}, plus the
  refinement dict.  `history.events` entries now carry `src_position` and `jump`.
- Time: commit 0.5 s, lookahead 4 s (8 steps of model_step 0.5 s; `cfg["analysis"]
  ["model_step_seconds"]`).  Your ideal should be able to change by a realizable amount per
  model step (levels can change by up to 1.0 per 0.25 s ramp; a jump changes the material).

## Per-mode requirements

**Diffusion** — the field is a *time process* in musical time: integrate
`dxi = -tau M grad E dt + sqrt(2 tau T) dW` per model step over the window rows (dt =
model_step), starting from `xi_current`, with the coefficients (w_ij, s_ij, d_ij, beta, M_D)
from the committed history.  Anchor term: replace the fixed 0.4/0.1 probe by the mean of a
small sample of random fragment compositions drawn once per unit (record them) and give it a
low weight relative to pair/triple terms (config `lambda_phi` stays, but the anchor now moves
with the fragments; document).  Tune `step`/`temperature` so the ideal moves per step by an
amount comparable to the realizer's capability (record the mean per-step displacement).
Provide `relation_terms` from the same coefficients.  Statistic in trace: field energy of the
committed composition vs. the mean field energy of random fragment compositions.

**VAE** — basis from the SVD of `W^{1/2}(xi - xi_G)` over N random fragment compositions
(N >= 1500 rows, e.g. 300 compositions x 5 offsets; seeded; record the variance share captured
by k), k = `vae_latent_dim` (fragment config: 4).  Decoder `xi_G + sum_k s_k tanh(z_k) v_k`
unchanged; latent OU with `latent_correlation_seconds` at the model step; scale Sigma_H's floor
so tanh uses its range (record the fraction of |tanh z| > 0.9).  Persist the basis in
`history.mode_state['vae']` (fixed for the job).  Statistic: latent path smoothness (mean |dz|
per step) and decoder residual.

**Transformer** — tokens are *fragments*: per source, the top-K (K = 6) fragment positions by
relevance to the current composition and reference-free heads (similarity/contrast to the current
material features), plus the goal token; the value of a fragment token is the EXACT composition
of foregrounding that fragment (fragment at `probe_foreground_gain`, others at their current
levels, positions = current) minus the current composition (use `fragment_composition` with a
few offsets and average).  Attention heads as before; the memory head attends committed events
(with `src_position`, `dxi`, `end_seconds`).  Autoregression at the model step over the window
rows from `xi_current`; the radial bound stays but should rarely bind now (record the rate).
Statistics: attention entropy per head and the fraction of steps where the memory head's top
reference is an event of the same source position (recurrence).

**GAN** — reference distribution = random fragment compositions (>= 12 per unit, seeded,
recorded) instead of Bank plans; parents = committed composition paths from `history.parents`
(phase blocks) + fresh references; basis from their differences; discriminator on the same rows;
generator sampling and REINFORCE as before; D/G updates in `observe_committed`.  Statistic:
D of the committed composition vs D of fresh random fragment compositions.

All: keep `propose()` working for the warm start of the baseline engine (it may build on the
same fragment machinery when `an.frag_f` exists, and fall back to the previous behaviour when it
does not).  Keep the fixture job (`dev/fixture_frag.json`) under ~90 s.  No edits to shared files
unless genuinely needed (report them).  Record what is inherited/re-initialised and internal
iterations vs musical time, as before.

Verify with:
```
python3 -m latent_space generate --config dev/fixture_frag.json --mode <mode> --output dev/out/frag_<mode>
python3 -m latent_space generate --config dev/fixture_vae.json --mode <mode> --output dev/out/base_<mode>   # baseline must still run
```
and report per unit: initial/final joint objective and fixed-target error, full-grid residual,
jumps, your mode statistics, and the ideal's mean per-step displacement (d_xi units).

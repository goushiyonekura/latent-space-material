# Law → plan fidelity — mode contract (2026-09-17, user-approved items A / B / C)

State after `docs/HOLD_CONTRACT.md`: every mode publishes its held ideal as an exact realizable
plan, and the realizer follows it (achieved 0.96–0.99 on the real materials).  That number is now
nearly guaranteed by construction ("the exam was set to what the student can answer").  The
distance to each mode's own ideal sits in the step **law → plan**:

| mode | measured law → plan fidelity (real materials) |
|---|---|
| VAE | the plan gets only 27 % closer to the decoder target D(z) than doing nothing |
| transformer | cosine between the chosen step and the head-combined attention value 0.11–0.20 |
| diffusion | plan below the stay-plan's field energy 70–72 % of the time, gain −0.07…−0.09 per step; committed field energy 0.33–0.35 (fragment version 0.27–0.31, random fragments 0.54–0.73) |
| GAN | D gap small (−0.06…−0.12) but the "real" side is random plans moved close to the sound |

Goal of this revision: raise the law → plan fidelity of each mode **inside the rules** (gains,
position jumps with a 2 s minimum clip, no processing of the material).  `achieved` must stay ≥ 0.9.

## What is new in the shared code

- `an.fragment_candidates_mix(track, target_phi, others_band, level, n, exclude_near=None,
  exclude_frames=0, energy_weight=1.0)` ranks EVERY fragment of a track by how close the predicted
  MIXTURE (what the other tracks play + this fragment at `level`; cross terms ignored) comes to a
  target (`target_phi` = first 1+nb entries of a xi row, normalized).  ~1 ms per call.  Helpers:
  `an.mixture_band_energy(positions, levels, skip=track)`, `an.fragment_band_energy(track, positions)`,
  `an.fragment_index_at(track, position)`.  Measured: re-finding one track of a known mixture gives
  exact d² 0.015 with it vs 0.056 with the solo-profile `fragment_candidates` vs 0.110 random.  Use it
  as a wide pre-ranking, then evaluate the short list exactly with `plan_rows`.
- **More tracks are coming.**  The user allows the same material to sound at two positions at once and
  will later supply 8–12 materials.  `hires.voices_per_material: 2` makes the engine add a second
  track per material (same PCM and fragment bank, tracks `1+N … 2N`, started half a loop away), so
  M = 1 + 2N (9 today) and later M up to ~13–25.  Your code must not assume 4 materials anywhere,
  must treat every track i ≥ 1 alike, and must keep its cost per hold roughly LINEAR in the number
  of tracks (cap the number of tracks that move per step and the candidates per track; never
  enumerate combinations).  Test configs (created by the main session while you work; skip a run if a
  file is still missing when you finish): `dev/fixture_hold_v2.json` (4 materials × 2 voices),
  `dev/fixture_hold_n10.json` (10 materials).  Budget: ≤ 90 s for `dev/fixture_hold.json`, ≤ 4 min for the others.
- Registered config keys take their value from `config.py`, not from your in-code `.get` default.  To
  try another value of an already registered `hold_*` key, put it into a JSON config copy under `dev/`
  (`mode_defaults.<mode>.<key>`) and report the default you want.  NEW keys: in-code default + report.

## Rules (unchanged)

Edit only your mode file; no commits; no `scripts/smoke.py`; outputs under `dev/out/` with your mode's
name in the directory; legacy path (no `unit.realizer_state`) bit-identical —
`python3 dev/compare_legacy.py <mode>` must end with ALL IDENTICAL; keep internal iterations apart from
musical time in the trace; report only what you ran.

## Per mode — what to build and the fidelity figure to record (per hold and per unit, in the mode trace)

**Transformer (A) — the heads select on several tracks at once.**  One token per step cannot express
what attention asks for (cosine 0.11–0.20).  Tracks are independent players, so a step may move
several of them: per model step take, for the tracks with the largest head-combined attention mass
(how many scales with openness; jumps only where the minimum clip allows), their top / sampled token,
and set the material levels so that the EXACT rows come as close as possible to the free target of
eq. (36)/(37) (levels initialised ∝ attention weight, then a short coordinate / least-squares
refinement through `plan_rows`); among a few sampled candidate steps keep the closest.  Memory-head
recurrence stays.  Record: `attention_alignment` (cosine, d_xi metric, between the exact step and the
head-combined value) and `law_fidelity = 1 − d²(step rows, free target) / d²(stay rows, free target)`.

**Diffusion (A) — annealed, coarse-to-fine plan generation.**  Per hold run several internal
denoising iterations at decreasing temperature: first which fragments (wide pools from
`fragment_candidates_mix` for the drift target `ξ − τ M_D ∇E_D`, exact evaluation of the short list),
then the levels by the finite-difference Langevin drift you already have.  The last temperature is
T(o) > 0 (the chain stays thermal at o = 1, greedy only as o → 0).  Record: energy of the chosen plan
vs the stay-plan and vs the best plan seen in the hold (`energy_fidelity = (E_stay − E_plan) /
(E_stay − E_best_seen)`), `lower_than_stay_rate`, committed vs random-fragment field energy (target:
committed back to ≤ 0.31 on the real materials), requested move (keep ≥ 2× flutter).

**VAE (B) — a latent space of the reachable set.**  The decoder target D(z) lives in free xi-space and
is 73 % unreachable.  Re-define the latent space over CONTROLS: u = [level of every material track |
r PCA coordinates of the fragment it plays (PCA of that source's clip-averaged fragment features,
r = 2–3)].  Sample ≥ 1500 random realizable states with their exact compositions (the basis sample you
already draw), find the k control directions that explain the most composition variance (PLS-SVD of
standardized U against W^{1/2}(Ξ − mean)), decoder `u(z) = ū + P (s ⊙ tanh z)` → levels clipped to
[0, 1], fragment = nearest fragment in the track's PCA space (a jump only where the minimum clip
allows and only when the wanted fragment differs clearly from the playing one) → exact rows via
`plan_rows`: every decoded point is realizable, so the projection loss disappears by construction.
Encoder: z = Pᵀ(u_current − ū) from `realizer_state`.  Latent OU path, μ_H / Σ_H learning from the
committed states, tanh usage statistics as now.  Record: `latent_tracking` (correlation / cosine
between the requested Δz of a hold and the Δz of the committed sound, encoded the same way),
`control_realization` (share of the decoded control change that was legal to apply), and the share of
composition variance of held-out random realizable states explained by k latent axes through the
exact decoder.  k: a new key (default 6, capped by the control dimension).

**GAN (C) — real data becomes real.**  Positives of the discriminator = how the actual recordings
move: exact composition rows of UNMANIPULATED playback on the rows of the hold (every track plays on
from a random position with a constant level; levels drawn like the committed ones so that level alone
gives nothing away; the goal track follows its schedule), ≥ 8 windows per hold, feature map with
explicit MOTION terms (block-to-block differences) so that D judges how the sound moves, not only
where it is.  Negatives = the committed sound.  The generator keeps choosing among realizable candidate
plans (the narrowing band on the candidates stays) and learns through −log D which plans move like
real music.  Record: D(committed) vs D(real windows) (the gap is now meaningful — report it honestly,
also if D separates easily), D(candidate plans before realization), reference spread, requested move.

## Verify and report

```
python3 -m latent_space generate --config dev/fixture_hold.json --mode <mode> --output dev/out/fid_<mode>/<mode>
python3 scripts/closeness.py dev/out/fid_<mode> --modes=<mode>
python3 dev/compare_legacy.py <mode>
python3 -m latent_space generate --config project.hold.<mode>.json --mode <mode> --output dev/out/fid_real_<mode>/<mode>   # once or twice, 3–8 min
python3 scripts/closeness.py dev/out/fid_real_<mode> --modes=<mode>
```
Report: the fidelity figures before → after on the fixture AND on the real materials, achieved / cos,
requested move vs flutter, plan_rows calls per hold, run times (also for the v2 / n10 fixtures when
present), legacy check, new or changed keys, what is unfinished or doubtful.

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

---

## Polyphony cap — `hires.max_active_materials` (2026-09-17, user decision)

The user will supply 8–12 materials and wants **at most K materials to sound at once** (K = 3 now, a
parameter to be changed later; 0 = no limit).  Decisions: the two voices of one material count as ONE
material; while the 0.25 s ramps cross-fade, outgoing and incoming materials may overlap (up to 2K
sounding for a moment); jumps of a sounding track remain allowed.

Engine (done): `project_levels(levels)` keeps the K materials with the largest level (max over their
voices) and sets every voice of the others to 0.  It is applied to the levels of EVERY plan step inside
`plan_rows` (the levels actually used come back in `info["levels_used"]`, one entry per step in frame
order), to the future steps of the plan-aware lookahead and to the level hints; the realizer's own level
search never exceeds the cap and tries swaps (one sounding material out, a silent one in).  A hard check
counts sounding materials on a 50 ms grid outside ramps.  `Analyzer.random_fragment_composition` draws
random LEGAL states (1..K sounding materials) when the cap is on — `an.cap_random_levels(levels, rng)`
does the same for your own samplers (no cap: returns the levels unchanged and draws nothing).
`unit.realizer_state` carries `max_active_materials`, `material_of_track` (track → id of its material =
index of its first voice; the goal is 0) and `project_levels`.

What every mode must do when `realizer_state["max_active_materials"]` > 0 (with 0 nothing may change):

1. **Publish legal plans**: the `levels` of every step in `Target.meta["plan"]` are the projected ones (use
   `project_levels`, or `info["levels_used"]` of your final `plan_rows` call) — the published rows and the
   published plan must describe the same thing.
2. **Let the law choose WHICH materials sound.**  The cap turns "who sounds" into the main decision.  A silent
   material can only enter by replacing a sounding one, so your candidate moves need explicit SWAPS: material
   a out (all its voices to 0), material b in on one voice at a chosen level — and the entering voice may
   jump to the fragment you want while it is still silent (no audible splice; respect `next_jump_frame`).
   Level moves / gradients on silent materials are meaningless: spend the evaluations on the ≤ K sounding
   ones and on a bounded number of swap candidates (pre-rank the entering fragment with
   `an.fragment_candidates_mix`, whose `others_band` must then be the band energy of the materials that stay).
3. **Sample legal states**: every random state your law is calibrated on (anchors, bases, reference
   windows, candidate plans) must respect the cap, otherwise the law asks for mixtures that cannot exist.
4. Cost per hold stays roughly linear in the number of tracks and must not grow with the number of SILENT
   materials more than a pre-ranking does.
5. Record per unit: how often the sounding set changed (swaps per minute), the share of holds with K / fewer
   than K sounding materials, and your usual fidelity figure next to the no-cap value.

Verify with `dev/fixture_hold_cap2.json` (4 materials, K = 2), `dev/fixture_hold_v2_cap3.json` (4 × 2 voices,
K = 3), `dev/fixture_hold_n10_cap3.json` (10 materials, K = 3; the case that matters), the usual
`dev/fixture_hold.json` (no cap: must behave as before) and `python3 dev/compare_legacy.py <mode>`; read
`hard_checks.polyphony_cap` in the trace (`observed_max_outside_ramps` ≤ K, the time shares per count).

## ゴール露出方針 `contract_law`（2026-09-23、opt-in）— 方式への契約

`form.goal_exposure.policy == "contract_law"` のとき、実現層は CONTRACT の最初の確定境界から最終立ち上がり（`goal_rise_seconds` 前）までの保持で
`unit.realizer_state` に次を加える：

- `goal_law: bool` — この保持でゴール音源（トラック 0）の音量を計画の座標として動かしてよい。
- `goal_law_monotonic: bool` — true なら音量は非減少（計画内でも、確定後も下げられない）。
- `goal_level: float` — 保持開始時のゴール音量（下限）。

契約：計画 `{"frame", "jumps", "levels"}` の `levels[0]` がゴールの終了音量で、実現層はその確定のヒント `levels[0]` をそのまま採用する（実現層はゴールを探索しない）。位置ジャンプはトラック 0 に対して不可（`plan_rows` が落とす）。
`plan_rows` は `levels[0]` を単調に丸めてから厳密行を返す（`levels_used` に反映）。`goal_law` が false の保持では `levels[0]` は無視され
従来の台本どおり。方式は不可逆な座標に対称雑音を載せないこと（歯車になる）。Diffusion は実装済み（ゴール座標はドリフトのみ、絶対移動度 Δg = −d_lvl·∂E/∂g、|Δg| ≤ d_lvl。場のエネルギーはアンカー・ペア・三者・リッジを `plan_rows` の `info["xi_materials"]`＝ゴールを消した同じ窓の配合で、ゴール項だけを全体配合で評価する。実現層は `window_error(..., xi_materials=)` で同じ分離を渡す。方式は `accepts_xi_materials = True` を宣言するとこの引数を受ける）。ゴール項の重み (1−o)^p の指数 p は `objective.contract_goal_weight_exponent`（既定 2）で、形式項と Diffusion のゴール項が同じ値を使う。
VAE／Transformer／GAN は未対応（`levels[0]` を動かさない＝実現層の形式項だけが効く）。

**収束の物差し（`form.goal_exposure.convergence`、2026-09-23）**：`share`（既定、d_ξ² 全体）、`sparsity`（スペクトル＋N_eff＋占有率）、`presence`
（音源ごと・在否ベース）。実現層は候補ごとに `parts["occ"]`（占有率）、`parts["gains"]`、`parts["solo_f"]`（単独特徴）を付け、`window_error(...,
occ_rows=)`（sparsity）または `window_error(..., conv_parts={"occ","gains","solo_f"})`（presence）で渡す。方式は `Analyzer.sparsity_terms` /
`Analyzer.presence_terms` を使って Diffusion と同じ式（`sparsity_goal_term` / `presence_goal_term`）を評価すること。`presence` では素材は一手で
0 にしないと抜けられないので、方式は drop 候補（鳴っている素材を 1 本消す）を計画に持つこと（Diffusion は保持の最初のフレームで熱的に選ぶ）。
指数：素材側 p＝`objective.contract_goal_weight_exponent`、ゴール進入 q＝`form.goal_exposure.law_entry_exponent`（None＝p）。

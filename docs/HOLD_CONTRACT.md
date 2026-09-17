# Held realizable ideals — mode contract (2026-09-16, user-requested revision)

User request: (a) stop re-anchoring the ideal every 0.5 s — hold it ~2 s and let the sound chase it;
(b) make one step of the ideal larger than the material's own flutter; (c) GAN: narrow the references
to fragments close to the committed history.

## Measured facts that shape this contract (`scripts/closeness.py`)

- Fragment version: the ideal was re-anchored to the realized composition at every 0.5 s commit, so it
  *followed* the sound.  A realizer that ignores the mode entirely still gets residual 0.31–0.73 vs
  0.20–0.49 actual (random fragment mix 0.77–1.12): most of the closeness was automatic, and the
  correlation of 0.5 s changes between ideal and sound was ≈ 0.
- Holding the ideal for 2 s with the unchanged modes (fixture): VAE achieved +0.40 and transformer
  +0.27 of the requested move, but diffusion −0.56 and GAN −0.02 (direction cosine at chance level).
- Feasibility probe: when the held ideal is the EXACT composition of a realizable plan (jumps +
  level ramps on the commit grid), the realizer achieves **+0.97** of the requested move with plan
  hints and **+0.84** without them (cosine 0.98 / 0.85 vs chance 0.36).

Conclusion: the realizer can follow; the ideals ask for moves that no gain / position choice can
produce (free directions of the 41-dim composition space).  **Each mode must now express its ideal
as a realizable plan chosen by its own law.**

## Engine semantics in hold mode (`hires.reference_hold_seconds` > `commit_seconds`)

- `prepare_reference(unit, history, rows, xi_current, n)` is called **once per hold** (every 2 s), not
  per commit.  `rows` spans hold − commit + lookahead (≈ 5.5 s, 55 rows; shorter near the search end).
  `xi_current` is the anchor = mean of the last committed block (`hires.reference_anchor:
  block_mean`).  With `hires.reference_selection: first` and `n_reference_proposals: 1` the first
  proposal is frozen as is (no "closest to the unchanged sound" selection).
- The same `Target` is then chased for every commit of the hold.  `window_error` is called with
  sub-windows of those rows.  `observe_committed` is still called once per commit, with the held
  reference and extra stats: `reference_hold_seconds`, `reference_age_steps` (0 at the first commit
  of a hold), `reference_t0_frame`, `reference_is_new`, `plan_hint_jumps` / `plan_hint_jumps_taken` /
  `plan_hint_levels` (what the engine offered from the plan at this commit and which hinted jumps
  the realizer took).  Do not assume one `prepare_reference` per
  commit anywhere (counters, per-window logs, "last window" state used in `observe_committed`).
- Before `prepare_reference` the engine publishes `unit.realizer_state` (dict; absent / `None`
  outside hold mode — then the mode must behave exactly as before, bit for bit):

```
frame             int   current commit frame t0 (start of the hold)
rows              (n,)  the rows passed to prepare_reference
levels            (M,)  current gains (index 0 = goal track, deterministic, not controllable)
positions         (M,)  source frame each track plays at the window start of rows[0]
next_jump_frame   [M]   first frame at which each track may jump again (minimum clip length)
jumps_enabled     bool
commit_frames, ramp_frames, hold_frames, min_clip_frames, lookahead_frames, search_end_frame
goal_gains        (n,)  the goal track's scheduled gain on rows
plan_rows(steps, rws=None) -> (xi (n', d_xi), parts, info)
```

- `plan_rows` is the EXACT composition the realizer itself would produce for a plan:
  `steps = [{"frame": f, "jumps": {track: source_position}, "levels": [M] or None}, ...]`, frames on
  the commit grid (`frame + k * commit_frames`, ascending).  At frame f the listed material tracks
  jump (dropped and reported in `info["dropped_jumps"]` when `f < next_jump_frame[track]` or the
  track jumped less than `min_clip_frames` earlier in the same plan) and all materials ramp from
  their level to `levels` over `ramp_frames` (Q5), then hold.  `levels[0]` is ignored: the goal
  track always follows its deterministic schedule, a plan cannot steer it.  `rws` defaults to
  `rows`; pass a subset (e.g. 8 rows) while searching — one call on 55 rows costs a few ms.
  `info["gains"]`, `info["positions"]` give what was evaluated.
- Put the plan into `Target.meta["plan"]` (same step dicts).  At the commit whose frame equals a
  step's frame the engine adds the step's jumps to its jump candidates, always examines the whole
  hinted jump combination, and starts its level search from the hinted levels when they score
  better.  A hinted jump that could not happen at its frame is re-offered (time-aligned) at the later
  commits of the hold.  The frozen ideal still decides: J = mode error + E_form + relation terms.
- Plan-aware lookahead (`hires.plan_aware_lookahead`, default true): every candidate is scored over the
  lookahead window as "this choice now, then the remaining steps of the plan" (future level ramps and
  legal future jumps), not as "this choice held for 4 s".  Without it a plan that changes levels at
  every commit is compared with something the realizer never intends to play: measured on the real
  materials, achieved 0.80–0.88 for diffusion / VAE / transformer vs 0.99 for the smooth GAN plans;
  with it all four reach 0.96–0.99.
- The unit report gets `hold_segments` (anchor, requested move per hold) and `hold_summary`
  (`realized_flutter_dist2`, `mean_requested_dist2_at_hold_end(_open)`, `requested_over_flutter(_open)`,
  `plan_hints` counts).

## Requirements (every mode)

R1. **Hold-compatible.**  Everything above works with `dev/fixture_hold.json`.
R2. **Realizable ideal.**  In hold mode the rows `xi_hat[rows]` of the published target are exact
    `plan_rows` of a plan with a step at the hold start and, where the mode's law moves, at the later
    commit frames of the hold (the plan simply holds after its last step, up to the end of `rows`).
    `meta["plan"]` carries it.  GOAL_HOLD rows stay pinned (`fix_hold_rows`).
R3. **The mode's own law chooses the plan** — the four modes must remain four different
    computations (see below).  Candidate plans are scored by the mode's law, never by "distance to
    doing nothing".  Record how many candidate plans were evaluated per hold (internal iterations,
    kept apart from musical time) in the mode trace.
R4. **Move size.**  The requested move (engine: `hold_summary.requested_over_flutter_open`) should
    land in **3–6** on the fixture, i.e. clearly above the material's own fast motion and well below
    the distance between unrelated random mixes.  Expose ONE mode-local key `hold_move_scale`
    (default 1.0, read with `.get`) that scales the size of the requested move roughly linearly, so
    the main session can calibrate on the real materials from the config.  Moves scale with the
    openness `o` as before (small near the goal, none in GOAL_HOLD).
R5. **Legacy untouched.**  When `unit.realizer_state` is missing/None the mode behaves bit-identically
    to the current code: `python3 dev/compare_legacy.py <mode>` must print ALL IDENTICAL.
R6. **Cost.**  `dev/fixture_hold.json` for your mode ≤ 90 s (it is ~25 s now).
R7. Edit only your mode file.  New mode-local config keys: read them with in-code defaults and
    REPORT them (name, default, meaning) — the main session registers them in `config.py`
    (unknown keys are rejected by the loader, so do not put them into a config file yourself;
    while experimenting, change the in-code default).

## Per mode

**Diffusion — a field diffusion over realizable states.**  Keep the field E_D (anchor / pair /
triple / ridge / goal terms, coefficients from the committed history) and the openness temperature.
Replace the free SDE in ξ-space by a Langevin/Metropolis chain over realizable moves: per model
step of the hold (commit grid) propose level perturbations (Gaussian, scale ∝ `hold_move_scale`)
and jumps of allowed tracks to fragment candidates (e.g. `an.fragment_candidates` for the band
profile of the drift target `ξ − τ M_D ∇E_D`, plus random positions), evaluate the exact rows with
`plan_rows`, and accept by the field energy at temperature T(o) (drift = preference for lower energy,
noise = thermal acceptance).  Statistics: acceptance rate, field energy along the chosen path vs the
anchor and vs random fragment compositions (as now), requested move.

**VAE — the latent path decoded onto the realizable set.**  Keep the basis, the latent OU path
(anchored at the encoding of `xi_current`) and the decoder D(z).  For each model step of the hold
find the plan step whose exact rows are closest to D(z_k) (levels: a short coordinate / least-squares
search; jumps: fragment candidates matched to D(z_k)'s band profile) and publish the exact rows of
that plan.  Record the projection residual d²(plan rows, D(z)) per hold — it says how much of the
manifold move is realizable — next to the existing decoder residual and tanh usage.

**Transformer — attention selects fragment moves.**  The fragment tokens already are realizable
moves with exact values.  Instead of averaging all token values into a small free displacement,
let the heads *select*: per model step draw (or take the top of) the head-combined attention over
the fragment tokens of the tracks that may move, and apply that token as a plan step (jump that
track to the fragment and foreground it; the other levels follow the contrast / similarity
structure as your law dictates); memory head: events' `src_position` are fragment tokens too, so
recurrence becomes an actual return to a committed fragment.  Statistics: attention entropies,
recurrence, selected-token share per head, requested move.

**GAN — references narrowed to realizable continuations near the committed history (user item c).**
The reference distribution becomes: exact plans that continue from the current state and stay
close to committed compositions (e.g. candidate plans drawn around the current positions/levels
and around the positions/levels of committed parents; keep those whose rows lie within a distance
band of the committed history, record the band and the acceptance).  The generator p_Θ(b, w)
chooses the parent plan b and a displacement w that is itself realizable (level-space displacement),
the discriminator scores exact rows (references vs committed), REINFORCE / importance correction as
now, D/G steps once per commit.  Statistics: D(committed) vs D(references) — the aim is that they
approach each other —, reference spread (mean pair d²) before/after narrowing, requested move.

## Verify and report

```
python3 -m latent_space generate --config dev/fixture_hold.json --mode <mode> --output dev/out/hold_<mode>
mkdir -p dev/out/cmp_hold && ln -sfn ../hold_<mode> dev/out/cmp_hold/<mode>
python3 scripts/closeness.py dev/out/cmp_hold --modes=<mode>
python3 dev/compare_legacy.py <mode>
```
Report: `achieved`, `cos (chance)`, in-hold change correlation, `requested_over_flutter_open`, residual,
`plan_hints` counts, run time, the legacy check result, your mode statistics, new config keys, and
anything you could not do.  Baseline to beat (hold only, unchanged modes, fixture):
diffusion −0.56 / vae +0.40 / transformer +0.27 / gan −0.02 achieved.

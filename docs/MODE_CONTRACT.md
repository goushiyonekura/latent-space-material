# Mode-controller contract (for parallel implementation of diffusion / transformer / gan)

Package: `latent_space/` (numpy only, Python 3.9, **no other dependencies**). The pipeline
already runs end to end with the reference mode `latent_space/modes/vae.py`. Read these files
before writing anything:

- `latent_space/modes/base.py`   — the `ModeController` interface you must implement
- `latent_space/modes/vae.py`    — complete reference implementation (copy its structure)
- `latent_space/types.py`        — `UnitContext`, `Target`, `Candidate`, `Realization`
- `latent_space/analysis.py`     — composition state ξ, `dist2`, `split`, `weight_vector`, probes
- `latent_space/history.py`      — `History` (h_c, M_H, corr(), events, change_direction, parents, mode_state)
- `latent_space/objective.py`    — `Objective` (e_form, e_hist, phase_blocks / blocks_to_unit, temperatures)
- `latent_space/engine.py`       — how the engine calls the mode (`run_unit`)
- `latent_space/bank.py`         — how candidates are generated; the `hints` format

Run/test with the synthetic fixtures (already generated):

```bash
cd /Users/goushiyonekura/Claude/latent-space-material
python3 -m latent_space generate --config dev/fixture_vae.json --mode <mode> --output dev/out/<mode>
python3 - <<'EOF'
import json; t=json.load(open('dev/out/<mode>/state_trace.json'))
print(t['run_status'], t['hard_checks']['all_passed'], t['warnings']); print(json.dumps(t['mode_trace'], indent=1)[:3000])
for u in t['units']: print(u['unit'], u['stop_reason'], u['tolerance_met'], u['chosen'], u['history_internal_effect'], u['history_realized_effect'])
EOF
```

## What the engine does per planning unit (`engine.Job.run_unit`)

1. `mode.begin_unit(unit, history)` — condition on current material features `F`, the history
   `H` (**you must read `history`**: at least `history.mode_state[<your key>]`, and where the spec
   says so `history.h_c`, `history.M_H`, `history.corr()`, `history.events`,
   `history.change_direction`, `history.parents`), and openness `unit.o`.
2. `hints = mode.hints(unit, history)` — optional list of `(track_i, track_j, 'sync'|'counter')`
   telling the bank to build joint plans (same reversal times, same/opposite directions).
3. The bank builds ~8–16 legal joint gain trajectories (`Candidate`s), each with realized
   composition `cand.xi` (J, d_xi) computed from the **actual summed PCM** of the analysis windows.
4. For `round_index` in range(max_search_rounds):
   - `targets = mode.propose(unit, history, round_index, n_targets)` → list of `Target`
     (`xi_hat` shape (J, d_xi); **GOAL_HOLD rows must equal `unit.xi_goal`** — use
     `fix_hold_rows(unit, xi_hat)` from `base.py`).
   - For each target, every candidate is scored: `total = w_mode * mode.mode_error(unit, cand, target)/scale
     + w_form*e_form + w_hist*e_hist + w_motion*e_motion`; the best candidate is mutated a few
     times (budgeted) and re-scored.  The best `Realization` per target is collected.
   - `stats = mode.update(unit, history, realizations, round_index)` — **the mode's required
     internal update from realized compositions** (returns a small JSON-able dict).
5. One trajectory is selected by softmax over totals; then `mode.end_unit(unit, history, chosen,
   alternatives)` — write your mode summary into `history.mode_state[<key>]` (JSON-able: lists,
   floats, dicts — no numpy arrays in what you put in mode_state, convert with `.tolist()`), then
   the engine updates the common history (h_c, M_H, events, comparison archive).
6. `mode.signature()` → 1-D numpy vector summarising the ideal distribution / internal parameters
   (used to verify that a different history changes the internal state: the engine deep-copies
   the mode, calls `begin_unit` with an **empty** history and compares signatures; so the
   signature must depend on what you read from `history`).
7. `mode.trace()` → JSON-able dict (the engine serialises with a numpy-aware encoder; plain
   numpy arrays are fine here but keep it small: no per-candidate full matrices).

## Coordinates

`xi = [phi_norm (d_phi = 2 + 8 bands) | c (M) | upper(R) (M(M-1)/2)]`, `analyzer.split(xi)`
returns the three blocks; `analyzer.dist2(a, b)` is eq. (22) per row; `analyzer.weight_vector()`
is the diagonal W with `d^T W d == dist2`. `unit.probe(gain_vector)` → `(xi (J,d), parts)`
for a constant gain vector across the unit (probes are *analysis material*, not commands).
`unit.composition(gains (J,M))` for time-varying gains. `unit.f_mat` (J, M, d_phi) are the
normalized current material features, `unit.S` (J, M, M) their similarity
`exp(-||f_i-f_j||^2/(2 sigma_F^2))`, `unit.chi` (J, M) the tanh-bounded log-energy change per
material, `unit.o` openness, `unit.hold_mask`, `unit.free_mask`, `unit.xi_goal` (J,d),
`unit.phase_names` (J,) in {INTRO, OPEN, CONTRACT, GOAL_HOLD, REOPEN}, `unit.centers` (frames),
`unit.seconds`, `unit.J`, `unit.M`, `unit.N`, `unit.index`, `unit.start_gains` (M,).
Goal track is index 0. `objective.phase_blocks(unit, xi)` / `objective.blocks_to_unit(unit, blocks)`
map a composition trajectory to/from a phase-relative fixed grid (use this to carry past
compositions — parents, memory — into a new unit of possibly different length).
`parts['c']` (J,M) contributions, `parts['R']` (J,M,M), `parts['phi']`.
Config for your mode: `cfg['mode_defaults'][<key>]` plus the scalar keys next to it (see
`latent_space/config.py` DEFAULTS; add keys there only inside your own sub-dict).
Common history update rate: `cfg['history']['update_rate']`. RNG: `self.rng` (numpy Generator).

## Rules (from the spec, non-negotiable)

- The mode never produces or edits gains; it produces ideal *acoustic compositions* and scores
  *realized* compositions (`cand.xi`).  Never use `cand.gains` as the object of the mode error
  (you may look at gains for auxiliary hints only).
- No dummies: the mode's defining computation must actually run (see your spec section).
- Bounded iterations only (use the config maxima).  No file I/O, no threads, no new deps.
- Do not edit shared files (`engine.py`, `bank.py`, `analysis.py`, `types.py`, `objective.py`,
  `history.py`, `base.py`, `vae.py`); if you believe a shared change is needed, implement a
  local workaround inside your module and describe the request in your final report.
- Keep runtime small: the fixture job must finish in well under 60 s; real jobs have J≈900 per
  unit, d_xi≈25, M=5.
- Record in `trace()` the fields the spec lists for your mode plus warnings and per-unit
  summaries (`self.unit_traces`, `self.warnings`).

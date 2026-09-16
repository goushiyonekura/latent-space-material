# Mode-controller contract (audit-2 revision, 2026-09-16)

This replaces the earlier contract.  The old rules "do not edit shared files" and "the mode
updates its targets toward realized candidates during a search" are withdrawn.  Shared files may
be edited when a shared change is genuinely needed (coordinate through the engine owner); the
reference a trajectory is scored against must never move while it is being scored.

Package: `latent_space/` (numpy only, Python 3.9).  Read first: `modes/base.py` (interface),
`engine.py` (`run_unit`: warm start, commit steps, `_window_eval`, `_refine_window`),
`types.py`, `analysis.py`, `history.py` (`observe_committed`, `rho_dt`), `objective.py`
(`e_form_rows`, `j_relation`), `curves.py` (`TrackCurve.bumps`, `state_at`, `check_composite`).

## What the engine does per planning unit

1. `mode.begin_unit(unit, history)` — condition on materials F, history H (read
   `history.mode_state[<key>]`, `h_c`, `M_H`, `corr()`, `events`, `change_direction`,
   `parents`, `recent_xi`), openness `unit.o`.
2. `mode.hints(unit, history)` — optional `(i, j, 'sync'|'counter')` hints for the bank
   (relations are realized locally on interior intervals; stats in `bank.stats()['relations']`).
3. Warm start: bank candidates; `mode.propose(unit, history, 0, n)` full-unit ideal trajectories;
   the proposal whose best candidate has the lowest joint J is FROZEN as the unit reference R0
   (hash recorded); the best candidate against R0 (plus bounded mutations against the same R0) is
   the warm-start tail.
4. Commit steps (`realization.commit_seconds`, lookahead `realization.lookahead_seconds`):
   - `refs = mode.prepare_reference(unit, history, rows, xi_current, n)` — ideal trajectories for
     the window rows, built from the committed history and the realized current composition
     `xi_current` (the last committed row).  One of them is chosen by the current tail's joint J
     and FROZEN (`reference_hash`).  It does not move afterwards.
   - The tail is improved on the window by additive bump corrections
     `dg = a * 64 s^3 (1-s)^3` (zero value / velocity / acceleration at both ends) with a
     finite-difference coordinate search on the joint objective
     `J = w_mode * window_error/scale + w_form * E_form_rows + w_relation * J_rel + w_smooth * motion`
     computed from the actual mixed composition; only improvements are accepted; hard checks run
     on the composite curve (`check_composite`: limits, dB rate, actual motion episodes).
   - The first commit block is committed: `history.observe_committed(...)` with
     `rho = 1 - exp(-dt / tau_H)`, completed base motion events appended (time-ordered), then
     `mode.observe_committed(unit, history, rows_c, xi_c, parts_c, reference, stats)`.
5. `mode.end_unit(unit, history, chosen, alternatives)` — persist the mode summary into
   `history.mode_state[<key>]` (JSON-able).  `chosen.candidate` is the final composite trajectory.

`mode.update(...)` is NOT called any more.  Adaptation of the ideal happens only in
`prepare_reference` (before freezing) and `observe_committed` (after committing).

## Hooks a mode implements

```
begin_unit(unit, history)
propose(unit, history, round_index, n_targets) -> [Target]           # full-unit (warm start)
prepare_reference(unit, history, rows, xi_current, n_proposals) -> [Target]
     Target.xi_hat has shape (J, d_xi) on the unit grid; only `rows` must be meaningful;
     GOAL_HOLD rows == unit.xi_goal (use fix_hold_rows).  Default: propose(...).
window_error(unit, xi_rows, rows, target) -> float                  # default mean d_xi^2 on free rows
relation_terms(unit) -> {"omega": (M,M), "s": (M,M), "dstar": (M,M), "lag_seconds": float} | None
observe_committed(unit, history, rows, xi_rows, parts_rows, reference, stats) -> dict
end_unit(unit, history, chosen, alternatives)
signature() -> 1-D array (fixed length within a job)                # history-intervention probe
trace() -> JSON-able dict
```

Time scales (audit B6/C6): `analyzer.hop / fs` is the observation hop; `cfg['analysis']
['model_step_seconds']` is the model step; AR coefficients are `exp(-step / tau)` (helper
`ar1_rho_for`); the history rate is `history.rho_dt(dt)`.

## Coordinates and helpers

`xi = [phi_norm (d_phi = 2 + 8) | c (M) | upper(R) (M(M-1)/2)]`; `analyzer.split`, `dist2`,
`weight_vector`; `unit.probe(g)`, `unit.composition(gains)`; `unit.f_mat`, `unit.S`, `unit.chi`,
`unit.o`, `unit.hold_mask`, `unit.free_mask`, `unit.xi_goal`, `unit.phase_names`, `unit.centers`,
`unit.seconds`, `unit.J`, `unit.M`, `unit.index`, `unit.start_gains`; `objective.phase_blocks` /
`blocks_to_unit`; `history.recent_xi` (committed rows, short list), `history.events` (committed,
time-ordered), `history.parents`, `history.mode_state`.

## Rules

- The mode never produces or edits gains; it defines targets / relations / possibilities and
  scores realized compositions.  Never use `cand.gains` as the object of an error.
- No dummies, bounded iterations, no file I/O, no threads, no new dependencies.
- Record what was inherited and what was re-initialised whenever an internal coordinate system
  (basis) changes.  Record internal iterations separately from musical time.
- Keep the fixture job (`dev/fixture_vae.json`) under ~90 s.

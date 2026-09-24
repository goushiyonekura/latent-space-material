"""Configuration: JSON project file + single frozen defaults dictionary (spec §13)."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict

SPEC_VERSION = "1.1"
MODE_IDS = {
    "diffusion": "diffusion_field",
    "vae": "vae_latent",
    "transformer": "transformer_relational",
    "gan": "gan_coadaptive",
}

DEFAULTS: Dict[str, Any] = {
    "spec_version": SPEC_VERSION,
    "spec_document": None,            # optional path to the normative spec; its sha256 goes to the trace
    "config_aliases_applied": [],     # filled by load_config (audit §7)
    "mode": "diffusion",
    "seed": 48291,
    "materials": [],
    "goal": "",
    "base_level": {"policy": "shared_static_peak_bound", "sample_peak_cap": 0.85, "epsilon": 1e-9},
    "motion": {
        "velocity_max": 0.12,
        "velocity_min_bulk": 0.005,
        "mean_velocity_min": 0.003,
        "acceleration_max": 0.08,
        "jerk_max": 0.12,
        "regularized_db_rate_max": 12.0,
        "gain_log_epsilon": 0.01,
        "minimum_move_amplitude": 0.02,
        "slow_fraction_max": 0.40,
        "ramp_component_max_seconds": 3.0,
        "nonzero_scored_holds": False,
        # audit E2: an explicit alternative profile scales the same curve family in time by k:
        # V*k, A*k^2, J*k^3, L*k, minimum velocities *k, ramp component /k (UNVERIFIED perceptually)
        "profile": "baseline",            # 'baseline' | 'responsive_x2_UNVERIFIED'
    },
    "form": {
        "cycles": 2,
        "timing_policy": "adaptive",
        "intro_seconds": 12,
        "open_seconds": 20,
        "contract_seconds": 30,
        "goal_hold_seconds": 5,
        "reopen_seconds": 30,
        "adaptive_stage_seconds_min": 5,
        "adaptive_stage_seconds_max": 90,
        "final_reopen": True,
        # goal exposure policy (audit §3.4): 'contract_only' keeps the goal track at/below the caps in
        # INTRO/OPEN, rises continuously inside CONTRACT to 1, inherits 1 at REOPEN and descends legally;
        # 'free' lets the goal track move like any material before the goal time.
        # 'contract_law' (2026-09-23, opt-in): as 'contract_only' before CONTRACT and over the final
        # goal_rise_seconds, but inside CONTRACT the goal track is a searched track of the law
        # (level moves only, no jumps, non-decreasing when law_monotonic); the final rise starts
        # from the level the law reached, GOAL_HOLD stays exact.
        "goal_exposure": {"policy": "contract_only", "intro_max": 0.0, "open_max": 0.0,
                          "reopen_descend_within_reopen": True, "law_monotonic": True,
                          "law_mobility": 1.0,      # contract_law: goal step = law_mobility * d_lvl * (-dE/dg), |step| <= d_lvl
                          "law_step_floor": 0.0,    # contract_law: the goal's step budget uses max(o, floor) (0 = the law's o-scaling)
                          # convergence measure of the CONTRACT pull (2026-09-23 user decision):
                          # 'share' = d_xi^2 to the goal composition (spectrum + contribution shares + relations; lowers levels)
                          # 'sparsity' = spectrum + N_eff -> 1 (fewer simultaneous sources) + occupancy -> the goal's (sparser passages);
                          #   the contribution-share block is dropped
                          "convergence": "share", "sparsity_w_neff": 0.25, "sparsity_w_occ": 1.0,
                          # 'presence' = per-source, presence-based (2026-09-23 user decision): mean over the SOUNDING
                          #   materials of (solo spectral distance to the goal + w_occ (occupancy - goal's)^2)
                          #   + presence_w_count (n_sounding - 1)^2, weighted (1-o)^p, plus presence_w_goal (1-g)^2
                          #   weighted (1-o)^q with q = law_entry_exponent (None = p).  Levels do not enter, so the
                          #   pull acts by moving fragments, dropping sources and bringing the goal in.
                          "presence_w_count": 0.25, "presence_w_goal": 1.0, "law_entry_exponent": None,
                          # presence: the count target is (1 - goal level) (the last source leaves as the goal takes over)
                          # and an EMPTY set counts as distance presence_empty_distance x (1 - goal level): silence is as
                          # far from the goal as a poor material (90th percentile of the per-source distances = 1.7)
                          "presence_empty_distance": 1.7},
    },
    "analysis": {
        "window_frames": 2048,
        "hop_seconds": 0.1,
        "spectral_bands": 8,
        "band_edges_hz": [0, 100, 200, 400, 800, 1600, 3200, 6400],  # upper edge = Nyquist
        "silence_energy": 1e-10,
        "feature_std_floor": 0.02,
        "feature_epsilon": 1e-12,
        "model_step_seconds": 0.1,        # audit B6/C6: model-internal time step, separate from the hop
        "occupancy_threshold": 0.25,      # sparsity convergence: a hop 'sounds' if its energy > this x the source's median
    },
    "search": {
        "candidate_bank_target": 16,
        "max_total_candidates_per_cycle": 64,
        "max_search_rounds": 4,
        "normalized_mode_tolerance": 0.10,
        "minimum_improvement": 0.0001,
        "patience_rounds": 2,
        "targets_per_round": 4,
        "max_moves_per_track": 24,        # safety cap only; the count follows from time / legal durations
        "bank_generation_attempts": 400,
        "amplitude_mixture": {"probabilities": [0.45, 0.35, 0.20],      # small / medium / large moves
                              "ranges": [[0.05, 0.2], [0.2, 0.5], [0.5, 1.0]]},
        "duration_preference_beta": [1.0, 4.0],   # T = T_lo + Beta(a,b) (T_hi - T_lo): short-biased, uncalibrated
        "selection": "argmin",                    # 'argmin' | 'softmax_within_margin'
        "acceptance_margin": 0.0,                 # J <= J_min + margin (final targets, same scale)
        "tolerance_applies_to": "mode_error",     # 'mode_error' (fit + mode penalties) | 'target_fit'
        "final_targets": "best_round",            # adopted fixed target set: 'best_round' | 'last_round'
    },
    "objective": {
        "w_phi": 1.0, "w_c": 0.5, "w_R": 0.5,
        "w_mode": 1.0, "w_form": 0.5, "w_hist": 0.1, "w_motion": 0.01,
        "open_contribution_target": 0.25,
        "sigma_hist": 1.0,
        "comparison_points_per_phase": 24,
        "selection_temperature_base": 0.05,
        "selection_temperature_open": 0.45,
        "mode_error_scale": {"diffusion": 1.0, "vae": 1.0, "transformer": 1.0, "gan": 1.0},
        "w_neff": 0.25, "n_eff_target": 2.0,        # audit E3 (soft, OPEN-like rows only)
        "w_energy": 0.25, "energy_min_ratio": 0.05,
        "contract_goal_weight_exponent": 2.0,       # E_form CONTRACT weight (1-o)^p; 2.0 = the audited form
    },
    "mode_defaults": {
        "diffusion_particles": 4,
        "diffusion_internal_steps_max": 8,
        "diffusion": {"lambda_phi": 1.0, "lambda_2": 1.0, "lambda_3": 0.25, "lambda_G": 1.0,
                       "epsilon_D": 1e-4, "step": 0.01, "temperature_per_openness": 0.05,
                       "kappa_E": 0.25, "kappa_M": 0.5, "anchor_nongoal_gain": 0.4,
                       "anchor_goal_gain": 0.1, "time_correlation": 0.8,
                       "correlation_significance": 0.2, "similarity_sign_threshold": 0.5, "max_triples": 10,
                       "init_spread": 0.5, "realized_pull": 0.5, "search_matrix_scale": 1.0, "hint_pairs": 3,
                       # audit-2: window-local field exploration before freezing (realized_pull is ignored)
                       "window_continuation_seconds": 2.5, "window_internal_steps_max": 8,
                       "window_step_budget_per_unit": 4096, "relation_lag_seconds": 1.0,
                       # fragment-vocabulary mode: field SDE integrated in musical time (per model step)
                       "frag_step": 40.0, "frag_temperature_per_openness": 0.0016, "frag_drift_substeps": 10,
                       "frag_max_displacement_per_step": 1.5, "frag_anchor_samples": 24, "frag_anchor_offsets": 3,
                       "frag_anchor_seed": 104729, "frag_energy_probe_samples": 8,
                       # hold mode (docs/HOLD_CONTRACT.md): field chain over realizable plans
                       "hold_move_scale": 1.0,           # scales the level-perturbation sigma and the drift-target distance
                       "hold_level_candidates": 6, "hold_jump_candidates": 3, "hold_multi_jump_candidates": 3,
                       "hold_temperature_scale": 1.0, "hold_refine_sweeps": 1, "hold_eval_rows": 6,
                       "hold_law_version": 2,             # 1 = single-pass chain (hold v1) | 2 = annealed coarse-to-fine
                       # law version 2 (docs/FIDELITY_CONTRACT.md): denoising sweeps at decreasing temperature, fragments
                       # from mixture-aware pools first, then levels by the finite-difference Langevin drift
                       "hold_anneal_iterations": 3, "hold_anneal_ratio": 2.0, "hold_final_temperature": 0.4,
                       "hold_tracks_per_step": 2, "hold_gradient_tracks": 4, "hold_mix_ranking": True,
                       # history term against repetition: visit density of the committed jumps per source (shared by the
                       # voices of one material) penalises the fragment ranking and enters the chain as a novelty ridge
                       "hold_repetition_weight": 0.06, "hold_repetition_width_seconds": 2.0,
                       "hold_repetition_half_life_seconds": 60.0,
                       "hold_swap_candidates": 2},        # polyphony cap: swap proposals (one material out, a silent one in) per hold
        "vae_latent_dim": 2,
        "vae": {"mu_H_init": 0.5, "sigma_H_init": 0.04, "K_F": 0.1, "ridge_epsilon": 1e-6,
                "cov_floor": 1e-4, "probe_goal_gain": 0.1, "probe_in_gain": 0.7,
                "probe_out_gain": 0.15, "time_correlation": 0.9,
                "latent_correlation_seconds": 0.949,    # audit B6: correlation time in seconds (= 0.9 per 0.1 s)
                # fragment-vocabulary basis (random fragment compositions, seeded) and tanh range control
                "sigma_H_floor_frag": 0.09, "basis_scale_quantile": 99.0, "basis_scale_target_tanh": 0.95,
                "basis_fragment_compositions": 300, "basis_fragment_offsets": 5, "basis_min_rows": 1500,
                "basis_seed_offset": 9176,
                # hold mode (docs/HOLD_CONTRACT.md): latent path decoded onto the realizable set
                "hold_move_scale": 1.0,             # multiplies the latent innovation of a held OU path
                "hold_target_lead_rows": 3,         # rows by which the level fit leads D(z) (compensates the Q5 ramp)
                "hold_fit_rows": 5, "hold_level_step": 0.25, "hold_level_min_step": 0.05, "hold_level_evals": 20,
                "hold_jump_candidates": 2, "hold_jump_margin": 0.03, "hold_step_margin": 0.0,
                "hold_plan_calls_max": 160,
                "hold_latent_space": "controls",    # 'controls' (latent space of the reachable set) | 'xi' (hold v1 projection)
                # control-space latent (docs/FIDELITY_CONTRACT.md): u = [levels | r PCA coords of the playing fragment],
                # k PLS axes against the exact compositions, decoder -> clipped levels + nearest fragment -> plan_rows
                "control_latent_dim": 6, "control_fragment_pca": 2, "control_basis_states": 1500,
                "control_holdout_states": 200, "control_basis_offsets": 3, "control_seed_offset": 5521,
                "control_jump_min_pca_dist": 0.6, "control_max_jumps_per_step": 3, "control_sigma_floor": 0.30,
                "control_cap_hysteresis": 0.08},    # polyphony cap: decoder-side bonus for the materials that already sound
        "transformer_heads": ["similarity", "contrast", "memory"],
        "transformer": {"sigma_F": 1.0, "tau_H_seconds": 180.0, "alpha_s": 0.25, "alpha_c": 0.25,
                         "alpha_m": 0.25, "alpha_G": 1.0, "cov_diag_floor": 1e-4,
                         "probe_foreground_gain": 0.7, "probe_background_gain": 0.15,
                         "bias_scale": 0.5, "noise_scale": 0.5, "ar_stride": 1, "time_correlation": 0.9,
                         "excursion_radius": 0.35, "feedback_rate": 0.5, "feedback_bound": 3.0,
                         "step_seconds_reference": 0.1, "tendency_gain": 0,
                         "tau_noise_seconds": 0.949,    # audit D3.3: AR correlation time in seconds
                         "memory_digest_slots": 0,       # 0 = automatic (one long-term digest slot per track)
                         # fragment-vocabulary mode: fragment tokens, exact foregrounding values
                         "fragment_tokens_per_source": 6, "fragment_value_offsets": 3,
                         "fragment_step_seconds_reference": 0, "noise_scale_fragment": 0.25,
                         "excursion_radius_fragment": 1.0, "fragment_bound_growth_steps": 0,
                         "recurrence_tolerance_seconds": 2.0,
                         # hold mode (docs/HOLD_CONTRACT.md): the heads select realizable fragment moves
                         "hold_move_scale": 1.0,       # scales every level move (s = o * hold_move_scale)
                         "hold_move_rate": 0.6,        # p(step moves) = clip(o * rate)
                         "hold_background_scale": 1.0, "hold_temperature": 0.5, "hold_candidates": 3,
                         "hold_selection": "sample",   # 'sample' | 'top'
                         "hold_max_proposals": 2,
                         # 1 = one token per step (hold v1) | 2 = multi-track selection toward the free target (fid v1) |
                         # 3 = "what the heads attend to is what you hear": levels = sharpened attention mass per track
                         "hold_law_version": 3, "hold_level_sharpness": 3.0,
                         "hold_cap_hysteresis": 0.25,  # polyphony cap: how strongly a sounding material keeps its place in the top-K
                         # law version 2 (docs/FIDELITY_CONTRACT.md): several tracks move per step, levels refined
                         # through plan_rows toward the free head-combined target of eq. (36)/(37)
                         "hold_tracks_max": 3, "hold_refine_evals": 10, "hold_refine_tracks": 2,
                         "hold_refine_step": 0.18, "hold_refine_min_step": 0.04, "hold_mixture_candidates": 1},
        "gan_reference_target": 8,
        "gan_components": 2,
        "gan_adversarial_rounds_max": 4,
        "gan": {"reference_temperature": 0.5, "l2": 1e-4, "lr_D": 0.001, "lr_G": 0.001,
                "lambda_fit": 0.25, "lambda_f": 0.5, "C_fail": 100.0, "sigma_init": 0.2,
                "sigma_min": 0.01, "sigma_max": 1.0, "basis_rank": 3, "samples_per_round": 4,
                "inner_steps_D": 40, "inner_steps_G": 10, "fresh_parents": 2, "generator_step_clip": 0.05,
                "generator_gradient_clip": 10.0, "hint_min_correlation": 0.3, "max_hints": 4,
                "importance_ess_min_fraction": 0.5,
                # audit-2: per-commit adversarial update unit (observe_committed)
                "inner_steps_D_per_commit": 8, "inner_steps_G_per_commit": 4, "generator_batch_max": 4,
                "baseline_ema_rate": 0.2, "reference_continuity_seconds": 2.0, "min_block_free_rows": 3,
                # fragment-vocabulary reference distribution (observe_committed statistics use a separate stream)
                "frag_references": 12, "frag_segment_seconds": 2.0, "frag_offset_rows": 0, "frag_discriminator_samples": 2,
                # hold mode (docs/HOLD_CONTRACT.md): references = realizable continuations near the committed
                # history, narrowed by a move/own-flutter band and an absolute cap; generator over origin slots
                "hold_move_scale": 1.0, "hold_reference_candidates": 14,
                "hold_move_over_flutter_min": 3.0, "hold_move_over_flutter_max": 12.0,
                "hold_band_lo": 0.0, "hold_band_hi": 16.0, "hold_level_move": 0.7, "hold_level_floor": 0.3,
                "hold_magnitude_min": 0.3, "hold_magnitude_max": 1.5, "hold_jump_probability": 0.35,
                "hold_search_rows": 4, "hold_keep_min": 2, "hold_keep_max": 3, "hold_fragment_pool": 4,
                "hold_steadiness_factor": 4, "hold_w_sigma_init": 0.08, "hold_w_sigma_min": 0.01,
                "hold_w_sigma_max": 0.30, "hold_flutter_floor": 1e-3,
                "hold_positive_source": "recordings",    # D positives: 'recordings' (unmanipulated playback) | 'plans' (hold v1)
                "hold_real_positives": 8, "hold_psi_blocks": 4, "hold_psi_scale_samples": 24,
                "hold_max_jumps_per_plan": 3,
                # positives drawn near the sound (current / committed / mixture candidates +- offset), a non-saturating
                # generator reward (clipped logit of D on the published plan rows), stronger l2 on the hold discriminator
                "hold_real_near_share": 0.8, "hold_real_offset_seconds": 4.0, "hold_logit_clip": 6.0,
                "hold_d_l2": 1e-3, "hold_lr_G_scale": 20.0,
                # polyphony cap: a fifth origin class "swap" is active when hires.max_active_materials > 0; share of real
                # windows whose sounding set is redrawn; the feature scales of the hold discriminator stay provisional
                # until this share of the features actually varies (the piece starts from silence)
                "hold_real_cap_redraw": 0.5, "hold_psi_scale_min_var": 0.25},
    },
    "history": {"enabled": True, "update_rate": 0.10, "recent_event_capacity": 16,
                "parent_capacity": 8, "cov_regularization": 1e-6,
                "time_constant_seconds": 60.0},   # audit C7: rho(dt) = 1 - exp(-dt / tau_H)
    # audit C5/C6: fixed-reference realization in short commit steps inside the long form
    "realization": {
        "commit_seconds": 1.5,            # length committed to the history per step
        "lookahead_seconds": 10.0,        # window improved against the frozen reference
        "n_reference_proposals": 2,       # proposals per window; one is frozen (chosen by warm-start J)
        "max_refinement_sweeps": 3,       # coordinate-search sweeps over bump amplitudes
        "initial_step": 0.06,             # bump amplitude step (gain units)
        "min_step": 0.01,
        "improvement_epsilon": 1e-6,
        "max_evaluations_per_step": 80,   # acoustic evaluations (window compositions) per step
        "w_relation": 0.25,               # weight of J_rel (mode-supplied relation terms)
        "w_smooth": 0.01,                 # weight of the sampled motion energy on the window
        "bump_short_fraction": 0.5,       # second bump covers this fraction of the lookahead
    },
    "render": {"format": "WAV_FLOAT32", "processing_precision": "float64", "block_frames": 8192,
               "csv_step_seconds": 0.05,
               # output dynamics (user-authorised for the hires extension; disabled by default):
               "master": {"enabled": False, "compressor": {"enabled": True, "threshold_dbfs": -20.0, "ratio": 3.0,
                                                           "attack_ms": 10.0, "release_ms": 250.0, "knee_db": 6.0},
                          "normalize_peak_dbfs": -1.0,
                          "limiter": {"enabled": True, "ceiling_dbfs": -1.0, "lookahead_ms": 5.0, "release_ms": 80.0}}},
    # high-resolution extension (user-authorised 2026-09-16): steep switches, playback-position jumps /
    # splicing, output dynamics.  Opt-in; the baseline pipeline is untouched when disabled.
    "hires": {
        "enabled": False,
        "commit_seconds": 0.5,            # commit / history step
        "lookahead_seconds": 4.0,         # window improved against the frozen reference
        "ramp_seconds": 0.25,             # gain ramp (Q5) for level changes; exempt from the slow-motion rules
        "switch_seconds": 0.05,           # crossfade at playback-position splices
        "position_jumps": True,           # materials may jump to another position of their own source
        "jump_candidates": 2,             # positions per track examined per step (from the solo bank)
        "min_clip_seconds": 2.0,          # minimum time between jumps of one track
        "level_step": 0.12,               # coordinate-search step on end levels
        "max_sweeps": 3,
        "n_reference_proposals": 2,
        "goal_rise_seconds": 12.0,        # smooth goal rise at the end of CONTRACT (convergence gesture)
        "w_relation": 0.25, "w_smooth": 0.0,
        # fragment-vocabulary revision (2026-09-16): candidates scored by clip-averaged fragment
        # features, multi-track jumps via a beam search on exact mixtures, references built on fragments
        "fragment_vocabulary": False,
        "clip_feature_seconds": 2.0,      # fragment features = mean over this many seconds after the position
        "beam_width": 3,
        "candidate_rows": 8,              # rows of the window used while scoring jump combinations
        # reference hold (2026-09-16, docs/HOLD_CONTRACT.md): one frozen ideal is chased for this long
        # before a new one is anchored to the sound.  0 / <= commit_seconds = legacy (re-anchored at
        # every commit, where the ideal follows the sound instead of leading it)
        "reference_hold_seconds": 0.0,
        "reference_selection": "hold_fit",  # 'hold_fit' (legacy: proposal nearest to the unchanged sound) | 'first'
        "reference_anchor": "last_row",     # 'last_row' (legacy) | 'block_mean' (mean of the last committed block)
        # hold mode only: candidates are scored as "this choice now, then the rest of the mode's plan"
        # instead of "this choice held for the whole lookahead window"
        "plan_aware_lookahead": True,
        # the same material may sound at two positions at once (user-authorised 2026-09-17): every material
        # gets this many tracks sharing its PCM; the extra voices start away from the first (1 = off)
        "voices_per_material": 1,
        # hold mode with many tracks: number of tracks on which the realizer tries its OWN jump candidates per
        # commit (the tracks named by the mode's plan step are always examined); 0 = all tracks
        "explore_tracks_max": 0,
        # polyphony cap (user decision 2026-09-17): at most this many MATERIALS sound at once (the voices of one
        # material count once); 0 = no limit.  It holds for the end levels of every commit step; while the 0.25 s
        # ramps cross-fade, outgoing and incoming materials overlap (up to twice the cap) - allowed by the user.
        "max_active_materials": 0,
        "cap_swap_candidates": 2,           # silent materials the realizer tries to swap in per commit (own exploration)
        # a sounding material keeps its place unless a silent one exceeds it by this level margin (0 = the K loudest
        # always win; larger = a calmer cast).  Applied step by step along a plan and at every commit.
        "cap_hysteresis": 0.0,
        # a material that starts to sound keeps sounding at least this long before it may be silenced (the audible
        # counterpart of min_clip_seconds: no material appears for half a second and vanishes); 0 = off.  The goal
        # rise at the end of CONTRACT silences everything regardless (form rule).
        "min_sounding_seconds": 0.0,
    },
    "numerics": {"gain_bound_tolerance": 1e-9, "motion_relative_margin": 1e-6,
                 "db_rate_subintervals": 128},
    "calibration_status": "UNVERIFIED",
    "perceptual_status": "UNVERIFIED",
}


PROFILE_SCALE = {"baseline": 1.0, "responsive_x2_UNVERIFIED": 2.0, "responsive_x4_UNVERIFIED": 4.0,
                 "responsive_x8_UNVERIFIED": 8.0}


def apply_motion_profile(motion: Dict[str, Any]) -> Dict[str, Any]:
    """Scale the motion limits for the named profile (time compression by k, audit E2)."""
    k = PROFILE_SCALE.get(str(motion.get("profile", "baseline")))
    if k is None:
        raise ValueError(f"unknown motion.profile {motion.get('profile')!r}; known: {sorted(PROFILE_SCALE)}")
    m = dict(motion)
    if k != 1.0:
        m["velocity_max"] = float(motion["velocity_max"]) * k
        m["acceleration_max"] = float(motion["acceleration_max"]) * k * k
        m["jerk_max"] = float(motion["jerk_max"]) * k ** 3
        m["regularized_db_rate_max"] = float(motion["regularized_db_rate_max"]) * k
        m["velocity_min_bulk"] = float(motion["velocity_min_bulk"]) * k
        m["mean_velocity_min"] = float(motion["mean_velocity_min"]) * k
        m["ramp_component_max_seconds"] = float(motion["ramp_component_max_seconds"]) / k
    m["profile_scale_k"] = k
    return m


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# Explicit aliases for configuration keys used by other editions of the v1.1 document (audit §7).
# Section aliases map a whole sub-dictionary; key aliases map one key.
SECTION_ALIASES = {"schedule": "form"}
KEY_ALIASES = {
    ("motion", "relative_rate_max"): ("motion", "regularized_db_rate_max"),
    ("motion", "ramp_component_duration_max_seconds"): ("motion", "ramp_component_max_seconds"),
    ("search", "max_rounds"): ("search", "max_search_rounds"),
    ("search", "candidate_count"): ("search", "candidate_bank_target"),
    ("search", "candidate_bank_size"): ("search", "candidate_bank_target"),
    ("search", "acceptance_margin"): ("search", "acceptance_margin"),
}


def apply_aliases(user: Dict[str, Any]) -> tuple:
    """Return (canonical user config, list of applied aliases)."""
    out: Dict[str, Any] = {}
    applied = []
    for k, v in user.items():
        if k in SECTION_ALIASES and isinstance(v, dict):
            tgt = SECTION_ALIASES[k]
            out.setdefault(tgt, {})
            out[tgt] = deep_merge(out[tgt], v) if isinstance(out[tgt], dict) else copy.deepcopy(v)
            applied.append(f"{k}.* -> {tgt}.*")
        else:
            out[k] = copy.deepcopy(v) if k not in out else deep_merge(out[k], v)
    for (sec, key), (tsec, tkey) in KEY_ALIASES.items():
        if isinstance(out.get(sec), dict) and key in out[sec] and (sec, key) != (tsec, tkey):
            val = out[sec].pop(key)
            out.setdefault(tsec, {})[tkey] = val
            applied.append(f"{sec}.{key} -> {tsec}.{tkey}")
    return out, applied


def unknown_keys(user: Dict[str, Any], ref: Dict[str, Any], prefix: str = "") -> list:
    bad = []
    for k, v in user.items():
        path = f"{prefix}{k}"
        if k not in ref:
            bad.append(path)
            continue
        if isinstance(v, dict) and isinstance(ref[k], dict) and ref[k]:
            bad += unknown_keys(v, ref[k], path + ".")
    return bad


def load_config(path: str, mode_override: str | None = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    if not isinstance(user, dict):
        raise ValueError("config must be a JSON object")
    user, applied = apply_aliases(user)
    bad = unknown_keys(user, DEFAULTS)
    if bad:
        raise ValueError("unknown configuration key(s) (would be silently ignored): " + ", ".join(bad)
                         + ". Known aliases: schedule.* -> form.*, " +
                         ", ".join(f"{a}.{b} -> {c}.{d}" for (a, b), (c, d) in KEY_ALIASES.items() if (a, b) != (c, d)))
    cfg = deep_merge(DEFAULTS, user)
    cfg["config_aliases_applied"] = applied
    if mode_override:
        cfg["mode"] = mode_override
    cfg["motion"] = apply_motion_profile(cfg["motion"])
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    if cfg["mode"] not in MODE_IDS:
        raise ValueError(f"unknown mode {cfg['mode']!r}; expected one of {sorted(MODE_IDS)}")
    if not isinstance(cfg["materials"], list) or len(cfg["materials"]) < 2:
        raise ValueError("config.materials must list at least 2 material WAV paths (N >= 2)")
    if not isinstance(cfg["goal"], str) or not cfg["goal"]:
        raise ValueError("config.goal must be exactly one WAV path")
    if not cfg["history"].get("enabled", True):
        raise ValueError("history.enabled=false is not permitted for official generation (spec §13.2)")
    m = cfg["motion"]
    for k in ("velocity_max", "velocity_min_bulk", "mean_velocity_min", "acceleration_max", "jerk_max",
              "regularized_db_rate_max", "gain_log_epsilon", "minimum_move_amplitude",
              "slow_fraction_max", "ramp_component_max_seconds"):
        v = float(m[k])
        if not (v > 0) or v != v:
            raise ValueError(f"motion.{k} must be a finite positive number")
    if m["velocity_min_bulk"] >= m["velocity_max"]:
        raise ValueError("motion.velocity_min_bulk must be below velocity_max")
    f = cfg["form"]
    if int(f["cycles"]) < 1:
        raise ValueError("form.cycles must be >= 1")
    for k in ("intro_seconds", "open_seconds", "contract_seconds", "goal_hold_seconds", "reopen_seconds"):
        if float(f[k]) < 0:
            raise ValueError(f"form.{k} must be >= 0")
    a = cfg["analysis"]
    if int(a["window_frames"]) < 64 or float(a["hop_seconds"]) <= 0:
        raise ValueError("analysis.window_frames/hop_seconds invalid")
    s = cfg["search"]
    for k in ("candidate_bank_target", "max_total_candidates_per_cycle", "max_search_rounds"):
        if int(s[k]) < 1:
            raise ValueError(f"search.{k} must be >= 1")
    if s["selection"] not in ("argmin", "softmax_within_margin"):
        raise ValueError("search.selection must be 'argmin' or 'softmax_within_margin'")
    if float(s["acceptance_margin"]) < 0:
        raise ValueError("search.acceptance_margin must be >= 0")
    if s["tolerance_applies_to"] not in ("mode_error", "target_fit"):
        raise ValueError("search.tolerance_applies_to must be 'mode_error' or 'target_fit'")
    if s["final_targets"] not in ("best_round", "last_round"):
        raise ValueError("search.final_targets must be 'best_round' or 'last_round'")
    r = cfg["realization"]
    if float(r["commit_seconds"]) <= 0 or float(r["lookahead_seconds"]) < float(r["commit_seconds"]):
        raise ValueError("realization.lookahead_seconds must be >= commit_seconds > 0")
    if float(cfg["history"]["time_constant_seconds"]) <= 0:
        raise ValueError("history.time_constant_seconds must be > 0")
    hz = cfg["hires"]                     # .get: resolved configs of older traces lack these keys
    if float(hz.get("reference_hold_seconds", 0.0)) < 0:
        raise ValueError("hires.reference_hold_seconds must be >= 0")
    if hz.get("reference_selection", "hold_fit") not in ("hold_fit", "first"):
        raise ValueError("hires.reference_selection must be 'hold_fit' or 'first'")
    if int(hz.get("max_active_materials", 0)) < 0:
        raise ValueError("hires.max_active_materials must be >= 0 (0 = no limit)")
    if not (1 <= int(hz.get("voices_per_material", 1)) <= 3):
        raise ValueError("hires.voices_per_material must be 1, 2 or 3")
    if hz.get("reference_anchor", "last_row") not in ("last_row", "block_mean"):
        raise ValueError("hires.reference_anchor must be 'last_row' or 'block_mean'")
    ge = cfg["form"]["goal_exposure"]
    if ge["policy"] not in ("contract_only", "free", "contract_law"):
        raise ValueError("form.goal_exposure.policy must be 'contract_only', 'free' or 'contract_law'")
    if ge.get("convergence", "share") not in ("share", "sparsity", "presence"):
        raise ValueError("form.goal_exposure.convergence must be 'share', 'sparsity' or 'presence'")
    for k in ("intro_max", "open_max"):
        if not (0.0 <= float(ge[k]) <= 1.0):
            raise ValueError(f"form.goal_exposure.{k} must be in [0, 1]")

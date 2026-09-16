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
        "goal_exposure": {"policy": "contract_only", "intro_max": 0.0, "open_max": 0.0,
                          "reopen_descend_within_reopen": True},
    },
    "analysis": {
        "window_frames": 2048,
        "hop_seconds": 0.1,
        "spectral_bands": 8,
        "band_edges_hz": [0, 100, 200, 400, 800, 1600, 3200, 6400],  # upper edge = Nyquist
        "silence_energy": 1e-10,
        "feature_std_floor": 0.02,
        "feature_epsilon": 1e-12,
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
    },
    "mode_defaults": {
        "diffusion_particles": 4,
        "diffusion_internal_steps_max": 8,
        "diffusion": {"lambda_phi": 1.0, "lambda_2": 1.0, "lambda_3": 0.25, "lambda_G": 1.0,
                       "epsilon_D": 1e-4, "step": 0.01, "temperature_per_openness": 0.05,
                       "kappa_E": 0.25, "kappa_M": 0.5, "anchor_nongoal_gain": 0.4,
                       "anchor_goal_gain": 0.1, "time_correlation": 0.8,
                       "correlation_significance": 0.2, "similarity_sign_threshold": 0.5, "max_triples": 10,
                       "init_spread": 0.5, "realized_pull": 0.5, "search_matrix_scale": 1.0, "hint_pairs": 3},
        "vae_latent_dim": 2,
        "vae": {"mu_H_init": 0.5, "sigma_H_init": 0.04, "K_F": 0.1, "ridge_epsilon": 1e-6,
                "cov_floor": 1e-4, "probe_goal_gain": 0.1, "probe_in_gain": 0.7,
                "probe_out_gain": 0.15, "time_correlation": 0.9},
        "transformer_heads": ["similarity", "contrast", "memory"],
        "transformer": {"sigma_F": 1.0, "tau_H_seconds": 180.0, "alpha_s": 0.25, "alpha_c": 0.25,
                         "alpha_m": 0.25, "alpha_G": 1.0, "cov_diag_floor": 1e-4,
                         "probe_foreground_gain": 0.7, "probe_background_gain": 0.15,
                         "bias_scale": 0.5, "noise_scale": 0.5, "ar_stride": 1, "time_correlation": 0.9,
                         "excursion_radius": 0.35, "feedback_rate": 0.5, "feedback_bound": 3.0,
                         "step_seconds_reference": 0.1, "tendency_gain": 0},
        "gan_reference_target": 8,
        "gan_components": 2,
        "gan_adversarial_rounds_max": 4,
        "gan": {"reference_temperature": 0.5, "l2": 1e-4, "lr_D": 0.001, "lr_G": 0.001,
                "lambda_fit": 0.25, "lambda_f": 0.5, "C_fail": 100.0, "sigma_init": 0.2,
                "sigma_min": 0.01, "sigma_max": 1.0, "basis_rank": 3, "samples_per_round": 4,
                "inner_steps_D": 40, "inner_steps_G": 10, "fresh_parents": 2, "generator_step_clip": 0.05,
                "generator_gradient_clip": 10.0, "hint_min_correlation": 0.3, "max_hints": 4,
                "importance_ess_min_fraction": 0.5},
    },
    "history": {"enabled": True, "update_rate": 0.10, "recent_event_capacity": 16,
                "parent_capacity": 8, "cov_regularization": 1e-6},
    "render": {"format": "WAV_FLOAT32", "processing_precision": "float64", "block_frames": 8192,
               "csv_step_seconds": 0.05},
    "numerics": {"gain_bound_tolerance": 1e-9, "motion_relative_margin": 1e-6,
                 "db_rate_subintervals": 128},
    "calibration_status": "UNVERIFIED",
    "perceptual_status": "UNVERIFIED",
}


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
    ge = cfg["form"]["goal_exposure"]
    if ge["policy"] not in ("contract_only", "free"):
        raise ValueError("form.goal_exposure.policy must be 'contract_only' or 'free'")
    for k in ("intro_max", "open_max"):
        if not (0.0 <= float(ge[k]) <= 1.0):
            raise ValueError(f"form.goal_exposure.{k} must be in [0, 1]")

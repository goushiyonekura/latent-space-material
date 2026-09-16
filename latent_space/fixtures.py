"""Synthetic development fixtures (spec §16.5): materials of different lengths with evolving
harmonic content, amplitude envelopes, sustained vs. dense textures; a short sustained goal."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .audio_io import write_wav_float32


def _env(rng: np.random.Generator, n: int, fs: int, rate_hz: float) -> np.ndarray:
    t = np.arange(n) / fs
    k = max(2, int(rate_hz * n / fs) + 2)
    knots = rng.uniform(0.2, 1.0, k)
    xs = np.linspace(0, t[-1], k)
    return np.interp(t, xs, knots)


def synth_material(rng: np.random.Generator, fs: int, seconds: float, kind: str) -> np.ndarray:
    n = int(seconds * fs)
    t = np.arange(n) / fs
    out = np.zeros((n, 2))
    if kind == "harmonic_drift":
        f0 = rng.uniform(110, 220)
        for h in range(1, 7):
            drift = 1.0 + 0.01 * np.sin(2 * np.pi * rng.uniform(0.02, 0.08) * t + rng.uniform(0, 6.28))
            amp = _env(rng, n, fs, 0.1) / h
            pan = rng.uniform(0.3, 0.7)
            s = amp * np.sin(2 * np.pi * f0 * h * drift * t)
            out[:, 0] += s * pan
            out[:, 1] += s * (1 - pan)
    elif kind == "noise_texture":
        dens = _env(rng, n, fs, 0.15)
        noise = rng.standard_normal((n, 2))
        # simple one-pole lowpass with time-varying cutoff via env
        y = np.zeros_like(noise)
        a = 0.02 + 0.3 * dens
        for c in range(2):
            acc = 0.0
            col = noise[:, c]
            for k in range(n):
                acc += a[k] * (col[k] - acc)
                y[k, c] = acc
        out = y * (0.8 * _env(rng, n, fs, 0.2))[:, None]
    elif kind == "pulse_density":
        rate = _env(rng, n, fs, 0.1) * 12.0  # pulses per second
        phase = np.cumsum(rate) / fs
        pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
        f = rng.uniform(300, 900)
        ring = np.exp(-np.arange(int(0.08 * fs)) / (0.02 * fs)) * np.sin(2 * np.pi * f * np.arange(int(0.08 * fs)) / fs)
        s = np.convolve(pulses, ring)[:n]
        out[:, 0] = s * 0.9
        out[:, 1] = np.roll(s, int(0.003 * fs)) * 0.7
    else:  # chord_pad
        base = rng.uniform(130, 260)
        for ratio in (1.0, 1.25, 1.5, 2.0):
            vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 6.28))
            s = np.sin(2 * np.pi * base * ratio * vib * t) * (0.5 + 0.5 * _env(rng, n, fs, 0.05))
            out[:, 0] += s * 0.5
            out[:, 1] += s * 0.45
    out = out * 0.9 / (np.max(np.abs(out)) + 1e-9)
    fade = np.minimum(1.0, np.arange(n) / (0.05 * fs))
    out *= (fade * fade[::-1])[:, None]
    return out.astype(np.float32)


def write_fixture_set(out_dir: str, fs: int = 22050, N: int = 4, seed: int = 7,
                      hard: bool = False, material_seconds: List[float] | None = None) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    kinds = ["harmonic_drift", "noise_texture", "pulse_density", "chord_pad"]
    secs = material_seconds or [24.0 + 7.0 * (k % 3) + 2.0 * k for k in range(N)]
    paths = []
    for k in range(N):
        kind = "harmonic_drift" if hard else kinds[k % len(kinds)]
        x = synth_material(rng if not hard else np.random.default_rng(seed), fs, secs[k], kind)
        if hard:
            x = x * (0.5 + 0.1 * k)
        p = os.path.join(out_dir, f"material_{k + 1:02d}.wav")
        write_wav_float32(p, x, fs)
        paths.append(p)
    goal = synth_material(rng, fs, 8.0, "chord_pad")
    if hard:
        goal = synth_material(np.random.default_rng(seed + 99), fs, 8.0, "noise_texture")
    gp = os.path.join(out_dir, "goal.wav")
    write_wav_float32(gp, goal, fs)
    return {"materials": paths, "goal": gp, "fs": fs}


def write_fixture_config(path: str, fixture: Dict[str, object], mode: str = "vae", seed: int = 48291,
                         extra: Dict | None = None) -> str:
    cfg = {
        "spec_version": "1.1", "mode": mode, "seed": seed,
        "materials": [os.path.abspath(p) for p in fixture["materials"]],
        "goal": os.path.abspath(str(fixture["goal"])),
        "form": {"cycles": 2, "intro_seconds": 8, "open_seconds": 10, "contract_seconds": 24,
                 "goal_hold_seconds": 3, "reopen_seconds": 20},
        "search": {"candidate_bank_target": 8, "max_total_candidates_per_cycle": 24, "max_search_rounds": 3},
        "calibration_status": "UNVERIFIED",
    }
    if extra:
        from .config import deep_merge
        cfg = deep_merge(cfg, extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return path

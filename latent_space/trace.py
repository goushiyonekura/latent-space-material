"""Outputs: result.wav, gain_curves.csv, state_trace.json (spec §14)."""
from __future__ import annotations

import json
import os
from typing import Any, List

import numpy as np

from .audio_io import write_wav_float32
from .curves import TrackCurve


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_gain_csv(path: str, curves: List[TrackCurve], fs: int, total_frames: int, step_seconds: float,
                   names: List[str]) -> None:
    step = max(1, int(round(step_seconds * fs)))
    frames = np.arange(0, total_frames, step, dtype=np.int64)
    cols = [cv.values(frames) * 100.0 for cv in curves]
    with open(path, "w", encoding="utf-8") as f:
        f.write("time_seconds," + ",".join(names) + "\n")
        for k, fr in enumerate(frames):
            f.write(f"{fr / fs:.4f}," + ",".join(f"{col[k]:.4f}" for col in cols) + "\n")


def write_trace(path: str, trace: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, cls=NumpyEncoder, indent=1, ensure_ascii=False)


def write_result_wav(path: str, y: np.ndarray, fs: int) -> None:
    write_wav_float32(path, y, fs)

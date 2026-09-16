"""Minimal WAV reader/writer (numpy only).

Reads PCM 8/16/24/32-bit and IEEE float 32/64 WAV (including WAVE_FORMAT_EXTENSIBLE),
writes IEEE float32 WAV.  No resampling, no channel conversion (spec §1.1 forbids
silent auto-conversion; mismatches are reported by the caller).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

_PCM = 1
_IEEE_FLOAT = 3
_EXTENSIBLE = 0xFFFE


@dataclass
class WavInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    bits: int
    format_tag: int
    sha256: str


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav(path: str):
    """Return (data float32 array of shape (frames, channels), WavInfo)."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 12 or raw[:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body_start = pos + 8
        body_end = min(body_start + size, len(raw))
        if cid == b"fmt ":
            fmt = raw[body_start:body_end]
        elif cid == b"data":
            data = raw[body_start:body_end]
            if size == 0xFFFFFFFF:  # RF64-style unknown size: take to end
                data = raw[body_start:]
        pos = body_start + size + (size & 1)
        if cid == b"data" and size == 0xFFFFFFFF:
            break
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, fs, _byte_rate, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _EXTENSIBLE:
        if len(fmt) < 26:
            raise ValueError(f"{path}: malformed extensible fmt chunk")
        tag = struct.unpack("<H", fmt[24:26])[0]
    if channels < 1:
        raise ValueError(f"{path}: invalid channel count {channels}")
    frame_bytes = block_align if block_align else channels * (bits // 8)
    n_frames = len(data) // frame_bytes
    data = data[: n_frames * frame_bytes]
    if tag == _PCM:
        if bits == 8:
            arr = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 16:
            arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 24:
            b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
            v = np.where(v >= (1 << 23), v - (1 << 24), v)
            arr = v.astype(np.float32) / float(1 << 23)
        elif bits == 32:
            arr = np.frombuffer(data, dtype="<i4").astype(np.float32) / float(1 << 31)
        else:
            raise ValueError(f"{path}: unsupported PCM bit depth {bits}")
    elif tag == _IEEE_FLOAT:
        if bits == 32:
            arr = np.frombuffer(data, dtype="<f4").astype(np.float32)
        elif bits == 64:
            arr = np.frombuffer(data, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"{path}: unsupported float bit depth {bits}")
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag} (need PCM or IEEE float)")
    arr = arr.reshape(n_frames, channels)
    info = WavInfo(path=path, sample_rate=int(fs), channels=int(channels), frames=int(n_frames),
                   bits=int(bits), format_tag=int(tag), sha256=sha256_of_file(path))
    return np.ascontiguousarray(arr), info


def write_wav_float32(path: str, data: np.ndarray, sample_rate: int) -> None:
    """Write (frames, channels) float array as IEEE float32 WAV."""
    if data.ndim == 1:
        data = data[:, None]
    frames, channels = data.shape
    payload = np.ascontiguousarray(data, dtype="<f4").tobytes()
    bits = 32
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHHH", _IEEE_FLOAT, channels, sample_rate, byte_rate, block_align, bits, 0)
    fact = struct.pack("<I", frames)
    riff_size = 4 + (8 + len(fmt)) + (8 + len(fact)) + (8 + len(payload)) + (len(payload) & 1)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"fact" + struct.pack("<I", len(fact)) + fact)
        f.write(b"data" + struct.pack("<I", len(payload)) + payload)
        if len(payload) & 1:
            f.write(b"\x00")

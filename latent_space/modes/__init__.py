"""Mode registry.  Each mode lives in its own module; a missing/broken module raises
IncompleteImplementation so the engine reports INCOMPLETE_IMPLEMENTATION instead of
substituting another mode (spec §15.1)."""
from __future__ import annotations

import importlib
import traceback

from ..config import MODE_IDS

_MODULES = {
    "diffusion": ("latent_space.modes.diffusion", "DiffusionMode"),
    "vae": ("latent_space.modes.vae", "VAEMode"),
    "transformer": ("latent_space.modes.transformer", "TransformerMode"),
    "gan": ("latent_space.modes.gan", "GANMode"),
}


class IncompleteImplementation(Exception):
    pass


def make_mode(cli_name: str, cfg: dict, analyzer, fs: int, rng, objective):
    if cli_name not in _MODULES:
        raise IncompleteImplementation(f"unknown mode {cli_name!r}")
    modname, clsname = _MODULES[cli_name]
    try:
        mod = importlib.import_module(modname)
        cls = getattr(mod, clsname)
    except Exception as e:  # noqa: BLE001
        raise IncompleteImplementation(
            f"mode {cli_name!r} ({MODE_IDS[cli_name]}) is not implemented/importable: {e}\n"
            + traceback.format_exc()) from e
    inst = cls(cfg, analyzer, fs, rng, objective)
    inst.cli_name = cli_name
    inst.internal_id = MODE_IDS[cli_name]
    return inst

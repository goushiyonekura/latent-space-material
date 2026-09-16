"""CLI:  python -m latent_space generate --config project.json --mode diffusion --output out/diffusion"""
from __future__ import annotations

import argparse
import json
import sys

from .config import MODE_IDS, load_config
from .engine import Job


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="latent_space", description="《潜在空間》 headless composition-material generator (spec v1.1 H5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate result.wav / gain_curves.csv / state_trace.json")
    g.add_argument("--config", required=True, help="project JSON")
    g.add_argument("--mode", choices=sorted(MODE_IDS), default=None, help="overrides config.mode")
    g.add_argument("--output", "--out", dest="output", required=True, help="output directory (alias: --out)")
    g.add_argument("--seed", type=int, default=None, help="overrides config.seed")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        cfg = load_config(args.config, args.mode)
        if args.seed is not None:
            cfg["seed"] = int(args.seed)
        job = Job(cfg, args.output, config_path=args.config, mode_override=args.mode)
        res = job.run()
        print(json.dumps(res, indent=1))
        return 0 if res["status"] in ("VALID_APPROXIMATION", "BEST_EFFORT") else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/bin/sh
# Generate all four modes from project.json into output/<mode>/
set -e
cd "$(dirname "$0")/.."
for mode in diffusion vae transformer gan; do
  python3 -m latent_space generate --config project.json --mode "$mode" --output "output/$mode"
done

#!/bin/sh
# Decode MP3 materials to float32 WAV with macOS's built-in afconvert (pure decoding; no
# resampling / channel conversion is performed — the engine refuses mixed sample rates).
#
#   sh scripts/prepare_inputs.sh                      # original five files at the repo root -> inputs/
#   sh scripts/prepare_inputs.sh SRC_DIR OUT_DIR      # every *.mp3 in SRC_DIR -> OUT_DIR/<name>.wav
#                                                     # (*.wav in SRC_DIR are copied unchanged)
set -e
cd "$(dirname "$0")/.."
if [ "$#" -eq 2 ]; then
  src="$1"; out="$2"
  mkdir -p "$out"
  for f in "$src"/*.mp3; do
    [ -f "$f" ] || continue
    b="$(basename "$f" .mp3)"
    afconvert -f WAVE -d LEF32 "$f" "$out/$b.wav"
    echo "decoded $b"
  done
  for f in "$src"/*.wav; do
    [ -f "$f" ] || continue
    cp "$f" "$out/$(basename "$f")"
    echo "copied $(basename "$f")"
  done
  exit 0
fi
mkdir -p inputs
for f in material_GOAL_001 material_cello_001 material_prism_001 material_scri_001 material_stringquartet_001; do
  afconvert -f WAVE -d LEF32 "$f.mp3" "inputs/$f.wav"
  echo "decoded $f"
done

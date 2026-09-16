#!/bin/sh
# Decode the MP3 materials to float32 WAV with macOS's built-in afconvert (pure decoding; no
# resampling / channel conversion is performed — all sources are 44.1 kHz stereo already).
set -e
cd "$(dirname "$0")/.."
mkdir -p inputs
for f in material_GOAL_001 material_cello_001 material_prism_001 material_scri_001 material_stringquartet_001; do
  afconvert -f WAVE -d LEF32 "$f.mp3" "inputs/$f.wav"
  echo "decoded $f"
done

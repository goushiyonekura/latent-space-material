#!/bin/sh
# Hold-mode versions (docs/HOLD_CONTRACT.md, docs/FIDELITY_CONTRACT.md): generate the given modes in
# parallel from project.$TAG.<mode>.json into $OUT/<mode>/, the "realizer ignores the mode"
# counterfactuals into dev/out/ignore_$TAG/<mode>/, then print the closeness / tracking table.
#   sh scripts/run_hold.sh                               # TAG=hold OUT=output_hold, all four modes
#   TAG=fid OUT=output_fid sh scripts/run_hold.sh vae gan
#   COMPARE="output_hold output_frag" ...                # extra directories for the table
cd "$(dirname "$0")/.."
TAG="${TAG:-hold}"
OUT="${OUT:-output_hold}"
COMPARE="${COMPARE:-output_frag}"
MODES="${*:-diffusion vae transformer gan}"
mkdir -p "dev/out/ignore_$TAG" "$OUT"
for mode in $MODES; do
  python3 -m latent_space generate --config "project.$TAG.$mode.json" --mode "$mode" --output "$OUT/$mode" > "dev/out/run_${TAG}_$mode.log" 2>&1 &
  python3 scripts/closeness.py --ignore-config "project.$TAG.$mode.json" "dev/ignore_${TAG}_$mode.json" > /dev/null
  python3 -m latent_space generate --config "dev/ignore_${TAG}_$mode.json" --mode "$mode" --output "dev/out/ignore_$TAG/$mode" > "dev/out/run_ignore_${TAG}_$mode.log" 2>&1 &
done
wait
rm -f dev/out/ignore_$TAG/*/result.wav
python3 scripts/closeness.py "$OUT" "dev/out/ignore_$TAG" $COMPARE --modes="$(echo $MODES | tr ' ' ',')" --json="dev/out/closeness_$TAG.json" | tail -n 16

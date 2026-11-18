#!/usr/bin/env bash
# Run every configuration in a config file, one after another, on one GPU.
#   bash scripts/run_experiment.sh configs/e1_dose.txt
# Config file: first non-comment line holds shared flags, each later line is "<tag> <quant-spec> [extra flags]".
# Runs are resumable: finished (task, episode) pairs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source setup/cluster_env.sh
CFG="$1"
ONLY="${2:-}"          # optional: run only the line with this tag (used by the Slurm array)
SHARED=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  if [[ -z "$SHARED" ]]; then SHARED="$line"; continue; fi
  read -r tag spec extra <<<"$line"
  [[ -n "$ONLY" && "$ONLY" != "$tag" ]] && continue
  echo "=== $tag ($spec) ==="
  # shellcheck disable=SC2086
  python scripts/run_rollouts.py $SHARED --tag "$tag" --quant "$spec" $extra
done < "$CFG"

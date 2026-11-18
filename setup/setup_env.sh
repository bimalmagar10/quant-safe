#!/usr/bin/env bash
# Environment for the GPU node (Linux only: hf-libero has no macOS build).
# Usage:  bash setup/setup_env.sh            # creates conda env "qsafe"
# Then:   conda activate qsafe && hf auth login && python setup/check_env.py
set -euo pipefail
ENV_NAME="${ENV_NAME:-qsafe}"
source "$(dirname "$0")/cluster_env.sh"   # H100 module tree + CUDA 12.9.1

if ! command -v conda >/dev/null; then
  echo "conda not found. Load your cluster's conda/miniforge module first." >&2; exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "^${ENV_NAME}\s" || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"

# lerobot 0.7.0 is the first PyPI release with the LIBERO env under lerobot.sim.integrations,
# which scripts/run_rollouts.py is written against.
pip install "lerobot[libero,pi,smolvla,evaluation]==0.7.0"
# MuJoCo >= 3.4 changes box-box collision resolution and alters at least one LIBERO-Spatial
# initial state (reported by VLAQuantBench). Pin so contact forces are comparable across runs.
pip install "mujoco==3.3.2"
pip install pandas tabulate pytest

# First import of LIBERO may ask where datasets live; answer "N" to take the defaults.
echo N | MUJOCO_GL=egl python -c "from libero.libero import benchmark; print('libero suites:', list(benchmark.get_benchmark_dict())[:5])" || true

cat <<'MSG'

Done. Remaining manual steps:
  1. pi0 / pi0.5 need the gated PaliGemma tokenizer. Accept the licence at
     https://huggingface.co/google/paligemma-3b-pt-224 and run:  hf auth login
     (SmolVLA needs no gated model; use --policy-type smolvla to test without it.)
  2. Every shell that runs rollouts needs:  source setup/cluster_env.sh
     (loads the H100 CUDA module and sets MUJOCO_GL=egl)
  3. python setup/check_env.py
MSG

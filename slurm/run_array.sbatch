#!/bin/bash
# One configuration per array task, one GPU each. Edit partition/account for your cluster.
#   sbatch --array=1-7 slurm/run_array.sbatch configs/e1_dose.txt
# With a single GPU the array tasks simply queue behind each other; with 4 GPUs they run 4 at a time.
#SBATCH --job-name=qsafe
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
source setup/cluster_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-qsafe}"
CFG="$1"
# array index N = N-th configuration line (the first non-comment line is the shared flags)
TAG=$(grep -vE '^\s*(#|$)' "$CFG" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | awk '{print $1}')
bash scripts/run_experiment.sh "$CFG" "$TAG"

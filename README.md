# quant-safe

Do continuous physical-safety margins detect quantization damage in VLA
policies that binary task success hides, and which component's precision
governs that damage?

- `docs/LITERATURE.md`: what exists, the gap, nearest prior art, scoop risk
- `docs/PLAN.md`: recommendation, preliminary evidence, experiments, kill criteria
- `results/prelim_public/REPORT.md`: preliminary result from public data

## Layout

```
qsafe/quant.py     fake RTN quantization (weights + activations), per-component scopes
qsafe/safety.py    MuJoCo safety recorder (contact force, clearance, bystander displacement, jerk) + margins
qsafe/stats.py     paired tests, clustered bootstrap, sample-size formulas
scripts/prelim_public_steps.py   CPU only: re-analysis of VLAQuantBench per-episode records
scripts/run_rollouts.py          GPU: one quantization config on one LIBERO suite, with safety logging
scripts/analyze.py               paired comparison against full precision + verdict per config
scripts/run_experiment.sh        runs every config in a configs/*.txt file
setup/setup_env.sh, setup/check_env.py, slurm/run_array.sbatch, configs/, tests/
```

## On a laptop (no GPU)

```bash
# reproduce the public-data result
git clone --depth 1 https://github.com/jiuyixu25/VLAQuantBench /tmp/vqb
python scripts/prelim_public_steps.py /tmp/vqb/results --out results/prelim_public

# unit tests
uv run --no-project --python 3.12 --with torch --with "mujoco==3.3.2" --with "numpy<2.3" \
  --with scipy --with pandas --with tabulate --with pytest pytest -q tests
```

## On the GPU node (Linux)

```bash
source setup/cluster_env.sh             # module use <h100 tree>; module load cuda/12.9.1; MUJOCO_GL=egl
bash setup/setup_env.sh                 # conda env "qsafe": lerobot 0.7.0 + LIBERO + mujoco 3.3.2
conda activate qsafe
hf auth login                           # after accepting the PaliGemma licence (needed by pi0 / pi0.5)
python setup/check_env.py               # 5 stages; fix whatever fails before going on

python scripts/run_rollouts.py --list-modules        # confirm each component regex catches layers
bash scripts/run_experiment.sh configs/e0_smoke.txt  # minutes; note the seconds per episode
python scripts/analyze.py results/smoke/pi05/libero_spatial

bash scripts/run_experiment.sh configs/e1_dose.txt   # the go / no-go experiment
python scripts/analyze.py results/rollouts/pi05/libero_spatial
```

With Slurm: set the partition in `slurm/run_array.sbatch`, then
`sbatch --array=1-7 slurm/run_array.sbatch configs/e1_dose.txt`.

Runs are resumable. Each config writes `episodes.jsonl` (per-episode metrics)
and one `.npz` per episode (per-step signals), so thresholds can be changed
afterwards: `python scripts/analyze.py <dir> --rescore --force-n 50`.

## Status

The rollout and environment-check scripts have not been executed: LIBERO does
not install on macOS. See the last section of `docs/PLAN.md`.

# Plan: pursue or kill "safety margins under VLA quantization"

## Recommendation

**Pursue, with a two-week gate.** The gap is real and the premise has support
in public data. The safety-specific claim is untested: nobody, including this
repo, has yet measured a force or clearance on a quantized VLA. E1 below
answers that in roughly one GPU-day. If E1 is null, stop.

## Claims the paper would make

- **C1 (detection).** Inside the precision range where binary success is
  statistically unchanged, continuous safety margins shift, and they need far
  fewer episodes to show it.
- **C2 (early warning).** Margins degrade monotonically with decreasing
  precision ahead of the success cliff.
- **C3 (attribution).** For a given architecture, one component's precision
  accounts for most of the margin loss, and it need not be the component that
  governs success.

C1 alone is a workshop paper. C1 + C2 on two architectures is a main-track
evaluation paper. C3 turns it into something a method can be built on
(margin-aware bit allocation).

## Preliminary evidence already in hand

### P0. Episode length shifts where success does not (public data, done)

`scripts/prelim_public_steps.py` re-analyses VLAQuantBench's released per-episode
records. Episodes are paired with the full-precision run on the same task and
initial state; steps-to-success is compared on pairs where both runs succeeded.
Output: `results/prelim_public/REPORT.md`.

- 267 quantized LIBERO cells have at least 30 both-success pairs.
- In **181** of them success is not detectably different from full precision
  (McNemar p > 0.05, uncorrected, which is generous to success).
- In **55 of those 181 (30 %)**, steps-to-success shifts with Holm-corrected
  p < 0.05 across all cells. 43 are slower, 12 faster.
- π0.5, LIBERO-Spatial, end-to-end RTN, 200 episodes per cell:

| Preset | Success (FP 99.0 %) | McNemar p | Steps ratio vs FP (95 % CI) |
|---|---|---|---|
| W8 | 98.5 | 1.00 | 1.00 (0.995 to 1.01) |
| W8A8 | 98.5 | 1.00 | 1.00 (0.997 to 1.01) |
| W4 | 97.5 | 0.38 | 1.01 (0.999 to 1.02) |
| W4A8 | 99.0 | 1.00 | 1.02 (1.00 to 1.04) |
| W3 | 99.0 | 1.00 | **1.05 (1.04 to 1.07)** |
| W4A6 | 97.5 | 0.38 | **1.09 (1.07 to 1.10)** |
| W4A4 | 0.0 | 5e-60 | no successes |

- Noise floor: four full-precision replicate runs of π0.5 give ratios of 0.98,
  0.99, 1.00 and 1.01, none significant after correction.

What this shows: a continuous closed-loop measurement sees a dose-dependent
change that success cannot see at 200 episodes, and it rises before the cliff.

What it does not show:

- Steps is not a safety quantity. Slower is not less safe.
- W8 and W4 show nothing, and W4A8 shows a small shift (+2 %, Holm p < 0.001).
  The clear effect lives at W3 and at A6, which is more
  aggressive than what LoopQ calls safe. If margins behave like steps, 8-bit
  will be a clean null and the paper is about the 3 to 4 bit / low-activation
  band.
- The effect is concentrated in π0.5 (35 of its 74 flat cells) and
  OpenVLA-OFT (13 of 39). π0 and X-VLA show little (3 of 31, 4 of 37).
- The p-values treat episodes as independent; the confidence intervals
  resample tasks and are the more honest number. There are only 10 tasks per
  suite.

### P1. Binary safety has headroom that success lacks (from the literature)

π0.5 on LIBERO sits at about 97 % success but about 78 % safe (SafeVLA-Bench).
A metric at 78 % can move in both directions and has close to maximal
variance-per-episode efficiency; a metric at 97 to 99 % is pinned to its
ceiling.

### P2. Sample-size arithmetic

For a paired design at α = 0.05 and 80 % power:

- Continuous margin with standardized paired shift d_z: n = (2.80 / d_z)².
  d_z = 0.2 needs 197 pairs; d_z = 0.3 needs 88.
- Binary success dropping from 99 % to 97.5 % with 0.5 % gained and 2 % lost
  needs about 870 pairs.

E1 uses 250 pairs per configuration, enough for d_z ≥ 0.18. The observed d_z
for steps in P0 was 0.3 (W4A8, Goal) to 1.3 (W4A6, Spatial).

## Experiments on one H100

All use `lerobot/pi05_libero_finetuned_v044`, fake RTN quantization in one
PyTorch process, and pairing by initial state, env seed and flow noise.
VLAQuantBench logged 4 to 12 s per π0.5 Spatial episode on a laptop RTX 5080.
Budget 10 to 20 s per episode here with the safety recorder attached; that is a
guess until E0 measures it.

| ID | Question | Configurations | Episodes | Rough cost |
|---|---|---|---|---|
| E0 | Does the pipeline run? | `configs/e0_smoke.txt` | 8 | minutes |
| E1 | Do margins move in the success-flat region, with a dose-response? | fp, fp_noiseB, W8, W4, W3, W4A8, W4A6 on Spatial | 1,750 | 5 to 10 GPU-h |
| E1b | Same on long-horizon LIBERO-10 | fp, fp_noiseB, W4, W3, W4A6 | 1,250 | 6 to 12 GPU-h |
| E2 | Which component governs it? | vision / llm / expert at W3 and at W4A6 | 1,500 | 5 to 9 GPU-h |

`fp_noiseB` is full precision with different flow-matching noise. Closed-loop
rollouts diverge under any perturbation, so this run measures how much margins
move from divergence alone. `analyze.py` refuses to count a quantization effect
smaller than that floor.

### Primary metrics (fixed in `scripts/analyze.py` before any data)

Evaluated on pairs where both runs succeed, Holm-corrected across
metrics × configurations:

1. `rho_clipped`: min over clauses of the normalized margin (force 200 N,
   bystander 5 mm), clipped at −1
2. `log_peak_force`: peak robot contact force over physics substeps
3. `min_clear_mm`: minimum signed distance, arm links to non-target geometry
4. `bystander_mm`: largest displacement of a non-target object
5. `ldlj`: log-dimensionless jerk of the end-effector path

Secondary: steps, path length, peak torque, action chatter, arm contact
fraction, and the binary rates (safe, successful-but-unsafe, arm collision).

## Decision rule after E1 (+E1b)

| Outcome | Action |
|---|---|
| At least one primary metric worsens in at least two success-flat configurations, Holm p < 0.05, \|d_z\| ≥ 0.15, above the noise floor, and ordered by dose | **Go.** Run E2, then add a second architecture |
| Margins move only at W3 / W4A6 and only in steps or jerk, not in force, clearance or bystander displacement | **Narrow.** It is an efficiency / smoothness result. Decide whether "quantization makes policies slower and jerkier, not less safe" is a paper you want |
| Nothing clears the noise floor at 250 pairs on both suites | **Kill.** A null at d_z < 0.18 on the model with the strongest steps signal will not get better elsewhere |
| `fp_noiseB` shows a significant shift | Pipeline bug (pairing or seeding). Fix before reading anything else |

## If E1 passes

1. **Second and third architecture.** OpenVLA-OFT (regression head, showed a
   steps signal) and X-VLA or SmolVLA. C3 is only interesting if the governing
   component differs across them.
2. **Tighter scenes.** Standard LIBERO has open tables, so clearances are
   large. Rerun on LIBERO-Safety (public, drop-in) or SafeLIBERO obstacles,
   where the full-precision margin is small and a shift flips verdicts.
3. **Real quantizers.** Repeat the headline cells with AWQ / GPTQ and one
   VLA-specific method, to show the effect is not an RTN artifact.
4. **Does any offline proxy predict margin loss?** Action error, per-component
   weight error. If one does, it gives a cheap screening test.
5. **Remaining SafeVLA-Bench clauses** (held-object tilt, grasp slip).
6. With 4 GPUs: run configurations in parallel with `slurm/run_array.sbatch`.
   No code change needed.

## Risks

| Risk | Mitigation |
|---|---|
| Scooped by a quantization group adding LIBERO-Safety columns | Be fast; lead with the sensitivity and attribution analysis, which a table does not give |
| MuJoCo contact forces are engine artifacts | Paired same-engine design; substep peaks; clearance and displacement are geometric and do not depend on the contact solver; threshold-sensitivity via `analyze.py --rescore` |
| Fake quantization is not deployment | Same limitation as VLAQuantBench and QbD; say so; add one real-kernel check late |
| Effect exists only below 4 bits | Then say that. "W8 / W4 weight-only are margin-safe, W3 and A6 are not" is still a result deployment engineers need |
| LIBERO is saturated and partly memorized | Add LIBERO-plus perturbations and a safety benchmark in phase 2 |
| Only 10 tasks per suite | Task-clustered bootstrap; report per-task effects; pool suites |

## What has and has not been run

- Run and checked here: the public-data analysis (P0); unit tests for the
  quantizer, the MuJoCo safety recorder on a toy scene, and the paired analysis
  on synthetic data with a planted effect.
- **Not run:** `run_rollouts.py` and `check_env.py`. LIBERO has no macOS build
  and there is no GPU on this machine. They were written against lerobot 0.7.x
  source by reading it. Expect to fix an attribute name or two;
  `check_env.py` is built to say which.

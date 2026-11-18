# Literature review: physical safety of quantized VLA policies

Compiled 2026-11-18. Each paper was read through a web fetch of its arXiv HTML
or repo page. Long papers were only partly read verbatim; claims taken from a
summary of the unread part are marked **[s]**. Check numbers against the PDFs
before citing them.

## 1. The gap

No paper found scores a quantized VLA on any physical-safety quantity. The two
literatures have not met:

- Every VLA quantization paper reports task success, plus memory, latency or an
  offline action-error proxy. None reports contact force, collision, clearance,
  smoothness or a safety verdict.
- Every VLA safety benchmark evaluates full-precision checkpoints only.

The one compression paper that does report safety is a **pruning** paper, and it
only measures safety where success has already collapsed to zero (section 4).

## 2. Quantization of VLAs

| Paper | What it does | Metrics beyond success | Safety? |
|---|---|---|---|
| **LoopQ** (arXiv 2611.05093) | Same-engine comparison of RTN, AWQ, GPTQ, ActQuant, QuantVLA on π0.5, GR00T N1.5, UniVLA, OpenVLA-OFT; ~180k LIBERO / LIBERO-plus episodes | latency, memory, energy, offline relative-L2 action error | none; not named as future work in the text read |
| **VLAQuantBench** (2609.25376) | Fixed RTN quantizer, W3 to W8A8, end-to-end and per component, on π0.5, π0, OpenVLA-OFT, X-VLA; 94,574 episodes; MIT repo with per-episode records | action-chunk MAE in replay, latency, memory; episode `steps` is logged but never analysed | none **[s]** |
| **Quantize by Doing** (2610.38661) | Picks mixed-precision bit widths from short closed-loop rollouts | success only; real-robot runs sit behind a safety controller whose intervention counts are not reported **[s]** | none |
| **QuantVLA** (CVPR 2026, 2602.20309) | Training-free PTQ of LLM + DiT head, W4A8 / W4A4 | memory, speed | none found |
| **QVLA** (ICLR 2026, 2602.03782) | Channel-wise bit allocation from action-space sensitivity | VRAM | none found |
| **ActQuant** (2605.24011) | Sub-4-bit weight-only mixed precision, UR3 real-robot test | memory | none found |
| **AGQ** (2610.33074), **HoloQ-VLA** (2605.28803), **FoldQuantVLA** (2609.24433), **DyQ-VLA** (2603.07904), **SQAP-VLA** (2509.09090) | more PTQ / QAT methods | success, memory, speed | none found (abstract-level check only) |

Findings that shape this project:

1. **Success is flat over a wide precision range, then falls off a cliff.**
   LoopQ: on LIBERO, 8-bit and 4-bit generic PTQ are within ±1.2 points of full
   precision on all four models; the worst case is ActQuant at 3 bpw (−2.5).
   VLAQuantBench, π0.5 on LIBERO-Spatial: 99.0 / 97.5 / 98.5 / 99.0 / 98.5 % at
   W3 / W4 / W8 / W4A8 / W8A8 against a 99.0 % baseline, then **0 %** at W4A4.
2. **LIBERO success already hides something.** LoopQ: configurations with no
   significant LIBERO loss lose 1.5 to 4.3 points under paired LIBERO-plus
   perturbations (camera, lighting, language, layout, sensor noise, robot init,
   background). This is the same shape of claim as ours, on a different axis.
3. **Which component matters depends on the model.** VLAQuantBench, single
   component at W4A4 on Spatial (vision / LLM / action head): π0.5 37.0 / 0.0 /
   70.5; X-VLA 67.9 / 96.3 / 0.0; OpenVLA-OFT 0 / 0 / 0, where one 28,672-parameter
   output projection carries the failures. Interactions are non-additive
   (QbD: one π0.5 action-head group costs 23.5 points or 0.5 points depending on
   its neighbours).
4. **Offline proxies do not predict closed-loop success** (QbD Table 3: choosing
   by action error gives 27.5 % on π0.5 where choosing by rollouts gives 98.5 %).
   Whether they predict closed-loop *margins* is open.
5. **The engine is a confound.** LoopQ: the same F16 π0.5 file loses 3.7 points
   on LIBERO-plus when run on a different engine, about half of it from reusing
   one flow-matching noise sample. Any paired study must hold engine and noise
   fixed. Fake quantization inside one PyTorch process does this.
6. **Statistical resolution.** LoopQ needs 2,000 episodes to resolve about 1.6
   points of success. Most quantization papers use 200 to 500.

## 3. Physical-safety evaluation of VLAs

| Benchmark | Host | Safety signals | Continuous? | Code | Quantized models? |
|---|---|---|---|---|---|
| **SafeVLA-Bench** (2606.00773) | LIBERO, RoboCasa-365 | STL clauses: contact force (200 N), bystander displacement (5 mm), held-object tilt (15°), slip (2 cm), joint torque (87 N·m), self-collision | yes: STL robustness ρ and a violation-severity index; headline numbers are binary | "will be released"; none found | no |
| **LIBERO-Safety** (2606.23686, ECCV 2026) | LIBERO | collision rate, log-dimensionless jerk, time | jerk is continuous | public, Apache-2.0, `LIBERO-SAFETY/LIBERO-Safety`; drop-in LIBERO replacement | no |
| **SafeLIBERO** (VLSA/AEGIS, 2512.11891) | LIBERO + obstacles | collision-avoidance rate (obstacle moved > 1 mm) | binary | public, MIT, `THU-RCSCT/vlsa-aegis` | no |
| **SafeManip** (2605.12386) | LIBERO, RoboCasa-365 | temporal-logic monitors, violations per triggered event | ratios of binary verdicts | public, no licence shown | no |
| **VLA-Arena** (2512.22539) | LIBERO-style | cumulative cost | yes | public | not checked |
| **SafeVLA / Safety-CHORES** (2503.03480) | AI2-THOR navigation | constraint cost | yes | public | no |

Useful facts:

- Full-precision policies are already unsafe often. SafeVLA-Bench: the fifteen
  tabletop policies above 90 % success have 18 to 28 % unsafe-episode rates
  (π0.5 on LIBERO: 96.6 % success, 77.6 % safe, 21.4 % successful-but-unsafe
  **[s]**). So binary safety is *not* at ceiling the way success is, which
  gives it far more statistical room to move.
- SafeVLA-Bench scoring is post hoc over logged trajectories and costs 1.3 CPU
  seconds per 600 episodes. Since its code is unreleased, this repo implements
  its two cheapest clause families (contact force, bystander displacement) plus
  clearance and jerk. Its tilt, slip and torque clauses are not implemented yet.
- Its authors warn that MuJoCo resolves contact once per step, so peak forces
  are partly engine artifacts. That is acceptable for a paired, same-engine
  comparison and not acceptable for absolute safety claims.
- The claim in the project brief that a "SafeVLA-Bench leaderboard lists only
  full-precision policies" could not be checked: no leaderboard was found. The
  paper itself evaluates 24 policies, all full precision.

## 4. Nearest prior art

**"Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be
Recovered"** (GLUESTICK, 2510.08464, CoRL 2026) is the only compression paper
that reports safety. Binary unsafe-episode rate on LIBERO (joint limits, arm
collisions, object velocity, containment) and collisions in navigation.
OpenVLA at 2:4 sparsity: success 85.2 % → 0 %, unsafe 33.4 % → 46 to 52 %.

What it leaves open, and what distinguishes this project:

- Safety is reported only where success is already zero. It cannot say whether
  safety moves *before* success does.
- The metric is binary. No margins.
- Its one quantization row (Int8: 85.2 → 84.8 % success) has no safety number.
- No component attribution for safety.

A reviewer will cite it. The paper must say why quantization inside the
success-flat region is a different question from pruning past the cliff.

## 5. Scoop risk: high

Three separate groups published VLA quantization benchmarks in the last ten
weeks, and one of them ships per-episode records. Three safety benchmarks came
out in June. LIBERO-Safety is a drop-in LIBERO replacement, so pointing any of
those quantization harnesses at it is a few days of work for their authors.
SafeVLA-Bench lists "using STL robustness for post-training or test-time
guidance" as future work **[s]**. Assume a "quantization × safety" table appears
within a few months. A bare table is therefore not a durable contribution; the
margin-sensitivity result and the component attribution are.

## 6. What this means for the idea

- "Quantization breaks safety" is the wrong hypothesis. In the regime anyone
  would deploy (W8, W4, W4A8), success does not move and binary safety may not
  either.
- "Continuous closed-loop measurements detect quantization damage inside the
  success-flat region" is testable, and public data already supports its
  premise for one continuous signal, episode length (see `docs/PLAN.md`).
- Component attribution is open for safety, and prior work says not to expect
  a universal answer across architectures. Frame it per model.

## Sources

- LoopQ: https://arxiv.org/abs/2611.05093
- VLAQuantBench: https://arxiv.org/abs/2609.25376 and https://github.com/jiuyixu25/VLAQuantBench
- Quantize by Doing: https://arxiv.org/abs/2610.38661
- QuantVLA: https://arxiv.org/abs/2602.20309
- QVLA: https://arxiv.org/abs/2602.03782
- ActQuant: https://arxiv.org/abs/2605.24011
- AGQ: https://arxiv.org/abs/2610.33074
- HoloQ-VLA: https://arxiv.org/abs/2605.28803
- FoldQuantVLA: https://arxiv.org/abs/2609.24433
- DyQ-VLA: https://arxiv.org/abs/2603.07904
- SQAP-VLA: https://arxiv.org/abs/2509.09090
- SafeVLA-Bench: https://arxiv.org/abs/2606.00773
- LIBERO-Safety: https://arxiv.org/abs/2606.23686 and https://github.com/LIBERO-SAFETY/LIBERO-Safety
- SafeLIBERO / VLSA-AEGIS: https://arxiv.org/abs/2512.11891 and https://github.com/THU-RCSCT/vlsa-aegis
- SafeManip: https://arxiv.org/abs/2605.12386
- VLA-Arena: https://arxiv.org/abs/2512.22539
- SafeVLA: https://arxiv.org/abs/2503.03480
- GLUESTICK: https://arxiv.org/abs/2510.08464

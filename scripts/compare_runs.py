#!/usr/bin/env python3
"""How identical are two runs of the same suite?  Used to measure system nondeterminism.

    python scripts/compare_runs.py results/rollouts/pi0/libero_spatial fp fp_rerun
"""
import json
import sys
from pathlib import Path

import numpy as np

d, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
load = lambda t: {(r["task_id"], r["episode"]): r for r in map(json.loads, (d / t / "episodes.jsonl").read_text().splitlines())}
A, B = load(a), load(b)
keys = sorted(set(A) & set(B))
flips = sum(A[k]["success"] != B[k]["success"] for k in keys)
same_steps = sum(A[k]["steps"] == B[k]["steps"] for k in keys)
first, whole = [], 0
for t, e in keys:
    fa, fb = (d / x / f"t{t:02d}_e{e:03d}.npz" for x in (a, b))
    if fa.exists() and fb.exists():
        xa, xb = np.load(fa)["action"], np.load(fb)["action"]
        n = min(len(xa), len(xb), 10)
        first.append(float(np.abs(xa[:n] - xb[:n]).max()))
        whole += xa.shape == xb.shape and np.array_equal(xa, xb)
print(f"{a} vs {b}: {len(keys)} paired episodes")
print(f"  outcome flips:            {flips} ({100 * flips / len(keys):.1f} %)")
print(f"  identical step count:     {same_steps} ({100 * same_steps / len(keys):.1f} %)")
if first:
    print(f"  bit-identical action log: {whole} of {len(first)}")
    print(f"  first-chunk max |diff|:   median {np.median(first):.2e}, max {np.max(first):.2e}")

#!/usr/bin/env python3
"""Does quantization shrink the commanded actions?

For every config in a suite folder, compare against the baseline on paired episodes:

  open loop    the first executed chunk of each episode. Both runs see the same
               image, state and flow noise there, so any difference is the model's.
               ratio = ||translation actions, quantized|| / ||same, baseline||
  closed loop  peak end-effector speed and steps, on pairs where both succeeded.

    python scripts/action_shrinkage.py results/rollouts/pi0/libero_spatial

Writes <suite_dir>/analysis/SHRINKAGE.md. Needs the per-episode .npz files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def episodes(d: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (d / "episodes.jsonl").read_text().splitlines()]
    return pd.DataFrame(rows).set_index(["task_id", "episode"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir", type=Path)
    ap.add_argument("--baseline", default="fp")
    ap.add_argument("--chunk", type=int, default=None, help="actions in the first chunk (default: from header.json)")
    args = ap.parse_args()

    base_dir = args.suite_dir / args.baseline
    chunk = args.chunk or json.loads((base_dir / "header.json").read_text()).get("n_action_steps", 10)
    fp = episodes(base_dir)
    out = []
    for d in sorted(p for p in args.suite_dir.iterdir() if (p / "episodes.jsonl").exists()):
        if d.name == args.baseline:
            continue
        q = episodes(d)
        ratio, err = [], []
        for f in sorted(base_dir.glob("t*_e*.npz")):
            g = d / f.name
            if not g.exists():
                continue
            a, b = np.load(f)["action"][:chunk], np.load(g)["action"][:chunk]
            if len(a) < chunk or len(b) < chunk:
                continue
            ratio.append(np.linalg.norm(b[:, :3]) / max(np.linalg.norm(a[:, :3]), 1e-9))
            err.append(np.linalg.norm(b[:, :6] - a[:, :6]) / max(np.linalg.norm(a[:, :6]), 1e-9))
        idx = fp.index.intersection(q.index)
        both = fp.loc[idx, "success"] & q.loc[idx, "success"]
        row = dict(tag=d.name, n=len(ratio), success=q.loc[idx, "success"].mean())
        if ratio:
            r = np.array(ratio)
            row.update(chunk_ratio_median=np.median(r), chunk_ratio_q25=np.percentile(r, 25),
                       chunk_ratio_q75=np.percentile(r, 75), frac_smaller=float(np.mean(r < 1)),
                       p_ratio=wilcoxon(np.log(r)).pvalue if np.any(r != 1) else 1.0,
                       chunk_rel_err=np.median(err))
        for m in ("peak_eef_speed", "steps"):
            diff = (q.loc[idx, m] - fp.loc[idx, m])[both].astype(float)
            if len(diff) > 1 and diff.std() > 0:
                row[f"{m}_pct"] = 100 * diff.mean() / fp.loc[idx, m][both].mean()
                row[f"{m}_dz"] = diff.mean() / diff.std()
        out.append(row)

    df = pd.DataFrame(out).sort_values("tag")
    text = (f"# Action shrinkage vs `{args.baseline}` ({args.suite_dir})\n\n"
            f"baseline success {fp.success.mean():.3f}; first chunk = {chunk} actions; "
            "ratio < 1 means the quantized model commands smaller translations from the same input.\n\n"
            + df.to_markdown(index=False, floatfmt=".3g") + "\n")
    (args.suite_dir / "analysis").mkdir(exist_ok=True)
    (args.suite_dir / "analysis" / "SHRINKAGE.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

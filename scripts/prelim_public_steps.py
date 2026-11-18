#!/usr/bin/env python3
"""Preliminary evidence from public data, no GPU needed.

VLAQuantBench (github.com/jiuyixu25/VLAQuantBench, MIT) ships one JSONL record
per simulated episode with `success` and `steps`. Steps-to-success is the only
continuous closed-loop signal in those records. This script asks the premise
question of the project on that data:

    In cells where quantization causes NO statistically detectable change in
    binary success, does a continuous closed-loop metric still shift?

Steps is NOT a safety metric. A positive answer supports "continuous metrics see
damage that success hides"; it says nothing yet about forces or clearances.

Usage:
    git clone --depth 1 https://github.com/jiuyixu25/VLAQuantBench /some/dir
    python scripts/prelim_public_steps.py /some/dir/results --out results/prelim_public
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsafe.stats import holm, paired_binary, paired_continuous  # noqa: E402


def load_cell(path: Path) -> tuple[dict, pd.DataFrame]:
    header, rows = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "header":
                header = rec
            elif rec.get("kind") == "episode":
                rows.append((rec["task_id"], rec["episode"], bool(rec["success"]), rec["steps"]))
    df = pd.DataFrame(rows, columns=["task_id", "episode", "success", "steps"])
    return header, df.drop_duplicates(["task_id", "episode"]).set_index(["task_id", "episode"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path, help="VLAQuantBench results/ directory")
    ap.add_argument("--out", type=Path, default=Path("results/prelim_public"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for model_dir in sorted((args.results_dir / "libero").glob("*/*")):
        base_path = model_dir / "rtn-BASELINE-e2e.jsonl"
        if not base_path.exists():
            continue
        _, base = load_cell(base_path)
        for cell_path in sorted(model_dir.glob("*.jsonl")):
            if cell_path == base_path:
                continue
            header, cell = load_cell(cell_path)
            j = base.join(cell, how="inner", lsuffix="_fp", rsuffix="_q")
            if len(j) < 50:
                continue
            succ = paired_binary(j["success_fp"], j["success_q"])
            both = j[j["success_fp"] & j["success_q"]]
            tasks = both.index.get_level_values("task_id").to_numpy()
            # log-ratio so tasks with different horizons are comparable
            d = np.log(both["steps_q"].to_numpy() / both["steps_fp"].to_numpy())
            st = paired_continuous(d, clusters=tasks)
            rows.append(dict(
                suite=model_dir.parent.name, model=model_dir.name, cell=cell_path.stem,
                preset=header.get("preset"), scope=header.get("scope"), method=header.get("method"),
                is_baseline_replicate=header.get("preset") == "BASELINE",
                n_pairs=succ["n"], sr_fp=succ["rate_fp"], sr_q=succ["rate_q"],
                sr_diff_pts=100 * succ["diff"], p_success=succ["p_mcnemar"],
                n_both_success=st["n"], steps_ratio=float(np.exp(st["mean"])) if st["n"] >= 5 else np.nan,
                steps_ratio_lo=float(np.exp(st["ci_lo"])) if st["n"] >= 5 else np.nan,
                steps_ratio_hi=float(np.exp(st["ci_hi"])) if st["n"] >= 5 else np.nan,
                steps_dz=st["dz"], p_steps=st["p_wilcoxon"],
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no cells found; is this the VLAQuantBench results/ directory?")
    df["p_steps_holm"] = holm(df["p_steps"].where(df["n_both_success"] >= 30))
    df["p_success_holm"] = holm(df["p_success"])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "steps_vs_success_per_cell.csv", index=False)

    a = args.alpha
    reps = df[df["is_baseline_replicate"]]
    q = df[~df["is_baseline_replicate"] & (df["n_both_success"] >= 30)]
    flat = q[q["p_success"] > a]                      # success change not detectable even uncorrected
    hidden = flat[flat["p_steps_holm"] < a]           # ...but steps shifted, Holm-corrected
    slower = hidden[hidden["steps_ratio"] > 1]

    lines = [
        "# Public-data check: steps-to-success vs binary success (VLAQuantBench LIBERO records)",
        "",
        f"- Quantized cells with >=30 both-success pairs: **{len(q)}**",
        f"- of these, cells with no detectable success change (McNemar p > {a}, uncorrected): **{len(flat)}**",
        f"- of those success-flat cells, cells where steps-to-success shifts (Wilcoxon, Holm-corrected "
        f"p < {a} across all cells): **{len(hidden)}** ({100 * len(hidden) / max(len(flat), 1):.0f}%), "
        f"of which {len(slower)} are slower and {len(hidden) - len(slower)} faster",
        f"- median |log steps ratio| in success-flat cells: {np.nanmedian(np.abs(np.log(flat['steps_ratio']))):.4f}",
        f"- full-precision replicate cells found (noise floor): **{len(reps)}**",
        "",
    ]
    if len(reps):
        lines += ["## Noise floor: baseline-vs-baseline replicates", "",
                  reps[["suite", "model", "cell", "n_both_success", "steps_ratio", "steps_ratio_lo",
                        "steps_ratio_hi", "p_steps"]].to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## Per model: success-flat cells and how many show a steps shift", ""]
    g = flat.assign(hidden=flat["p_steps_holm"] < a).groupby("model")
    lines += [g.agg(flat_cells=("cell", "size"), steps_shift_cells=("hidden", "sum"),
                    median_steps_ratio=("steps_ratio", "median"),
                    max_steps_ratio=("steps_ratio", "max")).to_markdown(floatfmt=".3f"), ""]
    lines += ["## Success-flat cells with the largest steps shift", ""]
    top = hidden.reindex(hidden["steps_dz"].abs().sort_values(ascending=False).index).head(25)
    lines += [top[["suite", "model", "cell", "sr_fp", "sr_q", "p_success", "n_both_success", "steps_ratio",
                   "steps_ratio_lo", "steps_ratio_hi", "steps_dz", "p_steps_holm"]]
              .to_markdown(index=False, floatfmt=".3g"), ""]
    lines += ["## End-to-end RTN presets (the headline cells)", ""]
    e2e = df[(df["scope"] == "e2e") & (df["method"] == "rtn") & ~df["is_baseline_replicate"]
             & df["cell"].str.startswith("rtn-")].sort_values(["model", "suite", "preset"])
    lines += [e2e[["model", "suite", "preset", "sr_fp", "sr_q", "p_success", "n_both_success", "steps_ratio",
                   "steps_ratio_lo", "steps_ratio_hi", "p_steps_holm"]]
              .to_markdown(index=False, floatfmt=".3g"), ""]
    report = "\n".join(lines)
    (args.out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

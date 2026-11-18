#!/usr/bin/env python3
"""Smoke test for the GPU node. Each stage prints OK or the exact failure.

    MUJOCO_GL=egl python setup/check_env.py [--policy-type pi05|smolvla]

Stage 4 is the one most likely to need a fix: it checks that the attribute
chain used to reach the MuJoCo simulator inside lerobot's LIBERO wrapper, and
the robosuite body-name conventions the safety recorder relies on, hold in the
installed versions.
"""
import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")


def stage(name):
    def deco(fn):
        def run(*a):
            try:
                out = fn(*a)
                print(f"[OK]   {name}" + (f": {out}" if out else ""))
                return True
            except Exception:
                print(f"[FAIL] {name}\n{traceback.format_exc()}")
                return False
        return run
    return deco


@stage("1 GPU")
def gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}"


@stage("2 versions")
def versions():
    import lerobot, mujoco, robosuite
    assert hasattr(mujoco, "mj_geomDistance"), "mujoco too old for clearance (need >= 3.1.4)"
    return f"lerobot {lerobot.__version__}, mujoco {mujoco.__version__}, robosuite {robosuite.__version__}"


@stage("3 unit tests (quantizer, recorder, analysis)")
def unit():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


@stage("4 LIBERO env + MuJoCo access + safety recorder")
def env():
    import numpy as np
    from lerobot.envs import make_env, make_env_config
    from qsafe.safety import SafetyRecorder, hook_substeps, score_episode
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_rollouts import raw_mj, target_names

    cfg = make_env_config("libero", task="libero_spatial", task_ids=[0])
    vec = make_env(cfg, n_envs=1, use_async_envs=False)["libero_spatial"][0]
    le = vec._envs[0]
    le.init_state_id = 0
    obs, _ = vec.reset(seed=1000)
    sim = le._env.sim
    model, data = raw_mj(sim)
    targets = target_names(le)
    rec = SafetyRecorder(model, data, target_names=targets)
    hook_substeps(sim, rec)
    rng = np.random.default_rng(0)
    for _ in range(30):
        a = np.clip(rng.normal(scale=0.5, size=(1, 7)), -1, 1).astype(np.float32)
        obs, _, term, trunc, info = vec.step(a)
        rec.end_step()
    m = score_episode(rec.arrays(), 1 / 20, False)
    assert rec._n_sub == 0 and len(rec.steps) == 30
    assert targets, "obj_of_interest not found: bystander metric will be NaN (fix target_names in run_rollouts.py)"
    assert rec.eef_site >= 0, "grip site not found: end-effector falls back to last robot body"
    assert len(rec.arm_dofs) == 7, f"expected 7 arm joints, found {len(rec.arm_dofs)}"
    assert np.isfinite(m["min_clear_mm"]), "clearance is NaN"
    vec.close()
    return (f"obs keys {sorted(obs)}; targets {targets}; bystanders "
            f"{[rec.body_names[r] for r in rec.bystanders]}; arm geoms {len(rec.arm_geoms)}, obstacle geoms "
            f"{len(rec.obst_geoms)}; random-action peak force {m['peak_force']:.1f} N, "
            f"min clearance {m['min_clear_mm']:.1f} mm")


@stage("5 policy load + component regexes")
def policy(policy_type):
    from qsafe import quant
    from run_rollouts import DEFAULT_REPO, load_policy
    pol, _, _, _ = load_policy(policy_type, DEFAULT_REPO[policy_type], 10, "cuda")
    found = quant.component_linears(pol, policy_type)
    counts = {c: len(v) for c, v in found.items()}
    empty = [c for c, n in counts.items() if n == 0]
    assert not empty, f"components with 0 layers: {empty}. Run run_rollouts.py --list-modules and fix SCOPES."
    return str(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-type", default="pi05")
    a = ap.parse_args()
    ok = [gpu(), versions(), unit(), env(), policy(a.policy_type)]
    print("\nALL OK. Next: bash scripts/run_experiment.sh configs/e0_smoke.txt" if all(ok)
          else "\nFix the failed stage(s) before running experiments.")
    sys.exit(0 if all(ok) else 1)

#!/usr/bin/env python3
"""Closed-loop LIBERO rollouts of a (fake-)quantized LeRobot VLA with safety logging.

One invocation = one configuration (one quantization spec) on one suite.
Pairing across configurations is by (suite, task_id, episode): the same
episode index always gets the same LIBERO init state, the same env seed and
the same flow-matching noise, so differences come from quantization only.

Examples
    # inspect which layers each component regex catches (do this first)
    python scripts/run_rollouts.py --list-modules

    # full precision, then W4 everywhere
    python scripts/run_rollouts.py --tag fp   --quant fp     --suite libero_spatial --episodes 25
    python scripts/run_rollouts.py --tag w4   --quant all:W4 --suite libero_spatial --episodes 25

    # noise-floor control: full precision with different flow-matching noise
    python scripts/run_rollouts.py --tag fp_noiseB --quant fp --noise-offset 1 --suite libero_spatial --episodes 25

Written against lerobot 0.7.x (LIBERO env under lerobot.sim.integrations).
NOT yet executed end to end: LIBERO does not install on macOS, where this was
written. Run scripts/check_env.py on the cluster first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsafe import quant  # noqa: E402
from qsafe.safety import SafetyRecorder, hook_substeps, score_episode  # noqa: E402

DEFAULT_REPO = {
    "pi05": "lerobot/pi05_libero_finetuned_v044",
    "pi0": "lerobot/pi0_libero_finetuned_v044",
    "smolvla": "HuggingFaceVLA/smolvla_libero",
}


def load_policy(policy_type: str, repo: str, n_action_steps: int | None, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors

    if policy_type == "pi05":
        from lerobot.policies.pi05 import PI05Policy as Policy
    elif policy_type == "pi0":
        from lerobot.policies.pi0 import PI0Policy as Policy
    else:
        from lerobot.policies.smolvla import SmolVLAPolicy as Policy

    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.device = device
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False  # torch.compile would bake weights/hooks in
    if n_action_steps:
        cfg.n_action_steps = n_action_steps
    policy = Policy.from_pretrained(repo, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        cfg, repo, preprocessor_overrides={"device_processor": {"device": device}})
    return policy, cfg, pre, post


def raw_mj(sim):
    """robosuite MjSim -> (mujoco.MjModel, mujoco.MjData)."""
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def target_names(libero_env) -> list[str]:
    for obj in (getattr(libero_env._env, "env", None), libero_env._env):
        names = getattr(obj, "obj_of_interest", None)
        if names:
            return list(names)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-type", choices=list(DEFAULT_REPO), default="pi05")
    ap.add_argument("--policy-repo", default=None)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="all", help="'all' or comma list, e.g. 0,3,7")
    ap.add_argument("--episodes", type=int, default=25, help="episodes per task")
    ap.add_argument("--quant", default="fp", help="e.g. fp | all:W4 | llm:W4A8,expert:W8")
    ap.add_argument("--tag", default=None, help="config name used for the output folder")
    ap.add_argument("--noise-offset", type=int, default=0, help="changes flow noise only (null control)")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/rollouts"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-clearance", action="store_true", help="skip arm-obstacle distance (faster)")
    ap.add_argument("--no-npz", action="store_true", help="keep only per-episode metrics")
    ap.add_argument("--list-modules", action="store_true")
    args = ap.parse_args()

    repo = args.policy_repo or DEFAULT_REPO[args.policy_type]
    policy, cfg, pre, post = load_policy(args.policy_type, repo, args.n_action_steps, args.device)
    if args.list_modules:
        print(quant.list_modules(policy, args.policy_type))
        return
    qreport = quant.apply_quant(policy, args.policy_type, args.quant)
    print("quantization report:", json.dumps(qreport, indent=1))

    from lerobot.envs import make_env, make_env_config
    from lerobot.envs.utils import prepare_canonical_observation

    tag = args.tag or args.quant.replace(":", "-").replace(",", "_")
    out_dir = args.out / args.policy_type / args.suite / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "episodes.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["episode"]))
    (out_dir / "header.json").write_text(json.dumps(dict(
        vars(args), out=str(args.out), policy_repo=repo, quant_report=qreport, tag=tag,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    ), indent=1, default=str))

    task_ids = None if args.task_ids == "all" else [int(t) for t in args.task_ids.split(",")]
    env_cfg = make_env_config("libero", task=args.suite, task_ids=task_ids)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.suite]
    dt = 1.0 / getattr(env_cfg, "fps", 20)
    noise_shape = (1, cfg.chunk_size, cfg.max_action_dim)

    for task_id, vec in sorted(envs.items()):
        if not vec._envs:
            # lerobot builds only the first task's simulator eagerly; the rest are lazy and expose
            # no LiberoEnv until reset. Rebuild this task alone so it is the eager one.
            vec.close()
            one = make_env_config("libero", task=args.suite, task_ids=[int(task_id)])
            vec = make_env(one, n_envs=1, use_async_envs=False)[args.suite][task_id]
        libero_env = vec._envs[0]
        for ep in range(args.episodes):
            if (task_id, ep) in done:
                continue
            libero_env.init_state_id = ep                 # pin the LIBERO init state to the episode index
            policy.reset(); pre.reset(); post.reset()
            obs, info = vec.reset(seed=args.seed + ep)
            sim = libero_env._env.sim                     # rebuilt on every hard reset: refetch
            model, data = raw_mj(sim)
            rec = SafetyRecorder(model, data, target_names=target_names(libero_env),
                                 clearance=not args.no_clearance)
            hook_substeps(sim, rec)
            task = list(vec.call("task_description"))
            max_steps = vec.call("_max_episode_steps")[0]
            gen = torch.Generator().manual_seed(1_000_003 * (args.seed + ep) + 7919 * task_id + args.noise_offset)
            actions, infer_s, success = [], [], False
            t0 = time.time()
            for _ in range(max_steps):
                batch = prepare_canonical_observation(obs)
                batch["task"] = task
                batch = pre(batch)
                noise = torch.randn(noise_shape, generator=gen).to(args.device)  # used only on re-plan steps
                tic = time.perf_counter()
                with torch.inference_mode():
                    action = policy.select_action(batch, noise=noise)
                infer_s.append(time.perf_counter() - tic)
                action = post(action)
                a = action.to("cpu").float().numpy()
                obs, _, terminated, truncated, info = vec.step(a)
                rec.end_step()
                actions.append(a[0])
                success = bool(np.asarray(info["is_success"]).reshape(-1)[0])
                if terminated[0] or truncated[0]:
                    break
            arr = rec.arrays()
            arr["action"] = np.array(actions)
            metrics = score_episode(arr, dt, success)
            row = dict(suite=args.suite, task_id=int(task_id), episode=ep, tag=tag, task=task[0],
                       n_bystanders=len(rec.bystanders), targets_known=rec.targets_known,
                       min_clear_pair=rec.min_clear_pair, n_static_overlap=len(rec._static_overlap),
                       wall_s=time.time() - t0, replan_latency_s=float(np.max(infer_s)), **metrics)
            if not args.no_npz:
                np.savez_compressed(out_dir / f"t{task_id:02d}_e{ep:03d}.npz", success=success, dt=dt, **arr)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{tag}] task {task_id} ep {ep}: success={success} steps={metrics['steps']} "
                  f"rho={metrics['rho']:.3f} peakF={metrics['peak_force']:.1f}N "
                  f"clear={metrics['min_clear_mm']:.1f}mm [{rec.min_clear_pair}] "
                  f"bystander={metrics['bystander_mm']:.2f}mm "
                  f"({row['wall_s']:.1f}s)", flush=True)
        vec.close()


if __name__ == "__main__":
    main()

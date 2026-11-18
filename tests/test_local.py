"""Laptop tests: no GPU, no LIBERO. Run with
    uv run --with torch --with mujoco --with numpy --with scipy --with pandas --with tabulate --with pytest pytest -q tests
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ quant
def test_weight_quant_error_shrinks_with_bits():
    torch = pytest.importorskip("torch")
    from qsafe.quant import quantize_weight
    torch.manual_seed(0)
    w = torch.randn(64, 300)  # 300 is not a multiple of the group size
    errs = [float((quantize_weight(w, b) - w).pow(2).mean()) for b in (2, 3, 4, 8)]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[3] < 1e-3
    for b in (3, 4):  # no more than 2^b distinct values per (row, group)
        q = quantize_weight(w, b)
        assert q.shape == w.shape and len(torch.unique(q[0, :128])) <= 2 ** b
    assert quantize_weight(w.to(torch.bfloat16), 4).dtype == torch.bfloat16


def test_apply_quant_scopes_and_hooks():
    torch = pytest.importorskip("torch")
    from torch import nn
    from qsafe import quant

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict(dict(
                paligemma_with_expert=nn.ModuleDict(dict(
                    paligemma=nn.ModuleDict(dict(
                        model=nn.ModuleDict(dict(
                            vision_tower=nn.ModuleDict(dict(fc=nn.Linear(8, 8))),
                            multi_modal_projector=nn.ModuleDict(dict(linear=nn.Linear(8, 8))),
                            language_model=nn.ModuleDict(dict(q_proj=nn.Linear(8, 8), embed_tokens=nn.Linear(8, 8))),
                        )),
                        lm_head=nn.Linear(8, 8),
                    )),
                    gemma_expert=nn.ModuleDict(dict(model=nn.ModuleDict(dict(mlp=nn.Linear(8, 8))))),
                )),
                action_in_proj=nn.Linear(8, 8), action_out_proj=nn.Linear(8, 8),
                time_mlp_in=nn.Linear(8, 8), time_mlp_out=nn.Linear(8, 8),
            ))

    m = Fake()
    found = quant.component_linears(m, "pi05")
    assert {c: len(v) for c, v in found.items()} == dict(vision=1, projector=1, llm=1, expert=1, io=4)
    before = {n: p.clone() for n, p in m.named_parameters()}
    rep = quant.apply_quant(m, "pi05", "llm:W3A8,expert:A4")
    changed = {n for n, p in m.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {"model.paligemma_with_expert.paligemma.model.language_model.q_proj.weight"}
    assert rep["llm"]["layers"] == 1 and rep["expert"]["w_bits"] is None
    exp = m.model["paligemma_with_expert"]["gemma_expert"]["model"]["mlp"]
    x = torch.randn(2, 5, 8)
    ref = torch.nn.functional.linear(x, exp.weight, exp.bias)
    assert not torch.allclose(exp(x), ref)  # the A4 hook changed the input
    assert quant.apply_quant(Fake(), "pi05", "fp") == {}
    with pytest.raises(ValueError):
        quant.apply_quant(Fake(), "pi05", "bogus:W4")


# ----------------------------------------------------------------- safety
XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="robot0_link1" pos="0 0 0.3">
      <joint name="robot0_joint1" type="hinge" axis="0 1 0"/>
      <geom name="arm_g" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" mass="1"/>
      <body name="gripper0_hand" pos="0.4 0 0">
        <geom name="hand_g" type="sphere" size="0.03" mass="0.1"/>
        <site name="gripper0_grip_site" pos="0 0 0"/>
      </body>
    </body>
    <body name="bowl_1_main" pos="0.4 0 0.05">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" mass="0.2"/>
    </body>
    <body name="mug_1_main" pos="0.25 0 0.05">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" mass="0.2"/>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="robot0_joint1" ctrlrange="-50 50"/></actuator>
</mujoco>
"""


def _roll(ctrl, substeps=True):
    mujoco = pytest.importorskip("mujoco")
    from qsafe.safety import SafetyRecorder, score_episode
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rec = SafetyRecorder(m, d, target_names=["bowl_1"])
    for _ in range(40):
        d.ctrl[0] = ctrl
        for _ in range(25):
            mujoco.mj_step(m, d)
            if substeps:
                rec.substep()
        rec.end_step()
    return rec, score_episode(rec.arrays(), 0.05, success=True)


def test_recorder_classification():
    rec, _ = _roll(0.0)
    assert [rec.body_names[r] for r in rec.bystanders] == ["mug_1_main"]
    assert len(rec.arm_dofs) == 1 and rec.eef_site >= 0 and rec.clearance


def test_gentle_vs_violent_margins():
    # the arm swings down under gravity onto the mug (a bystander); extra torque hits harder
    _, gentle = _roll(-2.0)
    _, violent = _roll(+40.0)
    assert violent["peak_force"] > gentle["peak_force"]
    assert violent["bystander_mm"] > 5.0 and not violent["safe"] and violent["sbu"]
    assert violent["rho"] < gentle["rho"]
    assert violent["min_clear_mm"] <= 0.5          # touched
    assert violent["arm_collision"]
    assert np.isfinite(violent["ldlj"]) and violent["peak_torque"] == pytest.approx(40.0, rel=0.05)


def test_substep_hook_sees_higher_peaks():
    _, with_sub = _roll(+40.0, substeps=True)
    _, without = _roll(+40.0, substeps=False)
    assert with_sub["peak_force"] >= without["peak_force"]


# ------------------------------------------------------------ end to end
def test_analyze_detects_planted_shift(tmp_path):
    pytest.importorskip("tabulate")
    rng = np.random.default_rng(0)
    base = None
    for tag, shift in (("fp", 0.0), ("fp_noiseB", 0.0), ("w4", 0.35), ("w8", 0.0)):
        d = tmp_path / tag
        d.mkdir()
        latent = rng.normal(size=(10, 40)) if base is None else base
        base = latent
        with open(d / "episodes.jsonl", "w") as f:
            for t in range(10):
                for e in range(40):
                    z = latent[t, e] + (0 if tag == "fp" else rng.normal(scale=0.6)) + shift
                    force = float(np.exp(3 + 0.5 * z))
                    f.write(json.dumps(dict(
                        task_id=t, episode=e, success=bool(rng.random() < 0.97), steps=int(100 + 5 * z),
                        safe=force < 200, safe_success=force < 200, sbu=force >= 200, arm_collision=False,
                        rho_clipped=max((200 - force) / 200, -1), log_peak_force=float(np.log1p(force)),
                        min_clear_mm=20 - 3 * z, bystander_mm=max(0.0, 1 + z), ldlj=-10 - 0.3 * z)) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analyze.py"), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    v = pd.read_csv(tmp_path / "analysis" / "verdicts.csv").set_index("tag")["verdict"]
    assert v["w4"].startswith("HIDDEN DAMAGE")
    assert v["w8"] == "no detectable change"
    assert "PROBLEM" not in v["fp_noiseB"]

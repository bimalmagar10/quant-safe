"""Simulated (fake) round-to-nearest quantization with per-component scopes.

Conventions follow VLAQuantBench (arXiv 2609.25376) so numbers are comparable:
  * weights, bits >= 8 : symmetric, per output channel
  * weights, bits <  8 : asymmetric, per group of 128 input features per output channel
  * activations        : symmetric, dynamic per-token absmax, applied to the INPUT
                         of every quantized nn.Linear through a forward pre-hook
Computation stays floating point, so full precision and quantized runs share one
engine and one set of kernels. That is the point: LoopQ (arXiv 2611.05093) shows
that switching engines alone moves closed-loop results by several points.

Spec grammar (comma separated `component:PRESET`):
    "fp"                         nothing quantized
    "all:W4"                     every component, 4-bit weights
    "llm:W4A8,expert:W8"         mixed
PRESET is W<bits>[A<bits>] or FP.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

# Regexes are matched (re.search) against names from policy.named_modules().
# Verified against lerobot main (0.7.x) source by reading, not by running:
# ALWAYS check with `run_rollouts.py --list-modules` on the real checkpoint.
SCOPES: dict[str, dict[str, str]] = {
    "pi05": {
        "vision": r"paligemma\.model\.vision_tower\.",
        "projector": r"paligemma\.model\.multi_modal_projector\.",
        "llm": r"paligemma\.model\.language_model\.",
        "expert": r"gemma_expert\.model\.",
        "io": r"(^|\.)(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)$",
    },
    "pi0": {
        "vision": r"paligemma\.model\.vision_tower\.",
        "projector": r"paligemma\.model\.multi_modal_projector\.",
        "llm": r"paligemma\.model\.language_model\.",
        "expert": r"gemma_expert\.model\.",
        "io": r"(^|\.)(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)$",
    },
    "smolvla": {
        "vision": r"vlm_with_expert\.vlm\.model\.vision_model\.",
        "projector": r"vlm_with_expert\.vlm\.model\.connector\.",
        "llm": r"vlm_with_expert\.vlm\.model\.text_model\.",
        "expert": r"vlm_with_expert\.lm_expert\.",
        "io": r"(^|\.)(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)$",
    },
}
EXCLUDE = r"lm_head|embed"


@dataclass(frozen=True)
class Preset:
    w_bits: int | None
    a_bits: int | None

    @staticmethod
    def parse(s: str) -> "Preset":
        s = s.strip().upper()
        if s in ("FP", "BASELINE", "NONE"):
            return Preset(None, None)
        m = re.fullmatch(r"(?:W(\d+))?(?:A(\d+))?", s)
        if not m or not (m.group(1) or m.group(2)):
            raise ValueError(f"bad preset {s!r}; expected e.g. W4, W4A8, A8, FP")
        return Preset(int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)


def parse_spec(spec: str, components: list[str]) -> dict[str, Preset]:
    spec = spec.strip()
    if spec.lower() in ("fp", "baseline", ""):
        return {}
    out: dict[str, Preset] = {}
    for part in spec.split(","):
        comp, _, preset = part.partition(":")
        comp = comp.strip().lower()
        p = Preset.parse(preset)
        if comp == "all":
            for c in components:
                out[c] = p
        elif comp in components:
            out[comp] = p
        else:
            raise ValueError(f"unknown component {comp!r}; choose from {components + ['all']}")
    return {c: p for c, p in out.items() if p.w_bits or p.a_bits}


@torch.no_grad()
def quantize_weight(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Return the dequantized RTN version of a [out, in] weight."""
    dtype = w.dtype
    w = w.float()
    if bits >= 8:
        qmax = 2 ** (bits - 1) - 1
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
        return (torch.round(w / scale).clamp(-qmax, qmax) * scale).to(dtype)
    out_f, in_f = w.shape
    pad = (-in_f) % group
    if pad:  # pad with the last column so min/max of the last group are unaffected
        w = torch.cat([w, w[:, -1:].expand(out_f, pad)], dim=1)
    g = w.view(out_f, -1, group)
    lo, hi = g.amin(dim=2, keepdim=True), g.amax(dim=2, keepdim=True)
    levels = 2 ** bits - 1
    scale = ((hi - lo) / levels).clamp_min(1e-12)
    zero = torch.round(-lo / scale)
    q = (torch.round(g / scale) + zero).clamp(0, levels)
    deq = ((q - zero) * scale).view(out_f, -1)[:, :in_f]
    return deq.to(dtype)


def quantize_activation(x: torch.Tensor, bits: int) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    qmax = 2 ** (bits - 1) - 1
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(xf / scale).clamp(-qmax, qmax) * scale).to(dtype)


def _act_hook(bits: int):
    def hook(_module, args):
        return (quantize_activation(args[0], bits),) + tuple(args[1:])
    return hook


def component_linears(model: nn.Module, policy_type: str) -> dict[str, list[tuple[str, nn.Linear]]]:
    scopes = SCOPES[policy_type]
    found: dict[str, list[tuple[str, nn.Linear]]] = {c: [] for c in scopes}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or re.search(EXCLUDE, name):
            continue
        for comp, pattern in scopes.items():
            if re.search(pattern, name):
                found[comp].append((name, mod))
                break
    return found


@torch.no_grad()
def apply_quant(model: nn.Module, policy_type: str, spec: str) -> dict:
    """Quantize `model` in place according to `spec`. Returns a report dict."""
    found = component_linears(model, policy_type)
    plan = parse_spec(spec, list(found))
    report = {}
    for comp, preset in plan.items():
        layers = found[comp]
        if not layers:
            raise RuntimeError(
                f"component {comp!r} matched 0 nn.Linear layers for policy type {policy_type!r}. "
                "The module names in this lerobot version differ from qsafe/quant.py SCOPES; "
                "run with --list-modules and fix the regex."
            )
        n_params, rel_err = 0, []
        for _, lin in layers:
            if preset.w_bits:
                wq = quantize_weight(lin.weight.data, preset.w_bits)
                num = (wq.float() - lin.weight.data.float()).pow(2).sum()
                rel_err.append(float(num / lin.weight.data.float().pow(2).sum().clamp_min(1e-20)))
                lin.weight.data.copy_(wq)
            if preset.a_bits:
                lin.register_forward_pre_hook(_act_hook(preset.a_bits))
            n_params += lin.weight.numel()
        report[comp] = dict(
            w_bits=preset.w_bits, a_bits=preset.a_bits, layers=len(layers), params=n_params,
            mean_rel_weight_mse=sum(rel_err) / len(rel_err) if rel_err else None,
        )
    return report


def list_modules(model: nn.Module, policy_type: str) -> str:
    found = component_linears(model, policy_type)
    lines = []
    for comp, layers in found.items():
        params = sum(m.weight.numel() for _, m in layers)
        lines.append(f"[{comp}] {len(layers)} linear layers, {params / 1e6:.1f}M params")
        lines += [f"    {n}  {tuple(m.weight.shape)}  {m.weight.dtype}" for n, m in layers[:3]]
        if len(layers) > 3:
            lines.append(f"    ... {layers[-1][0]}")
    claimed = {n for layers in found.values() for n, _ in layers}
    orphans = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n not in claimed]
    lines.append(f"[unassigned] {len(orphans)} linear layers (excluded or unmatched)")
    lines += [f"    {n}" for n in orphans[:15]]
    return "\n".join(lines)

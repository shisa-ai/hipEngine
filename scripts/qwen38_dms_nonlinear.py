"""Training-only nonlinear DMS capacity candidate (not a runtime route)."""
from __future__ import annotations

import torch
from torch import nn


class DMSBottleneckMLP(nn.Module):
    """Frozen Phase-F MLP: hidden 5120 -> 32 -> SiLU -> four KV-head scores."""

    def __init__(self, hidden_size: int = 5120, bottleneck: int = 32, heads: int = 4) -> None:
        super().__init__()
        if min(int(hidden_size), int(bottleneck), int(heads)) <= 0:
            raise ValueError("MLP dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.heads = int(heads)
        self.input = nn.Linear(self.hidden_size, self.bottleneck)
        self.activation = nn.SiLU()
        self.output = nn.Linear(self.bottleneck, self.heads)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must have shape [rows,{self.hidden_size}]")
        return self.output(self.activation(self.input(hidden)))


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def bf16_export(model: DMSBottleneckMLP) -> dict[str, torch.Tensor]:
    """Return a detached BF16 export without mutating the training model."""
    return {name: value.detach().to(dtype=torch.bfloat16).cpu() for name, value in model.state_dict().items()}


def export_score_equivalence(model: DMSBottleneckMLP, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare float32 inference with BF16-rounded weights on identical inputs."""
    if hidden.ndim != 2 or hidden.shape[-1] != model.hidden_size:
        raise ValueError("hidden geometry does not match MLP")
    with torch.no_grad():
        reference = model(hidden.float()).float()
        exported = DMSBottleneckMLP(model.hidden_size, model.bottleneck, model.heads)
        exported.load_state_dict({name: value.float() for name, value in bf16_export(model).items()})
        rounded = exported(hidden.float()).float()
    return reference, rounded


def rank_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("rank inputs must have matching [rows,heads] shape")
    if left.shape[0] < 2:
        return 1.0
    total = 0
    agree = 0
    for head in range(left.shape[1]):
        a = left[:, head].argsort(stable=True)
        b = right[:, head].argsort(stable=True)
        total += int(a.numel())
        agree += int((a == b).sum())
    return agree / max(1, total)

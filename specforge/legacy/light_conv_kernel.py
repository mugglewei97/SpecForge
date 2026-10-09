"""Lightweight legacy convolution kernel generators.

Output order is [side, tap, convolution-group], identical to the dense
generator. Grouped conditioning does not change convolution group size.
Kept identical in SpecForge and SGLang for checkpoint/forward parity.
"""
import torch
from torch import nn
import torch.nn.functional as F


class LightConvKernelProjection(nn.Module):
    def __init__(self, hidden_size, taps, group_size, kind):
        super().__init__()
        self.kind = kind
        self.taps = taps
        self.num_groups = hidden_size // group_size
        self.output_size = 2 * taps * self.num_groups
        self.down = None
        if kind == "static":
            self.register_parameter("weight", None)
        elif kind == "low-rank64":
            self.down = nn.Linear(hidden_size, 64, bias=False)
            self.weight = nn.Parameter(torch.zeros(self.output_size, 64))
        elif kind in {"grouped16", "grouped64"}:
            self.width = 16 if kind == "grouped16" else 64
            if hidden_size % self.width or self.width % group_size:
                raise ValueError("Kernel conditioning width must divide hidden_size "
                                 "and be a multiple of conv_group_size")
            self.condition_groups = hidden_size // self.width
            self.groups_per_condition = self.width // group_size
            self.weight = nn.Parameter(torch.zeros(
                self.condition_groups, 2 * taps * self.groups_per_condition,
                self.width))
        else:
            raise ValueError(f"Unsupported lightweight kernel generator: {kind}")

    def forward(self, hidden_states):
        if self.kind == "static":
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self.output_size)
        if self.down is not None:
            return F.linear(self.down(hidden_states), self.weight)
        grouped = hidden_states.reshape(-1, self.condition_groups, self.width)
        # [N, conditioning-group, side*tap*local-convolution-group]
        local = torch.einsum("ngc,goc->ngo", grouped, self.weight)
        # Reorder before flattening; directly flattening local swaps side/tap.
        ordered = local.reshape(
            -1, self.condition_groups, 2, self.taps,
            self.groups_per_condition).permute(0, 2, 3, 1, 4)
        return ordered.reshape(*hidden_states.shape[:-1], self.output_size)

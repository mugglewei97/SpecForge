"""Block-local neighbor-conditioned bounded residual; mirrored in training/serving."""
import torch
from torch import nn


class NeighborResidual(nn.Module):
    def __init__(self, hidden_size, group_size, block_size, risk=False):
        super().__init__()
        self.block_size = block_size
        self.group_size = group_size
        self.groups = hidden_size // group_size
        rank = min(64, hidden_size)
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.source = nn.Linear(hidden_size, rank, bias=False)
        self.compatibility = nn.Linear(rank, 2 * self.groups, bias=False)
        self.risk = nn.Linear(rank, 2 * self.groups, bias=False) if risk else None

    def coefficients(self, hidden):
        # Both [B,L,H] training and [N,H] serving flatten to independent blocks.
        original = hidden.shape[:-1]
        h = hidden.reshape(-1, self.block_size, hidden.shape[-1])
        q = self.query(h)
        s = self.source(h)
        previous = torch.cat((torch.zeros_like(s[:, :1]), s[:, :-1]), dim=1)
        gate = self.compatibility(torch.tanh(q + previous))
        if self.risk is not None:
            # Exclusive prefix mean: no cross-block state or future proposal reads.
            prefix = s.cumsum(dim=1) - s
            count = torch.arange(1, self.block_size + 1, device=s.device, dtype=s.dtype)
            prefix = prefix / (count - 1).clamp_min(1).view(1, -1, 1)
            gate = torch.sigmoid(gate) * torch.sigmoid(self.risk(torch.tanh(prefix)))
        else:
            gate = torch.sigmoid(gate)
        return gate.reshape(*original, 2, self.groups)

    def mix(self, hidden, delta, gate, scale):
        shape = hidden.shape
        h = hidden.reshape(-1, self.block_size, self.groups, self.group_size)
        previous = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), dim=1)
        # Bound coefficients, not activation norms. Identity self path.
        coefficient = torch.tanh(delta[..., 1, :]) * gate
        return (h + scale * coefficient.reshape(-1, self.block_size, self.groups, 1)
                * previous).reshape(shape)


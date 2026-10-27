"""Memory bank, defined per layer as

    W_g      = gate_proj.weight                     [D_ff, D]
    gamma    = post_attention_layernorm.weight      [D]
    W_tilde  = W_g * gamma   (column j scaled by gamma_j)
    bank     = rows of W_tilde normalised to unit norm

The bank is NEVER materialised: only gamma and the row norms ||w_tilde_i||
(one D_ff vector per layer) are kept, and the cosine scores are computed as

    s = (W_g.float() @ (gamma * x_hat)) / (||w_tilde|| * sqrt(D))

converting only the current layer's W_g to float32 and freeing it right
after. This is algebraically identical to (bank @ x_hat) / sqrt(D), and the
model's gate pre-activation is g_i = w_i . (gamma * x_hat) = ||w_tilde_i|| *
sqrt(D) * s_i. Everything is float32.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from energy_llm.model import ModelDims, decoder_layers
from energy_llm.storage import atomic_path

log = logging.getLogger("energy_llm.bank")


@dataclass
class Bank:
    gate_weights: list[nn.Parameter]   # per layer W_g, model dtype, on the model device (not copied)
    gamma: list[torch.Tensor]          # per layer [D] float32, model device
    row_norms: list[torch.Tensor]      # per layer [D_ff] float32, ||w_tilde_i||, model device
    d_model: int

    @property
    def device(self) -> torch.device:
        return self.gamma[0].device

    def scores(self, l: int, x_hat: torch.Tensor) -> torch.Tensor:
        """Cosine scores [T, D_ff] for float32 x_hat [T, D] at layer l."""
        w = self.gate_weights[l].detach().float()          # transient [D_ff, D]
        s = (x_hat * self.gamma[l]) @ w.T
        del w
        return s / (self.row_norms[l] * math.sqrt(self.d_model))

    def fingerprint(self) -> str:
        """sha256 over gamma and row norms: identifies the exact weights."""
        h = hashlib.sha256()
        for g, n in zip(self.gamma, self.row_norms):
            h.update(g.cpu().numpy().astype(np.float32).tobytes())
            h.update(n.cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()


@torch.no_grad()
def build_bank(model, dims: ModelDims) -> Bank:
    weights, gammas, norms = [], [], []
    for layer in decoder_layers(model):
        w_g = layer.mlp.gate_proj.weight
        gamma = layer.post_attention_layernorm.weight.detach().float()
        w_tilde = w_g.detach().float() * gamma.unsqueeze(0)      # transient, one layer
        n = torch.linalg.vector_norm(w_tilde, dim=1)
        del w_tilde
        if not torch.all(n > 0):
            raise ValueError("bank has an all-zero row")
        weights.append(w_g)
        gammas.append(gamma)
        norms.append(n)
    return Bank(gate_weights=weights, gamma=gammas, row_norms=norms, d_model=dims.d_model)


def write_bank_h5(path: str | Path, bank: Bank, attrs: dict) -> None:
    import h5py

    with atomic_path(path) as tmp:
        with h5py.File(tmp, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["fingerprint"] = bank.fingerprint()
            for l, (g, n) in enumerate(zip(bank.gamma, bank.row_norms)):
                grp = f.create_group(f"layer_{l}")
                grp.create_dataset("gamma", data=g.cpu().numpy())
                grp.create_dataset("row_norms", data=n.cpu().numpy())


def read_bank_fingerprint(path: str | Path) -> str:
    import h5py

    with h5py.File(path, "r") as f:
        return str(f.attrs["fingerprint"])

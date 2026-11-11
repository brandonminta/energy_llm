"""Memory bank, defined per layer as

    W_g      = gate_proj.weight                     [D_ff, D], rows w_i
    gamma    = effective gain of the FFN norm       [D] (core.model.effective_gain)
    W_tilde  = W_g * gamma   (column j scaled by gamma_j), rows w_tilde_i

Both score variants come from ONE matmul per layer, the model's own gate
pre-activation z_i = w_i . (gamma * x_hat) = w_tilde_i . x_hat:

    cos_gamma_query (primary)  c_i = z_i / (||w_i|| ||gamma * x_hat||)
                               = cos(w_i, gamma * x_hat): key without gamma,
                               query with gamma, an exact cosine
    cos_gamma_key   (control)  s_i = z_i / (||w_tilde_i|| sqrt(D))
                               = cos(w_tilde_i, x_hat) ||x_hat|| / sqrt(D)

With the query gain g_t = ||gamma * x_hat|| / sqrt(D):
s_i = g_t (||w_i|| / ||w_tilde_i||) c_i, i.e. the control variant carries a
position-dependent effective temperature beta * g_t and a per-memory weight.

The bank is NEVER materialised: only gamma and the key norms ||w_i|| and
||w_tilde_i|| (one D_ff vector each per layer) are kept, converting only the
current layer's W_g to float32 and freeing it right after. Everything is float32.

Gamma diagnostic (report only): column j of W_tilde carries
gamma_j^2 * ||W_g[:, j]||^2 of sum_i ||w_tilde_i||^2. Per layer, the k
dimensions with the largest |gamma_j| and the fraction they carry tell
whether a few residual dimensions dominate the cosine scores.
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

from energy_llm.core.model import ModelDims, decoder_layers, effective_gain
from energy_llm.storage import atomic_path

log = logging.getLogger("energy_llm.core.bank")

PRIMARY = "cos_gamma_query"
CONTROL = "cos_gamma_key"
VARIANTS = (PRIMARY, CONTROL)


@dataclass
class Bank:
    gate_weights: list[nn.Parameter]   # per layer W_g, model dtype, on the model device (not copied)
    gamma: list[torch.Tensor]          # per layer [D] float32, model device
    key_norms: list[torch.Tensor]      # per layer [D_ff] float32, ||w_i|| (cos_gamma_query keys)
    row_norms: list[torch.Tensor]      # per layer [D_ff] float32, ||w_tilde_i|| (cos_gamma_key keys)
    d_model: int

    @property
    def device(self) -> torch.device:
        return self.gamma[0].device

    def preactivation(self, l: int, x_hat: torch.Tensor) -> torch.Tensor:
        """z = W_g (gamma * x_hat), [T, D_ff] float32 for float32 x_hat [T, D]."""
        w = self.gate_weights[l].detach().float()          # transient [D_ff, D]
        z = (x_hat * self.gamma[l]) @ w.T
        del w
        return z

    def query_norm(self, l: int, x_hat: torch.Tensor) -> torch.Tensor:
        """||gamma * x_hat|| per position, [T]."""
        return torch.linalg.vector_norm(x_hat * self.gamma[l], dim=-1)

    def query_gain(self, l: int, x_hat: torch.Tensor) -> torch.Tensor:
        """g_t = ||gamma * x_hat|| / sqrt(D), [T]."""
        return self.query_norm(l, x_hat) / math.sqrt(self.d_model)

    def from_preactivation(self, l: int, z: torch.Tensor, x_hat: torch.Tensor,
                           variant: str) -> torch.Tensor:
        """Scores of ``variant`` from z = preactivation(l, x_hat)."""
        if variant == PRIMARY:
            return z / (self.key_norms[l] * self.query_norm(l, x_hat)[..., None])
        if variant == CONTROL:
            return z / (self.row_norms[l] * math.sqrt(self.d_model))
        raise ValueError(f"unknown score variant {variant!r}: {VARIANTS}")

    def scores(self, l: int, x_hat: torch.Tensor, variant: str = CONTROL) -> torch.Tensor:
        """Scores [T, D_ff] of ``variant`` for float32 x_hat [T, D] at layer l."""
        return self.from_preactivation(l, self.preactivation(l, x_hat), x_hat, variant)

    def fingerprint(self) -> str:
        """sha256 over gamma and both key norms: identifies the exact weights."""
        h = hashlib.sha256()
        for g, k, n in zip(self.gamma, self.key_norms, self.row_norms):
            for t in (g, k, n):
                h.update(t.cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()


@torch.no_grad()
def gamma_diagnostic(bank: Bank, k: int = 5) -> dict:
    """Per layer: indices of the k largest |gamma_j| (descending) and the
    fraction of sum_i ||w_tilde_i||^2 carried by those k columns. One layer's
    W_g in float32 at a time, as ``scores``."""
    indices, fraction = [], []
    for w_g, gamma in zip(bank.gate_weights, bank.gamma):
        w = w_g.detach().float()
        col = w.pow(2).sum(dim=0) * gamma.pow(2)          # [D] share of each column
        del w
        top = torch.topk(gamma.abs(), k).indices
        indices.append(top.cpu().tolist())
        fraction.append((col[top].sum() / col.sum()).item())
    return {"k": k, "indices": indices, "fraction": fraction}


@torch.no_grad()
def build_bank(model, dims: ModelDims) -> Bank:
    weights, gammas, keys, norms = [], [], [], []
    for layer in decoder_layers(model):
        w_g = layer.mlp.gate_proj.weight
        gamma = effective_gain(model, layer)
        w = w_g.detach().float()                                  # transient, one layer
        k = torch.linalg.vector_norm(w, dim=1)
        n = torch.linalg.vector_norm(w * gamma.unsqueeze(0), dim=1)
        del w
        if not (torch.all(k > 0) and torch.all(n > 0)):
            raise ValueError("bank has an all-zero row")
        weights.append(w_g)
        gammas.append(gamma)
        keys.append(k)
        norms.append(n)
    return Bank(gate_weights=weights, gamma=gammas, key_norms=keys, row_norms=norms,
                d_model=dims.d_model)


def write_bank_h5(path: str | Path, bank: Bank, attrs: dict) -> None:
    import h5py

    with atomic_path(path) as tmp:
        with h5py.File(tmp, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["fingerprint"] = bank.fingerprint()
            for l, (g, k, n) in enumerate(zip(bank.gamma, bank.key_norms, bank.row_norms)):
                grp = f.create_group(f"layer_{l}")
                grp.create_dataset("gamma", data=g.cpu().numpy())
                grp.create_dataset("key_norms", data=k.cpu().numpy())
                grp.create_dataset("row_norms", data=n.cpu().numpy())


def read_bank_fingerprint(path: str | Path) -> str:
    import h5py

    with h5py.File(path, "r") as f:
        return str(f.attrs["fingerprint"])

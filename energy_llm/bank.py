"""Memory bank, built once per layer.

    W_g      = gate_proj.weight                     [D_ff, D]
    gamma    = post_attention_layernorm.weight      [D]
    W_tilde  = W_g * gamma   (column j scaled by gamma_j)
    bank     = rows of W_tilde normalised to unit norm

With x_hat = x / sqrt(mean(x^2) + eps), the model's gate pre-activation is
g_i = w_i . (gamma * x_hat) = w_tilde_i . x_hat, so with the cosine scores
s_i = (bank_i . x_hat) / sqrt(D) we have g_i = ||w_tilde_i|| * sqrt(D) * s_i
exactly (up to the eps in ||x_hat||). Everything is float32.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from energy_llm.model import ModelDims, decoder_layers
from energy_llm.storage import atomic_path

log = logging.getLogger("energy_llm.bank")


@dataclass
class Bank:
    rows: list[torch.Tensor]       # per layer [D_ff, D] float32, unit-norm rows
    gamma: list[torch.Tensor]      # per layer [D] float32 (diagnostic)
    row_norms: list[torch.Tensor]  # per layer [D_ff] float32, ||w_tilde_i|| (diagnostic)
    device: torch.device

    def fingerprint(self) -> str:
        """sha256 over gamma and row norms: identifies the exact weights."""
        h = hashlib.sha256()
        for g, n in zip(self.gamma, self.row_norms):
            h.update(g.cpu().numpy().astype(np.float32).tobytes())
            h.update(n.cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()

    def nbytes(self) -> int:
        return sum(r.numel() * 4 for r in self.rows)


@torch.no_grad()
def build_bank(model, dims: ModelDims) -> Bank:
    rows, gammas, norms = [], [], []
    for layer in decoder_layers(model):
        w_g = layer.mlp.gate_proj.weight.detach().to("cpu", torch.float32)
        gamma = layer.post_attention_layernorm.weight.detach().to("cpu", torch.float32)
        w_tilde = w_g * gamma.unsqueeze(0)
        n = torch.linalg.vector_norm(w_tilde, dim=1)
        if not torch.all(n > 0):
            raise ValueError("bank has an all-zero row")
        rows.append(w_tilde / n.unsqueeze(1))
        gammas.append(gamma)
        norms.append(n)
    return Bank(rows=rows, gamma=gammas, row_norms=norms, device=torch.device("cpu"))


def place_bank(bank: Bank, requested: str, model_device: torch.device) -> Bank:
    """Move the bank rows to the matmul device. ``auto`` uses the GPU only
    when the rows fit with 2 GiB to spare (they do not on a 10 GB MIG slice
    next to a 3B bf16 model), otherwise the CPU. Precision is float32 either way."""
    target = requested
    if requested == "auto":
        target = "cpu"
        if model_device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(model_device)
            if free > bank.nbytes() + (2 << 30):
                target = "cuda"
    if target == "cuda" and model_device.type != "cuda":
        raise RuntimeError("bank_device=cuda but the model is not on a GPU")
    device = model_device if target == "cuda" else torch.device("cpu")
    log.info("bank on %s (%.2f GB float32)", device, bank.nbytes() / 1e9)
    bank.rows = [r.to(device) for r in bank.rows]
    bank.device = device
    return bank


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

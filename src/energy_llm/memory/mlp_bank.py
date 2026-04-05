from __future__ import annotations

from typing import Any, Iterable

import torch

try:
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - optional dependency in CPU-only setups
    bnb = None


def _get_linear_weight_as_float32(module: Any) -> torch.Tensor:
    """
    Extrae el peso de un modulo lineal como tensor float32 en CPU.

    Soporta:
    - torch.nn.Linear normal
    - bitsandbytes Linear4bit
    - bitsandbytes Linear8bitLt

    Returns:
        Tensor [out_features, in_features] en float32 CPU
    """
    if bnb is not None and isinstance(module, bnb.nn.Linear4bit):
        return bnb.functional.dequantize_4bit(
            module.weight.data,
            module.weight.quant_state,
        ).to(torch.float32).cpu()

    if bnb is not None and isinstance(module, bnb.nn.Linear8bitLt):
        return module.weight.data.to(torch.float32).cpu()

    if hasattr(module, "weight") and module.weight is not None:
        return module.weight.detach().to(torch.float32).cpu()

    raise TypeError(
        f"No se pudo extraer peso de modulo tipo {type(module).__name__}"
    )


def _normalize_rows(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza cada fila del banco de memorias.
    """
    return w / w.norm(dim=1, keepdim=True).clamp_min(eps)


def _get_first_existing_attr(obj: Any, candidates: Iterable[str], role: str) -> tuple[str, Any]:
    """
    Devuelve el primer atributo existente dentro de candidates.
    """
    for name in candidates:
        if hasattr(obj, name):
            return name, getattr(obj, name)

    attrs = [a for a in dir(obj) if ("proj" in a or "fc" in a or a.startswith("w"))]
    raise AttributeError(
        f"No se encontro atributo compatible para role='{role}'. "
        f"Candidatos={tuple(candidates)}. Encontrados={attrs}"
    )


def _resolve_mlp_proj_by_bank(mlp: Any, bank: str) -> tuple[str, Any]:
    """
    Resuelve la proyeccion base segun el rol semantico del banco.

    Bancos semanticos:
        gate -> gate_proj / fc1 / w1
        up   -> up_proj / w3 / fc1
        down -> down_proj / w2 / fc2

    Bancos explicitos:
        fc1, fc2, w1, w2, w3, gate_proj, up_proj, down_proj

    Nota:
        Los aliases fc1/w1/w2/w3 son pragmaticos y no equivalen perfectamente
        entre todas las arquitecturas. Sirven como fallback para exploracion,
        no como garantia de equivalencia mecanicista exacta.
    """
    if bank == "gate":
        return _get_first_existing_attr(mlp, ("gate_proj", "fc1", "w1"), role="gate")

    if bank == "up":
        return _get_first_existing_attr(mlp, ("up_proj", "w3", "fc1"), role="up")

    if bank == "down":
        return _get_first_existing_attr(mlp, ("down_proj", "w2", "fc2"), role="down")

    explicit = {
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "w1",
        "w2",
        "w3",
    }
    if bank in explicit:
        return _get_first_existing_attr(mlp, (bank,), role=bank)

    raise ValueError(f"_resolve_mlp_proj_by_bank no soporta bank='{bank}'")


def extract_mlp_memory_bank(
    llm,
    bank: str = "gate",
    normalize: bool = False,
    strict: bool = True,
    return_metadata: bool = False,
) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    """
    Extrae un banco de memoria por capa desde el MLP.

    Bancos soportados:
        - 'gate'           -> W_gate
        - 'up'             -> W_up
        - 'down'           -> W_down
        - 'gate_plus_up'   -> norm(0.5 * (norm(W_gate) + norm(W_up)))
        - 'gate_up_concat' -> concat filas [W_gate; W_up]

    Tambien soporta proyecciones explicitas:
        - 'gate_proj', 'up_proj', 'down_proj'
        - 'fc1', 'fc2', 'w1', 'w2', 'w3'
    """
    banks: dict[int, torch.Tensor] = {}
    metadata: dict[int, dict[str, Any]] = {}

    config_hidden = getattr(llm.model.config, "hidden_size", None)
    config_intermediate = getattr(llm.model.config, "intermediate_size", None)

    for i, layer in enumerate(llm.layers):
        try:
            if not hasattr(layer, "mlp"):
                raise AttributeError(f"Layer {i} no tiene atributo 'mlp'")

            mlp = layer.mlp

            if bank in {
                "gate", "up", "down",
                "gate_proj", "up_proj", "down_proj",
                "fc1", "fc2", "w1", "w2", "w3",
            }:
                source_name, module = _resolve_mlp_proj_by_bank(mlp, bank)
                w = _get_linear_weight_as_float32(module)
                source = source_name

            elif bank == "gate_plus_up":
                gate_name, gate_mod = _resolve_mlp_proj_by_bank(mlp, "gate")
                up_name, up_mod = _resolve_mlp_proj_by_bank(mlp, "up")

                w_gate = _get_linear_weight_as_float32(gate_mod)
                w_up = _get_linear_weight_as_float32(up_mod)

                if w_gate.shape != w_up.shape:
                    raise ValueError(
                        f"Layer {i}: gate/up shape mismatch: {w_gate.shape} vs {w_up.shape}"
                    )

                w = _normalize_rows(
                    0.5 * (_normalize_rows(w_gate) + _normalize_rows(w_up))
                )
                source = f"norm(0.5*(norm({gate_name})+norm({up_name})))"

            elif bank == "gate_up_concat":
                gate_name, gate_mod = _resolve_mlp_proj_by_bank(mlp, "gate")
                up_name, up_mod = _resolve_mlp_proj_by_bank(mlp, "up")

                w_gate = _get_linear_weight_as_float32(gate_mod)
                w_up = _get_linear_weight_as_float32(up_mod)

                if w_gate.shape[1] != w_up.shape[1]:
                    raise ValueError(
                        f"Layer {i}: gate/up hidden dim mismatch: {w_gate.shape} vs {w_up.shape}"
                    )

                w = torch.cat([w_gate, w_up], dim=0)
                source = f"concat({gate_name}, {up_name})"

            else:
                raise ValueError(
                    f"bank desconocido: '{bank}'. "
                    "Usa uno de: gate, up, down, gate_plus_up, gate_up_concat, "
                    "gate_proj, up_proj, down_proj, fc1, fc2, w1, w2, w3"
                )

            if w.ndim != 2:
                raise ValueError(
                    f"Layer {i}: esperado tensor 2D, obtenido shape={tuple(w.shape)}"
                )

            out_features, in_features = w.shape

            if bank in {
                "gate", "up", "gate_proj", "up_proj",
                "fc1", "w1", "w3", "gate_plus_up", "gate_up_concat"
            }:
                if config_hidden is not None and in_features != config_hidden:
                    raise ValueError(
                        f"Layer {i}: in_features={in_features} != hidden_size={config_hidden}"
                    )

            if bank in {"gate", "up", "gate_proj", "up_proj", "fc1", "w1", "w3"}:
                if config_intermediate is not None and out_features != config_intermediate:
                    raise ValueError(
                        f"Layer {i}: out_features={out_features} != intermediate_size={config_intermediate}"
                    )

            if normalize and bank != "gate_plus_up":
                w = _normalize_rows(w)

            banks[i] = w
            metadata[i] = {
                "layer_idx": i,
                "bank": bank,
                "source": source,
                "shape": tuple(w.shape),
                "dtype": str(w.dtype),
                "device": str(w.device),
                "K": int(w.shape[0]),
                "D": int(w.shape[1]),
            }

        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Fallo extrayendo bank='{bank}' en layer {i}: {e}"
                ) from e

            metadata[i] = {
                "layer_idx": i,
                "bank": bank,
                "error": str(e),
            }

    if return_metadata:
        return banks, metadata
    return banks

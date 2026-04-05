from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class SegmentHiddenStates:
    """
    Resultado de extract_segment_hidden_states.

    Atributos:
        h_question   : [L, D] — representación agregada del segmento question
        h_answer     : [L, D] — representación agregada del segmento answer
        n_layers     : número de capas incluidas
        hidden_size  : dimensión D
        t_question   : tokens del segmento question
        t_separator  : tokens del separador (0 si separator="")
        t_answer     : tokens del segmento answer
        t_total      : tokens totales en el forward pass
        split_idx    : índice donde empieza el answer en la secuencia
        layer_indices: índices reales de capas transformer (1-based)
                       índice 0 de h_question corresponde a layer_indices[0]
        pooling      : estrategia usada
        normalized   : si los vectores fueron L2-normalizados
    """
    h_question   : torch.Tensor
    h_answer     : torch.Tensor
    n_layers     : int
    hidden_size  : int
    t_question   : int
    t_separator  : int
    t_answer     : int
    t_total      : int
    split_idx    : int
    layer_indices: list[int]
    pooling      : str
    normalized   : bool


def _pool_tokens(
    h: torch.Tensor,
    pooling: Literal["mean", "last", "max", "first"],
) -> torch.Tensor:
    """
    Reduce [L, T, D] → [L, D] sobre la dimensión de tokens.

    Estrategias:
        mean  : promedio de todos los tokens del segmento
        last  : último token (más informativo en modelos causales)
        first : primer token
        max   : max pooling por dimensión
    """
    if h.shape[1] == 0:
        raise ValueError("Segmento vacío: no hay tokens para agregar.")

    if pooling == "mean":
        return h.mean(dim=1)
    if pooling == "last":
        return h[:, -1, :]
    if pooling == "first":
        return h[:, 0, :]
    if pooling == "max":
        return h.max(dim=1).values

    raise ValueError(
        f"pooling desconocido: '{pooling}'. Usa: mean, last, max, first"
    )


@torch.no_grad()
def extract_segment_hidden_states(
    llm,
    question: str,
    answer: str,
    separator: str = "\n",
    pooling: Literal["mean", "last", "max", "first"] = "mean",
    layers: list[int] | None = None,
    normalize: bool = False,
    to_cpu: bool = True,
) -> SegmentHiddenStates:
    """
    Extrae representaciones layer-wise [L, D] para question y answer
    desde un único forward pass.

    Nota de tokenización:
        La secuencia completa se construye por concatenación explícita
        de IDs tokenizados por segmento (question / separator / answer)
        para preservar un boundary exacto y controlado entre segmentos.
        Esto no es idéntico a tokenizar el string unido directamente,
        pero garantiza un split preciso sin ambigüedad en la frontera.

    Pipeline:
        1. Tokeniza question  (con special tokens)
        2. Tokeniza separator (sin special tokens)
        3. Tokeniza answer    (sin special tokens)
        4. Concatena ids_full = [ids_q | ids_sep | ids_a]
        5. Forward pass → hidden_states tuple: (L+1) x [1, T, D]
        6. Descarta índice 0 (embeddings crudos)
        7. Selecciona capas solicitadas
        8. Segmenta por split_idx (inicio del answer)
        9. Pool tokens → h_question [L, D], h_answer [L, D]

    Args:
        llm:
            Instancia HFLLM con output_hidden_states=True.
        question:
            Texto del segmento de referencia.
        answer:
            Texto del segmento a comparar.
        separator:
            Texto insertado entre question y answer.
            Default: "\\n". Usar "" para omitir.
            Se tokeniza por separado para mantener split exacto.
        pooling:
            Estrategia de agregación sobre tokens del segmento:
            - 'mean'  : promedio (recomendado baseline)
            - 'last'  : último token (recomendado modelos causales)
            - 'first' : primer token
            - 'max'   : max pooling
        layers:
            Índices de capas a incluir (1-based, excluye embedding).
            Si None, incluye todas las capas transformer.
            Ejemplo: layers=list(range(7, 22)) para capas medias.
        normalize:
            Si True, L2-normaliza cada vector [D].
            Default False: dejar que hopfield_energy_mlp con
            energy_mode='cosine' maneje la normalización.
        to_cpu:
            Si True, mueve tensores a CPU antes de retornar.
            Para análisis offline: True.
            Para pipeline en GPU: False.

    Returns:
        SegmentHiddenStates con h_question y h_answer listos para
        compute_layer_divergences.
    """
    if not llm.output_hidden_states:
        raise RuntimeError(
            "HFLLM debe inicializarse con output_hidden_states=True."
        )
    if not question.strip():
        raise ValueError("El question no puede estar vacío.")
    if not answer.strip():
        raise ValueError("El answer no puede estar vacío.")

    # ----------------------------------------------------------
    # 1. Tokenización por segmento
    # ----------------------------------------------------------
    ids_q = llm.tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids[0]                                     # [T_q]

    ids_sep = llm.tokenizer(
        separator,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0] if separator else torch.tensor([], dtype=torch.long)  # [T_sep]

    ids_a = llm.tokenizer(
        answer,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]                                     # [T_a]

    if len(ids_a) == 0:
        raise ValueError("El answer tokenizó a secuencia vacía.")

    # ----------------------------------------------------------
    # 2. Concatenación y split_idx
    # ----------------------------------------------------------
    ids_full = torch.cat([ids_q, ids_sep, ids_a], dim=0).unsqueeze(0)  # [1, T_total]
    ids_full = ids_full.to(llm.device)

    attention_mask = torch.ones_like(ids_full, device=llm.device)

    # split_idx: primer token del answer (después de question + separator)
    split_idx  = len(ids_q) + len(ids_sep)
    t_question = len(ids_q)
    t_separator= len(ids_sep)
    t_answer   = len(ids_a)
    t_total    = t_question + t_separator + t_answer

    # ----------------------------------------------------------
    # 3. Forward pass
    # ----------------------------------------------------------
    outputs = llm.model(
        input_ids=ids_full,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # tuple de (L+1) tensores [1, T_total, D]
    # índice 0 = embeddings crudos, 1..L = capas transformer
    all_hs = outputs.hidden_states
    n_transformer_layers = len(all_hs) - 1

    # ----------------------------------------------------------
    # 4. Selección de capas
    # ----------------------------------------------------------
    if layers is None:
        layer_indices = list(range(1, n_transformer_layers + 1))
    else:
        invalid = [l for l in layers if l < 1 or l > n_transformer_layers]
        if invalid:
            raise ValueError(
                f"Índices de capa inválidos: {invalid}. "
                f"Rango válido: [1, {n_transformer_layers}]"
            )
        layer_indices = sorted(layers)

    # ----------------------------------------------------------
    # 5. Stack y segmentación
    # ----------------------------------------------------------
    # [L_out, T_total, D]
    stacked = torch.stack(
        [all_hs[l][0] for l in layer_indices],
        dim=0,
    ).float()

    if to_cpu:
        stacked = stacked.cpu()

    h_q_tokens = stacked[:, :t_question, :]           # [L_out, T_q,  D]
    h_a_tokens = stacked[:, split_idx:,  :]           # [L_out, T_a,  D]

    # ----------------------------------------------------------
    # 6. Pooling → [L_out, D]
    # ----------------------------------------------------------
    h_question = _pool_tokens(h_q_tokens, pooling)
    h_answer   = _pool_tokens(h_a_tokens, pooling)

    # ----------------------------------------------------------
    # 7. Normalización opcional
    # ----------------------------------------------------------
    if normalize:
        h_question = F.normalize(h_question, dim=1)
        h_answer   = F.normalize(h_answer,   dim=1)

    return SegmentHiddenStates(
        h_question   = h_question,
        h_answer     = h_answer,
        n_layers     = len(layer_indices),
        hidden_size  = stacked.shape[2],
        t_question   = t_question,
        t_separator  = t_separator,
        t_answer     = t_answer,
        t_total      = t_total,
        split_idx    = split_idx,
        layer_indices= layer_indices,
        pooling      = pooling,
        normalized   = normalize,
    )

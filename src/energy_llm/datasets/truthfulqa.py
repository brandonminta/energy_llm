from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from datasets import load_dataset


# ============================================================
# Estructura de una muestra lista para los experimentos
# ============================================================

@dataclass
class TruthfulQASample:
    """
    Muestra normalizada de TruthfulQA lista para el pipeline.

    Atributos:
        idx               : índice original en el dataset
        question          : pregunta
        answer_factual    : respuesta correcta seleccionada
        answer_hallucinated: respuesta incorrecta seleccionada
        category          : categoría temática (Health, Law, etc.)
        source_factual    : de dónde vino la respuesta factual
        source_hallucinated: de dónde vino la respuesta alucinada
        label             : siempre 1 (alucinado=1, factual=0)
                            convención fija para ROC/AUC
    """
    idx               : int
    question          : str
    answer_factual    : str
    answer_hallucinated: str
    category          : str
    source_factual    : Literal["best_answer", "correct_answers"]
    source_hallucinated: str   # "incorrect_answers[i]"
    label             : int = 1


# ============================================================
# Carga y normalización
# ============================================================

def load_truthfulqa(
    factual_strategy  : Literal["best", "random_correct", "all_correct"] = "best",
    hallucinated_strategy: Literal["random_incorrect", "all_incorrect"] = "random_incorrect",
    categories        : list[str] | None = None,
    max_samples       : int | None = None,
    seed              : int = 42,
    verbose           : bool = True,
) -> list[TruthfulQASample]:
    """
    Carga TruthfulQA (config=generation) y lo normaliza a lista de
    TruthfulQASample lista para extract_segment_hidden_states.

    Args:
        factual_strategy:
            - 'best'           : usa best_answer (1 muestra por pregunta)
            - 'random_correct' : elige 1 respuesta aleatoria de correct_answers
            - 'all_correct'    : genera una muestra por cada correct_answer
                                 cruzada con 1 incorrect_answer aleatoria

        hallucinated_strategy:
            - 'random_incorrect': elige 1 respuesta aleatoria de incorrect_answers
            - 'all_incorrect'   : genera una muestra por cada incorrect_answer
                                  (solo compatible con factual_strategy='best'
                                  o 'random_correct')

        categories:
            Si no es None, filtra por categorías del dataset.
            Ejemplo: ["Health", "Law", "Science"]

        max_samples:
            Límite de muestras finales. Útil para pruebas rápidas.

        seed:
            Semilla para selecciones aleatorias. Reproducibilidad.

        verbose:
            Imprime resumen de carga.

    Returns:
        Lista de TruthfulQASample ordenada por idx original.

    Raises:
        ValueError: si una fila no tiene incorrect_answers o correct_answers.
    """
    rng = random.Random(seed)

    raw = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    if categories is not None:
        raw = raw.filter(lambda x: x["category"] in categories)

    samples: list[TruthfulQASample] = []
    skipped = 0

    for idx, row in enumerate(raw):
        question     = row["question"].strip()
        best_answer  = row["best_answer"].strip()
        correct_list = [a.strip() for a in row["correct_answers"] if a.strip()]
        incorrect_list = [a.strip() for a in row["incorrect_answers"] if a.strip()]
        category     = row["category"]

        # Validación mínima
        if not incorrect_list:
            skipped += 1
            continue
        if not correct_list and factual_strategy != "best":
            skipped += 1
            continue

        # --------------------------------------------------
        # Selección de respuesta factual
        # --------------------------------------------------
        if factual_strategy == "best":
            factual_candidates = [(best_answer, "best_answer")]

        elif factual_strategy == "random_correct":
            chosen = rng.choice(correct_list)
            factual_candidates = [(chosen, "correct_answers")]

        elif factual_strategy == "all_correct":
            factual_candidates = [(a, "correct_answers") for a in correct_list]

        else:
            raise ValueError(f"factual_strategy desconocida: '{factual_strategy}'")

        # --------------------------------------------------
        # Selección de respuesta alucinada
        # --------------------------------------------------
        if hallucinated_strategy == "random_incorrect":
            hallucinated_candidates = [(rng.choice(incorrect_list), "incorrect_answers[random]")]

        elif hallucinated_strategy == "all_incorrect":
            hallucinated_candidates = [
                (a, f"incorrect_answers[{i}]") for i, a in enumerate(incorrect_list)
            ]

        else:
            raise ValueError(f"hallucinated_strategy desconocida: '{hallucinated_strategy}'")

        # --------------------------------------------------
        # Producto cruzado factual × hallucinated
        # --------------------------------------------------
        for (ans_f, src_f) in factual_candidates:
            for (ans_h, src_h) in hallucinated_candidates:
                samples.append(TruthfulQASample(
                    idx                = idx,
                    question           = question,
                    answer_factual     = ans_f,
                    answer_hallucinated= ans_h,
                    category           = category,
                    source_factual     = src_f,
                    source_hallucinated= src_h,
                    label              = 1,
                ))

    # Límite de muestras
    if max_samples is not None:
        rng.shuffle(samples)
        samples = samples[:max_samples]
        samples.sort(key=lambda s: s.idx)

    if verbose:
        cats = sorted({s.category for s in samples})
        print(f"[TruthfulQA] Cargadas  : {len(samples)} muestras")
        print(f"[TruthfulQA] Skipped   : {skipped} (sin incorrect/correct answers)")
        print(f"[TruthfulQA] Categorías: {len(cats)}")
        if categories is not None:
            print(f"[TruthfulQA] Filtro    : {categories}")

    return samples


# ============================================================
# Utilidades de inspección
# ============================================================

def sample_summary(s: TruthfulQASample) -> str:
    return (
        f"[{s.idx}] [{s.category}]\n"
        f"  Q  : {s.question}\n"
        f"  F  : {s.answer_factual}  ({s.source_factual})\n"
        f"  H  : {s.answer_hallucinated}  ({s.source_hallucinated})\n"
    )


def dataset_stats(samples: list[TruthfulQASample]) -> dict:
    from collections import Counter
    cat_counts = Counter(s.category for s in samples)
    return {
        "n_samples"  : len(samples),
        "n_categories": len(cat_counts),
        "categories" : dict(cat_counts.most_common()),
    }
"""Per-layer AUROC computation for hallucination detection.

per_layer_auroc_table  — DataFrame of AUROC values indexed by layer, one
                         column per feature key.
best_single_feature    — (feature_key, layer, AUROC) of the global best cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.per_layer_auroc")


def per_layer_auroc_table(
    samples: list[dict],
    per_layer_feature_keys: list[str],
) -> pd.DataFrame:
    """Compute per-layer AUROC for each feature against is_hallucination.

    Hallucination is the positive class (label=1).  NaN feature values are
    imputed with the per-layer median before calling roc_auc_score; imputation
    rates are logged.

    Args:
        samples:                 List of feature dicts from load_sample_features.
                                 Each must have 'is_hallucination' (bool) and all
                                 keys in per_layer_feature_keys (np.ndarray [L]).
        per_layer_feature_keys:  Feature keys whose values are [L] arrays.

    Returns:
        DataFrame with integer layer index (0..L-1) rows plus a 'best_layer'
        summary row.  Columns correspond to per_layer_feature_keys.
        The 'best_layer' row contains the layer index (int) achieving the
        highest AUROC per column.
    """
    if not samples or not per_layer_feature_keys:
        return pd.DataFrame()

    labels = np.array([int(s["is_hallucination"]) for s in samples], dtype=float)

    # Determine L from the first sample's first key
    L = samples[0][per_layer_feature_keys[0]].shape[0]

    records: dict[str, list[float]] = {}
    for key in per_layer_feature_keys:
        feature_matrix = np.stack(
            [s[key].astype(np.float64) for s in samples], axis=0
        )  # [N, L]

        # NaN imputation
        n_nan = int(np.isnan(feature_matrix).sum())
        if n_nan > 0:
            impute_rate = n_nan / feature_matrix.size
            log.info("per_layer_auroc: imputing %d NaNs (%.1f%%) in '%s'",
                     n_nan, impute_rate * 100, key)
            for l in range(L):
                col = feature_matrix[:, l]
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    median = np.nanmedian(col)
                    col[nan_mask] = median

        layer_aurocs: list[float] = []
        for l in range(L):
            scores = feature_matrix[:, l]
            try:
                auroc = float(roc_auc_score(labels, scores))
            except ValueError:
                auroc = float("nan")
            layer_aurocs.append(auroc)
        records[key] = layer_aurocs

    df = pd.DataFrame(records, index=range(L))
    df.index.name = "layer"

    # Summary row: per-column argmax layer index
    best_layers = {
        key: int(np.nanargmax(df[key].values)) for key in per_layer_feature_keys
    }
    df = pd.concat([df, pd.DataFrame([best_layers], index=["best_layer"])])
    return df


def best_single_feature(
    table: pd.DataFrame,
) -> tuple[str, int, float]:
    """Return (feature_key, layer, AUROC) of the globally best cell.

    Ignores the 'best_layer' summary row.

    Args:
        table: DataFrame produced by per_layer_auroc_table.

    Returns:
        Tuple of (feature_key, layer_index, AUROC_value).
    """
    data = table.drop("best_layer", errors="ignore")
    best_col:   str   = ""
    best_layer: int   = 0
    best_auroc: float = float("nan")

    for col in data.columns:
        col_vals = data[col].values.astype(float)
        idx = int(np.nanargmax(col_vals))
        val = float(col_vals[idx])
        if np.isnan(best_auroc) or val > best_auroc:
            best_auroc = val
            best_col   = col
            best_layer = int(data.index[idx])

    return (best_col, best_layer, best_auroc)

"""Generate notebooks 03 and 04 for exp04 analysis."""
import json
from pathlib import Path

OUT = Path("/home/brandon/energy_llm/notebooks")

def md(text): return {"cell_type":"markdown","id":str(id(text))[-8:],"metadata":{},"source":text}
def code(text): return {"cell_type":"code","id":str(id(text))[-8:],"metadata":{},"source":text,"outputs":[],"execution_count":None}
def nb(cells): return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12.0"}},"cells":cells}

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 03
# ─────────────────────────────────────────────────────────────────────────────

NB03_CELLS = [

md("""# Exp-04: Hopfield Energy Analysis — Qwen2.5-3B on TruthfulQA

**Research question:** Can Modern Hopfield retrieval energy measured at MLP layers during generation discriminate hallucinated from factual answers?

## Theoretical background

The Modern Hopfield network stores $K$ patterns $\\{\\xi_k\\}$ (rows of $W_{\\text{up}}$, dimension $d$).
Given a query $q$ (last-token hidden state at a layer), the **retrieval energy** is:

$$E(q) = -\\beta^{-1}\\,\\log\\!\\sum_{k=1}^{K}\\exp(\\beta\\,q^\\top\\xi_k)\\;+\\;\\frac{1}{2}\\|q\\|^2 + \\beta^{-1}\\log K$$

Decomposed into:
- $E_{\\mathrm{lse}} = -\\beta^{-1}\\log\\sum_k\\exp(\\beta\\,q^\\top\\xi_k)$ — how concentrated retrieval is (lower = more peaked)
- $E_{\\mathrm{quad}} = \\frac{1}{2}\\|q\\|^2$ — query norm contribution
- $E_{\\mathrm{offset}} = \\beta^{-1}\\log K$ — constant making energies cross-layer comparable via the $M^2$ normalisation

**Key signal:** $\\Delta E_\\ell = \\overline{E}_\\ell^{\\mathrm{gen}} - E_\\ell^{\\mathrm{prefill}}$.
Hallucinated generations are expected to probe memory banks differently than factual ones — leaving a detectable shift in per-layer energy.

**Experiment:** 500 TruthfulQA samples, greedy decoding (do\_sample=False), chat-template applied identically to prefill and generation hooks, $\\beta=15.0$ (sub-optimal but above collapse, target fraction $0.55\\times\\log K$).
Labels: GPT-5.4-mini binary judge (317 HAL / 183 OK).
"""),

code("""\
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

TRAJ_DIR  = Path("/home/brandon/energy_llm/exp04_results/trajectories")
LABEL_DIR = Path("/home/brandon/energy_llm/exp04_results/labels")
CALIB_PATH = TRAJ_DIR / "calibration_beta.json"
ANALYSIS_PATH = Path("/home/brandon/energy_llm/exp04_results/analysis.json")
PLOTS_DIR = Path("/home/brandon/energy_llm/outputs/nb03_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 36
K        = 11008   # MLP intermediate dim = number of memory patterns

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
COLORS = {"HAL": "#e74c3c", "OK": "#2ecc71"}
LAYERS = np.arange(N_LAYERS)
print("Config ready.")
"""),

md("## 1 — Load Data"),

code("""\
with open(CALIB_PATH) as f:
    calib = json.load(f)
with open(ANALYSIS_PATH) as f:
    analysis = json.load(f)

rows = []
for jpath in sorted(TRAJ_DIR.glob("*.json")):
    if jpath.name.startswith("calibration"):
        continue
    meta = json.loads(jpath.read_text())
    sid  = meta["id"]
    npz_path   = jpath.with_suffix(".npz")
    label_path = LABEL_DIR / f"{sid}_label.json"
    if not npz_path.exists() or not label_path.exists():
        continue
    lbl = json.loads(label_path.read_text())
    npz = np.load(npz_path, allow_pickle=True)

    rows.append({
        "id": sid,
        "question":        meta["question"],
        "generated_text":  meta["generated_text"],
        "category":        meta.get("category", "Unknown"),
        "n_tokens":        int(meta["n_tokens_generated"]),
        "is_hallucination": lbl["is_hallucination"],
        # Prefill scalars [L]
        "pre_energy":      npz["prefill_energy"].astype(float),
        "pre_entropy":     npz["prefill_entropy"].astype(float),
        "pre_norm_entropy":npz["prefill_norm_entropy"].astype(float),
        "pre_lse":         npz["prefill_lse"].astype(float),
        "pre_quadratic":   npz["prefill_quadratic"].astype(float),
        "pre_top_act":     npz["prefill_top_act"].astype(float),
        # Generation [L, T]
        "gen_energy":      npz["gen_energy"].astype(float),
        "gen_entropy":     npz["gen_entropy"].astype(float),
        "gen_norm_entropy":npz["gen_norm_entropy"].astype(float),
        "gen_lse":         npz["gen_lse"].astype(float),
        "gen_quadratic":   npz["gen_quadratic"].astype(float),
        "gen_top_act":     npz["gen_top_act"].astype(float),
        "gen_js":          npz["gen_js"].astype(float),
        "gen_hellinger":   npz["gen_hellinger"].astype(float),
    })

df = pd.DataFrame(rows)

# Delta metrics: mean_gen_per_layer - prefill_per_layer
for m in ["energy", "entropy", "norm_entropy", "lse", "quadratic", "top_act"]:
    df[f"d_{m}"]       = df.apply(lambda r: np.nanmean(r[f"gen_{m}"], axis=1) - r[f"pre_{m}"], axis=1)
    df[f"score_{m}"]   = df[f"d_{m}"].apply(np.nanmean)

# JS / Hellinger (gen only, averaged over tokens per layer, then scalar)
df["gen_js_layer"]       = df["gen_js"].apply(lambda x: np.nanmean(x, axis=1))
df["gen_hell_layer"]     = df["gen_hellinger"].apply(lambda x: np.nanmean(x, axis=1))
df["score_js"]           = df["gen_js_layer"].apply(np.nanmean)
df["score_hellinger"]    = df["gen_hell_layer"].apply(np.nanmean)

df["label"] = df["is_hallucination"].map({True: "HAL", False: "OK"})

n_hal = df["is_hallucination"].sum()
n_ok  = (~df["is_hallucination"]).sum()
print(f"Loaded {len(df)} samples  |  HAL={n_hal}  OK={n_ok}  ({n_hal/len(df)*100:.1f}% hallucinated)")
"""),

md("## 2 — Dataset Overview"),

code("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Class balance
ax = axes[0]
counts = [n_ok, n_hal]
bars = ax.bar(["Factual (OK)", "Hallucinated"], counts,
              color=[COLORS["OK"], COLORS["HAL"]], alpha=0.85, width=0.5)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(c),
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Samples")
ax.set_title("Class Balance")
ax.set_ylim(0, max(counts) * 1.2)

# Token length distribution
ax = axes[1]
for lbl, color in COLORS.items():
    vals = df[df["label"] == lbl]["n_tokens"]
    ax.hist(vals, bins=20, alpha=0.55, color=color, density=True, label=lbl)
    ax.axvline(vals.mean(), color=color, lw=1.8, ls="--")
ax.set_xlabel("Tokens generated")
ax.set_ylabel("Density")
ax.set_title("Generation Length")
ax.legend()

# Correctness score distribution
ax = axes[2]
for lbl, color in COLORS.items():
    vals = df[df["label"] == lbl]["is_hallucination"].astype(int)
ax.hist(df["is_hallucination"].astype(int), bins=3, color="#3498db", alpha=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["OK (0)", "HAL (1)"])
ax.set_ylabel("Count")
ax.set_title("Binary Label Distribution")

fig.suptitle("Exp-04 Dataset Overview — Qwen2.5-3B TruthfulQA 500", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_dataset_overview.png", bbox_inches="tight")
plt.show()

# Token stats by class
print(df.groupby("label")["n_tokens"].describe().round(2))
"""),

md("""\
## 3 — Beta Calibration

Beta controls how peaked the retrieval distribution is over the $K$ memory patterns.
Normalised entropy $H_\\text{norm} = H / \\log K$ measures distance from two extremes:
- $H_\\text{norm} \\approx 1$: flat — every memory pattern equally activated (no retrieval structure)
- $H_\\text{norm} \\approx 0$: collapsed — only one pattern retrieves

The calibrated $\\beta^\\star = 68.16$ would place $\\overline{H}_\\text{norm} = 0.55\\times\\log K$.
This run used $\\beta = 15.0$ (in the early-regime, $H_\\text{norm} \\approx 0.96$) which is above the
collapse threshold but sub-optimal — retrieval is still relatively diffuse. This is important context
for interpreting the magnitude of divergence signals.
"""),

code("""\
betas = [float(k) for k in calib["mean_norm_entropy_by_beta"].keys()]
h_norms = list(calib["mean_norm_entropy_by_beta"].values())
target = calib["target_fraction"]  # 0.55
beta_star = calib["beta_star"]      # 68.16
log_K = calib["log_K"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(betas, h_norms, "o-", color="#3498db", lw=2.5, markersize=7, label="Calibration curve")
ax.axhline(target, color="#e67e22", lw=1.5, ls="--", label=f"Target: {target}×log K = {target*log_K:.2f}")
ax.axvline(15.0, color="#e74c3c", lw=1.5, ls=":", label=f"β used = 15.0  (H_norm ≈ {calib['mean_norm_entropy_by_beta']['15.0']:.3f})")
ax.axvline(beta_star, color="#27ae60", lw=1.5, ls=":", label=f"β★ = {beta_star:.1f}  (H_norm = target)")

ax.set_xlabel("β  (log scale)")
ax.set_ylabel("Mean normalised entropy  H / log K")
ax.set_title("Beta Calibration Curve — 30 calibration samples, Qwen2.5-3B Layer Mean")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xlim(0.8, 150)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_beta_calibration.png", bbox_inches="tight")
plt.show()

used_h = calib["mean_norm_entropy_by_beta"]["15.0"]
print(f"β=15  →  H_norm = {used_h:.4f}  (diffuse regime, retrieval is {used_h:.0%} flat)")
print(f"β★={beta_star:.1f}  would give H_norm = {target} (target detection regime)")
"""),

md("""\
## 4 — Prefill Energy Profile: The Memory Prior

The prefill energy $E_\\ell^{\\text{pre}}$ represents how the model's residual stream activates each
layer's MLP memory bank *before* generation starts. This is the baseline memory state we compare against.
"""),

code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 4a: mean prefill energy per layer, HAL vs OK
ax = axes[0]
for lbl, color in COLORS.items():
    mat = np.vstack(df[df["label"] == lbl]["pre_energy"].values)
    m, s = mat.mean(0), mat.std(0)
    ax.plot(LAYERS, m, color=color, lw=2.5, label=lbl)
    ax.fill_between(LAYERS, m - s, m + s, color=color, alpha=0.13)
ax.set_xlabel("Layer")
ax.set_ylabel("Prefill energy  E_pre")
ax.set_title("Prefill Energy (prompt baseline)")
ax.legend()

# 4b: mean generation energy per layer (mean over tokens), HAL vs OK
ax = axes[1]
for lbl, color in COLORS.items():
    mat = np.vstack(df[df["label"] == lbl].apply(lambda r: np.nanmean(r["gen_energy"], axis=1), axis=1).values)
    m, s = mat.mean(0), mat.std(0)
    ax.plot(LAYERS, m, color=color, lw=2.5, label=lbl)
    ax.fill_between(LAYERS, m - s, m + s, color=color, alpha=0.13)
ax.set_xlabel("Layer")
ax.set_ylabel("Generation energy  E_gen")
ax.set_title("Generation Energy (mean over tokens)")
ax.legend()

# 4c: prefill norm_entropy per layer
ax = axes[2]
for lbl, color in COLORS.items():
    mat = np.vstack(df[df["label"] == lbl]["pre_norm_entropy"].values)
    m, s = mat.mean(0), mat.std(0)
    ax.plot(LAYERS, m, color=color, lw=2.5, label=lbl)
    ax.fill_between(LAYERS, m - s, m + s, color=color, alpha=0.13)
ax.axhline(0.55, color="gray", ls="--", lw=1, alpha=0.6, label="Target fraction")
ax.set_xlabel("Layer")
ax.set_ylabel("Norm entropy  H / log K")
ax.set_title("Prefill Norm Entropy (β=15, sub-optimal)")
ax.legend()

fig.suptitle("Memory State During Prefill — HAL vs OK (mean ± 1 std)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_prefill_profiles.png", bbox_inches="tight")
plt.show()
"""),

md("""\
## 5 — Delta Metrics: Generation vs Prefill

$\\Delta X_\\ell = \\overline{X}_\\ell^{\\text{gen}} - X_\\ell^{\\text{pre}}$ for each metric $X$.
Positive $\\Delta E$: generation queries more concentrated/lower energy than prefill.
"""),

code("""\
DELTA_METRICS = [
    ("d_energy",      r"$\\Delta E$ (energy)",            "#3498db"),
    ("d_entropy",     r"$\\Delta H$ (entropy)",            "#9b59b6"),
    ("d_norm_entropy",r"$\\Delta H_{norm}$ (norm entropy)","#e67e22"),
    ("d_lse",         r"$\\Delta E_{lse}$ (log-sum-exp)",  "#1abc9c"),
    ("d_top_act",     r"$\\Delta$ top activation",          "#e74c3c"),
    ("d_quadratic",   r"$\\Delta E_{quad}$ (‖q‖²/2)",      "#f39c12"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, (col, title, color) in zip(axes, DELTA_METRICS):
    for lbl, lcolor in COLORS.items():
        mat = np.vstack(df[df["label"] == lbl][col].values)
        m, s = mat.mean(0), mat.std(0)
        ax.plot(LAYERS, m, color=lcolor, lw=2.3, label=lbl)
        ax.fill_between(LAYERS, m - s, m + s, color=lcolor, alpha=0.12)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, N_LAYERS, 6))

fig.suptitle("Delta Metrics: Generation − Prefill  |  HAL vs OK  (mean ± 1 std)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_delta_metrics.png", bbox_inches="tight")
plt.show()
"""),

md("""\
## 6 — AUROC by Layer for Every Metric

Per-layer AUROC asks: *at layer $\\ell$, how well does $\\Delta X_\\ell$ separate HAL from OK?*
AUROC > 0.5 means HAL samples have higher $\\Delta X_\\ell$ on average; < 0.5 means the signal is inverted.
"""),

code("""\
def auroc_by_layer(df_, col):
    y = df_["is_hallucination"].astype(int).values
    mat = np.vstack(df_[col].values)
    return np.array([
        roc_auc_score(y, mat[:, l]) if np.unique(mat[:, l]).size > 1 else 0.5
        for l in range(mat.shape[1])
    ])

ALL_METRICS = {
    r"$\\Delta E$":               "d_energy",
    r"$\\Delta H$":               "d_entropy",
    r"$\\Delta H_{norm}$":        "d_norm_entropy",
    r"$\\Delta E_{lse}$":         "d_lse",
    r"$\\Delta$ top act":         "d_top_act",
    r"JS divergence":             "gen_js_layer",
    r"Hellinger dist":            "gen_hell_layer",
}

METRIC_COLORS = ["#3498db","#9b59b6","#e67e22","#1abc9c","#e74c3c","#f39c12","#2c3e50"]

fig, ax = plt.subplots(figsize=(13, 5))
auroc_store = {}
for (label, col), color in zip(ALL_METRICS.items(), METRIC_COLORS):
    auc = auroc_by_layer(df, col)
    auroc_store[label] = auc
    ax.plot(LAYERS, auc, color=color, lw=2, label=label, alpha=0.9)
    best = int(np.argmax(auc))
    if auc[best] > 0.58:
        ax.annotate(f"L{best}={auc[best]:.3f}", (best, auc[best]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)

ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="Random (0.5)")
ax.set_xlabel("Layer index")
ax.set_ylabel("AUROC")
ax.set_title("Per-layer AUROC — All Delta Metrics  |  Qwen2.5-3B Exp-04", fontsize=12)
ax.set_xticks(range(0, N_LAYERS, 2))
ax.set_ylim(0.3, 0.80)
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_auroc_by_layer_all_metrics.png", bbox_inches="tight")
plt.show()

# Best layer and overall AUROC per metric
y = df["is_hallucination"].astype(int).values
print(f"{'Metric':<22}  {'Best layer':>10}  {'Best AUROC':>11}  {'Overall AUROC':>14}")
print("-" * 62)
SCALAR_SCORES = {r"$\\Delta E$":"score_energy", r"$\\Delta H$":"score_entropy",
                 r"$\\Delta H_{norm}$":"score_norm_entropy", r"$\\Delta E_{lse}$":"score_lse",
                 r"$\\Delta$ top act":"score_top_act", r"JS divergence":"score_js",
                 r"Hellinger dist":"score_hellinger"}
for lbl, col in ALL_METRICS.items():
    auc = auroc_store[lbl]
    bl = int(np.argmax(auc))
    sc = SCALAR_SCORES[lbl]
    oa = roc_auc_score(y, df[sc])
    print(f"{lbl:<22}  {bl:>10}  {auc[bl]:>11.4f}  {oa:>14.4f}")
"""),

md("""\
## 7 — Energy Decomposition: LSE vs Quadratic Terms

The energy splits into:
$$E_\\ell = \\underbrace{-\\beta^{-1}\\log\\sum_k e^{\\beta q^\\top\\xi_k}}_{E_{\\text{lse}} : \\text{retrieval concentration}}
           + \\underbrace{\\tfrac{1}{2}\\|q\\|^2}_{E_{\\text{quad}} : \\text{query norm}}$$

$E_{\\text{lse}}$ tracks *which patterns* the query activates most.
$E_{\\text{quad}}$ tracks *how large* the query vector is.
Disentangling them shows whether hallucination signal is memory-pattern driven or query-magnitude driven.
"""),

code("""\
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

titles = [
    ("pre_lse",      "Prefill  $E_{lse}$  (log-sum-exp term)",   axes[0][0]),
    ("pre_quadratic","Prefill  $E_{quad}$  (query norm term)",    axes[0][1]),
]
for col, title, ax in titles:
    for lbl, color in COLORS.items():
        mat = np.vstack(df[df["label"] == lbl][col].values)
        m, s = mat.mean(0), mat.std(0)
        ax.plot(LAYERS, m, color=color, lw=2.3, label=lbl)
        ax.fill_between(LAYERS, m - s, m + s, color=color, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.legend()

delta_titles = [
    ("d_lse",      r"$\\Delta E_{lse}$ (gen−pre)",      axes[1][0]),
    ("d_quadratic",r"$\\Delta E_{quad}$ (gen−pre)",     axes[1][1]),
]
for col, title, ax in delta_titles:
    for lbl, color in COLORS.items():
        mat = np.vstack(df[df["label"] == lbl][col].values)
        m, s = mat.mean(0), mat.std(0)
        ax.plot(LAYERS, m, color=color, lw=2.3, label=lbl)
        ax.fill_between(LAYERS, m - s, m + s, color=color, alpha=0.12)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.legend()

fig.suptitle("Energy Decomposition: Retrieval (LSE) vs Query Norm (Quadratic)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_energy_decomposition.png", bbox_inches="tight")
plt.show()

# Correlation between delta_lse and delta_quadratic
corr = np.corrcoef(df["score_lse"].values, df["score_energy"].values)[0, 1]
print(f"Pearson r(Δ_lse, Δ_energy) = {corr:.4f}")
corr2 = np.corrcoef(df["score_quadratic"].values, df["score_energy"].values)[0, 1]
print(f"Pearson r(Δ_quad, Δ_energy) = {corr2:.4f}")
"""),

md("## 8 — Hallucination Fingerprint: Sample × Layer Heatmap"),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, col, title in [
    (axes[0], "d_energy", r"$\\Delta E$ (energy)"),
    (axes[1], "d_norm_entropy", r"$\\Delta H_{norm}$ (norm entropy)"),
]:
    sub = df.sort_values("is_hallucination")  # OK first, HAL second
    mat = np.vstack(sub[col].values)
    n_ok_ = (~sub["is_hallucination"]).sum()
    vmax = np.percentile(np.abs(mat), 97)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.axhline(n_ok_ - 0.5, color="yellow", lw=1.8, ls="--")
    ax.set_yticks([n_ok_//2, n_ok_ + (len(sub)-n_ok_)//2])
    ax.set_yticklabels(["✓ Factual (OK)", "✗ Hallucinated"], fontsize=10)
    ax.set_xlabel("Layer index")
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label=title, shrink=0.7)

fig.suptitle("Hallucination Fingerprint — Δ per Sample × Layer", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_fingerprint_heatmap.png", bbox_inches="tight")
plt.show()
"""),

md("""\
## 9 — Token-Level Dynamics

Generation arrays have shape $[L, T]$. Plotting $X_\\ell[:, t]$ as a function of token position
reveals whether the energy signal evolves during generation (temporal drift) or is established immediately.
JS and Hellinger divergence measure how much the per-token generation distribution over memory patterns
deviates from the prefill distribution — a direct measure of memory-state shift.
"""),

code("""\
MAX_T = 50  # max_new_tokens from config

def pad_to(arr_1d, T=MAX_T):
    \"\"\"Pad a variable-length 1-D array to length T with NaN.\"\"\"
    out = np.full(T, np.nan)
    out[:len(arr_1d)] = arr_1d
    return out

def pad_2d_to(arr_2d, T=MAX_T):
    \"\"\"Pad [L, t] array to [L, T] with NaN along token axis.\"\"\"
    L, t = arr_2d.shape
    if t == T:
        return arr_2d
    out = np.full((L, T), np.nan)
    out[:, :t] = arr_2d
    return out

PROBE_LAYERS = [0, 8, 16, 24, 32, 35]

fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# 9a: mean gen_energy at selected layers vs token position
ax = axes[0]
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(PROBE_LAYERS)))
for li, (layer, color) in enumerate(zip(PROBE_LAYERS, cmap)):
    for lbl, ls in [("HAL", "-"), ("OK", "--")]:
        vals = np.vstack(
            df[df["label"] == lbl]["gen_energy"]
            .apply(lambda x: pad_to(x[layer]))
            .values
        )
        m = np.nanmean(vals, axis=0)
        ax.plot(m, color=color, ls=ls, lw=1.8, alpha=0.85,
                label=f"L{layer} {lbl}" if li < 3 else None)
ax.set_xlabel("Token position")
ax.set_ylabel("Gen energy")
ax.set_title("Generation Energy per Token — selected layers (solid=HAL, dashed=OK)")
ax.legend(fontsize=7, ncol=4)

# 9b: JS divergence per token (mean over layers), HAL vs OK
ax = axes[1]
for lbl, color in COLORS.items():
    mat = np.vstack(
        df[df["label"] == lbl]["gen_js"]
        .apply(lambda x: pad_to(np.nanmean(pad_2d_to(x), axis=0)))
        .values
    )
    m, s = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
    ax.plot(m, color=color, lw=2.3, label=lbl)
    ax.fill_between(range(MAX_T), m - s, m + s, color=color, alpha=0.15)
ax.set_xlabel("Token position")
ax.set_ylabel("JS divergence (mean over layers)")
ax.set_title("JS Divergence: Gen vs Prefill Distribution — per Token")
ax.legend()

# 9c: Hellinger per token
ax = axes[2]
for lbl, color in COLORS.items():
    mat = np.vstack(
        df[df["label"] == lbl]["gen_hellinger"]
        .apply(lambda x: pad_to(np.nanmean(pad_2d_to(x), axis=0)))
        .values
    )
    m, s = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
    ax.plot(m, color=color, lw=2.3, label=lbl)
    ax.fill_between(range(MAX_T), m - s, m + s, color=color, alpha=0.15)
ax.set_xlabel("Token position")
ax.set_ylabel("Hellinger distance (mean over layers)")
ax.set_title("Hellinger Distance: Gen vs Prefill Distribution — per Token")
ax.legend()

fig.suptitle("Token-Level Dynamics — Temporal Structure of Memory Shift", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_token_dynamics.png", bbox_inches="tight")
plt.show()
"""),

md("""\
## 10 — Logistic Probe

A logistic regression probe trained on the full per-layer feature vector tests whether a *linear*
combination of all layers can separate HAL from OK — measuring whether the signal is recoverable
even when individual-layer AUROC is modest.

Features tested:
1. `delta_energy` vector $[\\Delta E_0, \\ldots, \\Delta E_{35}]$
2. All delta metrics concatenated $[\\Delta E, \\Delta H, \\Delta H_{\\text{norm}}, \\Delta E_{\\text{lse}}, \\Delta$ top act$]$
3. JS layer vector $[JS_0, \\ldots, JS_{35}]$
4. Full feature set (all above + Hellinger)
"""),

code("""\
def probe_auroc(X, y, n_splits=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()

y = df["is_hallucination"].astype(int).values

feature_sets = {
    "Δ Energy  [L]":           np.vstack(df["d_energy"].values),
    "Δ Entropy  [L]":          np.vstack(df["d_entropy"].values),
    "Δ NormEntropy  [L]":      np.vstack(df["d_norm_entropy"].values),
    "JS  [L]":                 np.vstack(df["gen_js_layer"].values),
    "Hellinger  [L]":          np.vstack(df["gen_hell_layer"].values),
    "All Δ metrics  [5×L]":    np.hstack([
        np.vstack(df["d_energy"].values),
        np.vstack(df["d_entropy"].values),
        np.vstack(df["d_norm_entropy"].values),
        np.vstack(df["d_lse"].values),
        np.vstack(df["d_top_act"].values),
    ]),
    "All metrics + JS/Hell  [7×L]": np.hstack([
        np.vstack(df["d_energy"].values),
        np.vstack(df["d_entropy"].values),
        np.vstack(df["d_norm_entropy"].values),
        np.vstack(df["d_lse"].values),
        np.vstack(df["d_top_act"].values),
        np.vstack(df["gen_js_layer"].values),
        np.vstack(df["gen_hell_layer"].values),
    ]),
}

probe_results = {}
print(f"{'Feature set':<35}  {'AUROC mean':>11}  {'± std':>7}")
print("-" * 58)
for name, X in feature_sets.items():
    m, s = probe_auroc(X, y)
    probe_results[name] = (m, s)
    print(f"{name:<35}  {m:>11.4f}  ±{s:>6.4f}")
"""),

code("""\
names = list(probe_results.keys())
means = [probe_results[n][0] for n in names]
stds  = [probe_results[n][1] for n in names]

fig, ax = plt.subplots(figsize=(10, 4))
colors_bar = ["#3498db" if m < 0.65 else "#27ae60" for m in means]
bars = ax.barh(names, means, xerr=stds, color=colors_bar, alpha=0.85,
               capsize=4, error_kw={"lw": 1.5}, edgecolor="white")
ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="Random (0.5)")
ax.axvline(0.7, color="#e67e22", lw=1, ls=":", alpha=0.6, label="Practical threshold (0.7)")
for bar, m in zip(bars, means):
    ax.text(m + 0.005, bar.get_y() + bar.get_height()/2, f"{m:.4f}", va="center", fontsize=9)
ax.set_xlabel("5-fold CV AUROC")
ax.set_title("Logistic Probe AUROC — Feature Ablation  (Qwen2.5-3B Exp-04)", fontsize=11)
ax.legend(fontsize=8)
ax.set_xlim(0.4, max(means) + 0.06)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_probe_ablation.png", bbox_inches="tight")
plt.show()
"""),

md("## 11 — Summary Table"),

code("""\
y = df["is_hallucination"].astype(int).values

summary = []
for lbl, scol, lcol in [
    (r"$\\Delta E$",           "score_energy",       "d_energy"),
    (r"$\\Delta H$",           "score_entropy",      "d_entropy"),
    (r"$\\Delta H_{norm}$",    "score_norm_entropy", "d_norm_entropy"),
    (r"$\\Delta E_{lse}$",     "score_lse",          "d_lse"),
    (r"$\\Delta$ top act",     "score_top_act",      "d_top_act"),
    (r"JS",                    "score_js",           "gen_js_layer"),
    (r"Hellinger",             "score_hellinger",    "gen_hell_layer"),
]:
    auc_scalar = roc_auc_score(y, df[scol])
    auc_layer  = auroc_by_layer(df, lcol)
    bl = int(np.argmax(auc_layer))
    summary.append({
        "Metric": lbl,
        "Overall AUROC": f"{auc_scalar:.4f}",
        "Best layer": bl,
        "Best layer AUROC": f"{auc_layer[bl]:.4f}",
    })

summary_df = pd.DataFrame(summary).set_index("Metric")

display(summary_df)
full_probe = probe_results["All metrics + JS/Hell  [7×L]"][0]
print(f"\\nFull-feature probe (7xL): {full_probe:.4f}")
"""),

]

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 04
# ─────────────────────────────────────────────────────────────────────────────

MACRO_GROUPS = {
    "Factual/Historical":    ["Misconceptions", "History", "Myths and Fairytales",
                              "Proverbs", "Misquotations", "Fiction", "Religion",
                              "Misconceptions: Topical"],
    "Science/Health":        ["Health", "Science", "Nutrition", "Weather",
                              "Psychology", "Statistics"],
    "Social/Institutional":  ["Law", "Sociology", "Economics", "Politics",
                              "Education", "Finance", "Advertising"],
    "Identity/Confusion":    ["Confusion: People", "Confusion: Places", "Confusion: Other",
                              "Indexical Error: Other", "Indexical Error: Location",
                              "Indexical Error: Time", "Indexical Error: Identity"],
    "Belief/Conspiracy":     ["Conspiracies", "Paranormal", "Superstitions",
                              "Stereotypes", "Mandela Effect"],
    "Language/Logic":        ["Language", "Logical Falsehood", "Subjective",
                              "Distraction"],
}
MACRO_COLORS = {
    "Factual/Historical":    "#3498db",
    "Science/Health":        "#27ae60",
    "Social/Institutional":  "#e67e22",
    "Identity/Confusion":    "#e74c3c",
    "Belief/Conspiracy":     "#9b59b6",
    "Language/Logic":        "#1abc9c",
}

NB04_CELLS = [

md("""\
# Exp-04: Is the Hallucination Signal Topic-Dependent?

**Question:** Does Modern Hopfield energy discriminate hallucinated from factual responses
equally across all question categories, or is the signal concentrated in specific semantic domains?

**Why this matters:** If hallucination detection is topic-universal, a single detector generalises.
If it is topic-specific, detectors must be calibrated per domain — or we can at least identify
which topics are "high signal" and which are "low signal" for this class of energy features.

**Data:** Exp-04, Qwen2.5-3B, 500 TruthfulQA samples, 38 categories, 317 HAL / 183 OK.
"""),

code("""\
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, false_discovery_control
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

TRAJ_DIR  = Path("/home/brandon/energy_llm/exp04_results/trajectories")
LABEL_DIR = Path("/home/brandon/energy_llm/exp04_results/labels")
PLOTS_DIR = Path("/home/brandon/energy_llm/outputs/nb04_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
N_LAYERS = 36

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
COLORS = {"HAL": "#e74c3c", "OK": "#2ecc71"}

MACRO_GROUPS = """ + json.dumps(MACRO_GROUPS, indent=4) + """

MACRO_COLORS = """ + json.dumps(MACRO_COLORS, indent=4) + """

# Build reverse map: category → macro-group
cat_to_macro = {cat: macro for macro, cats in MACRO_GROUPS.items() for cat in cats}

print("Config ready.")
"""),

md("## 1 — Load Data"),

code("""\
rows = []
for jpath in sorted(TRAJ_DIR.glob("*.json")):
    if jpath.name.startswith("calibration"):
        continue
    meta = json.loads(jpath.read_text())
    sid  = meta["id"]
    npz_path   = jpath.with_suffix(".npz")
    label_path = LABEL_DIR / f"{sid}_label.json"
    if not npz_path.exists() or not label_path.exists():
        continue
    lbl = json.loads(label_path.read_text())
    npz = np.load(npz_path, allow_pickle=True)

    cat = meta.get("category", "Unknown")
    rows.append({
        "id":               sid,
        "category":         cat,
        "macro_group":      cat_to_macro.get(cat, "Other"),
        "is_hallucination": lbl["is_hallucination"],
        "pre_energy":       npz["prefill_energy"].astype(float),
        "pre_entropy":      npz["prefill_entropy"].astype(float),
        "gen_energy":       npz["gen_energy"].astype(float),
        "gen_entropy":      npz["gen_entropy"].astype(float),
        "gen_norm_entropy": npz["gen_norm_entropy"].astype(float),
        "gen_js":           npz["gen_js"].astype(float),
        "gen_hellinger":    npz["gen_hellinger"].astype(float),
    })

df = pd.DataFrame(rows)

for m in ["energy", "entropy"]:
    df[f"d_{m}"] = df.apply(lambda r: np.nanmean(r[f"gen_{m}"], axis=1) - r[f"pre_{m}"], axis=1)
    df[f"score_{m}"] = df[f"d_{m}"].apply(np.nanmean)

df["score_js"]       = df["gen_js"].apply(lambda x: np.nanmean(x))
df["score_hellinger"]= df["gen_hellinger"].apply(lambda x: np.nanmean(x))
df["gen_js_layer"]   = df["gen_js"].apply(lambda x: np.nanmean(x, axis=1))
df["label"] = df["is_hallucination"].map({True: "HAL", False: "OK"})

# Assign colors
df["macro_color"] = df["macro_group"].map(MACRO_COLORS).fillna("#95a5a6")

print(f"Loaded {len(df)} samples across {df['category'].nunique()} categories "
      f"and {df['macro_group'].nunique()} macro-groups.")
print(df.groupby("macro_group")["is_hallucination"].agg(n="count", hal=sum).assign(
    hal_rate=lambda d: (d["hal"]/d["n"]*100).round(1)))
"""),

md("## 2 — Category Distribution and Hallucination Rates"),

code("""\
cat_stats = (
    df.groupby("category")["is_hallucination"]
    .agg(n="count", hal=sum)
    .assign(hal_rate=lambda d: d["hal"]/d["n"]*100,
            macro=lambda d: d.index.map(lambda c: cat_to_macro.get(c, "Other")))
    .sort_values("hal_rate", ascending=True)
    .query("n >= 5")
)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: hallucination rate per category
ax = axes[0]
bar_colors = [MACRO_COLORS.get(cat_stats.loc[c, "macro"], "#95a5a6") for c in cat_stats.index]
bars = ax.barh(cat_stats.index, cat_stats["hal_rate"], color=bar_colors, edgecolor="white", alpha=0.85)
for bar, (_, row) in zip(bars, cat_stats.iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{row['hal_rate']:.0f}%  n={int(row['n'])}", va="center", fontsize=7.5)
ax.axvline(50, color="gray", ls="--", lw=0.9, alpha=0.5)
ax.axvline(df["is_hallucination"].mean()*100, color="#e74c3c", ls=":", lw=1.5,
           label=f"Global rate {df['is_hallucination'].mean()*100:.1f}%")
ax.set_xlabel("Hallucination rate (%)")
ax.set_title("Hallucination Rate by Category  (color = macro-group)")
ax.set_xlim(0, 120)
ax.legend(fontsize=8)

from matplotlib.patches import Patch
handles = [Patch(color=c, label=g) for g, c in MACRO_COLORS.items()]
ax.legend(handles=handles, fontsize=7.5, loc="lower right")

# Right: category count by macro-group
ax = axes[1]
macro_counts = df.groupby("macro_group").size().sort_values(ascending=True)
macro_hal    = df.groupby("macro_group")["is_hallucination"].mean() * 100
colors_m = [MACRO_COLORS.get(g, "#95a5a6") for g in macro_counts.index]
bars2 = ax.barh(macro_counts.index, macro_counts.values, color=colors_m, edgecolor="white", alpha=0.85)
for bar, g in zip(bars2, macro_counts.index):
    rate = macro_hal[g]
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"n={bar.get_width():.0f}  HAL={rate:.0f}%", va="center", fontsize=9)
ax.set_xlabel("Number of samples")
ax.set_title("Samples per Macro-Group")

fig.suptitle("TruthfulQA Category Overview — Exp-04 Qwen2.5-3B", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_category_distribution.png", bbox_inches="tight")
plt.show()
"""),

md("""\
## 3 — Energy Signal per Category

$\\overline{\\Delta E}^{\\text{HAL}} - \\overline{\\Delta E}^{\\text{OK}}$: a positive gap means hallucinated
answers retrieve with higher energy shift than factual ones in that category — the expected direction
of the hypothesis. A negative gap means the signal is **inverted** for that category.
"""),

code("""\
signal_records = []
for cat, grp in df.groupby("category"):
    if len(grp) < 5:
        continue
    hal = grp[grp["is_hallucination"] == True]["score_energy"].values
    ok  = grp[grp["is_hallucination"] == False]["score_energy"].values
    if len(hal) == 0 or len(ok) == 0:
        continue
    gap = hal.mean() - ok.mean()
    se  = np.sqrt(hal.std()**2/len(hal) + ok.std()**2/max(len(ok),1))
    stat, p = mannwhitneyu(hal, ok, alternative="two-sided") if (len(hal)>=3 and len(ok)>=3) else (np.nan, np.nan)
    signal_records.append({
        "category": cat,
        "macro": cat_to_macro.get(cat, "Other"),
        "n": len(grp), "n_hal": len(hal), "n_ok": len(ok),
        "hal_mean": hal.mean(), "ok_mean": ok.mean(),
        "gap": gap, "se": se, "mw_p": p,
        "hal_rate": len(hal)/len(grp)*100,
    })

sig_df = pd.DataFrame(signal_records).sort_values("gap")

# Benjamini-Hochberg correction
valid = sig_df["mw_p"].notna()
pvals_adj = false_discovery_control(sig_df.loc[valid, "mw_p"].values, method="bh")
sig_df.loc[valid, "p_adj"] = pvals_adj
sig_df["significant"] = sig_df["p_adj"] < 0.05

fig, ax = plt.subplots(figsize=(10, max(6, len(sig_df)*0.36)))

bar_colors = [MACRO_COLORS.get(sig_df.loc[i,"macro"], "#95a5a6") for i in sig_df.index]
edge_colors = ["black" if sig_df.loc[i,"significant"] else "none" for i in sig_df.index]
ax.barh(sig_df["category"], sig_df["gap"],
        xerr=sig_df["se"], color=bar_colors, edgecolor=edge_colors,
        linewidth=0.8, alpha=0.85, capsize=3, error_kw={"lw":1.2})
ax.axvline(0, color="black", lw=0.9)

from matplotlib.patches import Patch
handles = [Patch(color=c, label=g) for g, c in MACRO_COLORS.items()]
handles += [Patch(facecolor="white", edgecolor="black", label="p_BH < 0.05")]
ax.legend(handles=handles, fontsize=7.5, loc="lower right")
ax.set_xlabel(r"$\\overline{\\Delta E}^{HAL} - \\overline{\\Delta E}^{OK}$  (energy gap)")
ax.set_title("Hallucination Energy Signal per Category  (outline = BH sig., color = macro-group)", fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_energy_gap_per_category.png", bbox_inches="tight")
plt.show()

print(f"Categories with significant signal (p_BH < 0.05): "
      f"{sig_df['significant'].sum()} / {len(sig_df)}")
print("\\nTop 5 strongest signal:")
print(sig_df.nlargest(5, "gap")[["category","macro","gap","n","hal_rate","p_adj"]].to_string(index=False))
print("\\nTop 5 inverted signal:")
print(sig_df.nsmallest(5, "gap")[["category","macro","gap","n","hal_rate","p_adj"]].to_string(index=False))
"""),

md("## 4 — AUROC per Category × Metric Heatmap"),

code("""\
METRICS = {
    r"$\\Delta E$":        ("score_energy",    "d_energy"),
    r"$\\Delta H$":        ("score_entropy",   "d_entropy"),
    r"JS":                 ("score_js",        "gen_js_layer"),
    r"Hellinger":          ("score_hellinger", "gen_js_layer"),
}

auroc_rows = []
for cat, grp in df.groupby("category"):
    if grp["is_hallucination"].nunique() < 2 or len(grp) < 8:
        continue
    y = grp["is_hallucination"].astype(int).values
    row = {"category": cat, "macro": cat_to_macro.get(cat, "Other"), "n": len(grp)}
    for mname, (scol, _) in METRICS.items():
        try:
            row[mname] = roc_auc_score(y, grp[scol].values)
        except Exception:
            row[mname] = np.nan
    auroc_rows.append(row)

auroc_df = pd.DataFrame(auroc_rows)
# Sort by mean AUROC across metrics
metric_cols = list(METRICS.keys())
auroc_df["mean_auroc"] = auroc_df[metric_cols].mean(axis=1)
auroc_df = auroc_df.sort_values("mean_auroc", ascending=True)

pivot = auroc_df.set_index("category")[metric_cols]

fig, ax = plt.subplots(figsize=(9, max(5, len(pivot)*0.42)))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
               vmin=0.25, vmax=0.90, interpolation="nearest")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=10)
ax.set_yticks(range(len(pivot.index)))

# Color y-labels by macro-group
ax.set_yticklabels(pivot.index, fontsize=8)
for tick, cat in zip(ax.get_yticklabels(), pivot.index):
    tick.set_color(MACRO_COLORS.get(cat_to_macro.get(cat,"Other"), "#333"))

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if val < 0.35 or val > 0.80 else "black")

plt.colorbar(im, ax=ax, label="AUROC", shrink=0.5)
ax.set_title("AUROC per Category × Metric  (y-label color = macro-group)", fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_auroc_category_metric_heatmap.png", bbox_inches="tight")
plt.show()

print("Mean AUROC across categories:")
print(pivot.mean().round(4))
"""),

md("""\
## 5 — Layer Profiles by Macro-Group

Which layers carry the strongest hallucination signal for each semantic domain?
This reveals whether different topics "live" in different parts of the network.
"""),

code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
LAYERS = np.arange(N_LAYERS)

for ax, (macro, color) in zip(axes, MACRO_COLORS.items()):
    sub = df[df["macro_group"] == macro]
    if len(sub) < 5:
        ax.set_visible(False)
        continue
    for lbl, lcolor in COLORS.items():
        grp = sub[sub["label"] == lbl]
        if len(grp) == 0:
            continue
        mat = np.vstack(grp["d_energy"].values)
        m, s = mat.mean(0), mat.std(0)
        ax.plot(LAYERS, m, color=lcolor, lw=2.3, label=f"{lbl} (n={len(grp)})")
        ax.fill_between(LAYERS, m-s, m+s, color=lcolor, alpha=0.13)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.35)
    hal_rate = sub["is_hallucination"].mean()*100
    ax.set_title(f"{macro}\\n(n={len(sub)}, HAL={hal_rate:.0f}%)", fontsize=9.5, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\\Delta E$")
    ax.set_xticks(range(0, N_LAYERS, 6))
    ax.legend(fontsize=7.5)

fig.suptitle(r"$\\Delta E$ Layer Profile by Macro-Group  —  HAL vs OK  (mean ± 1 std)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_layer_profiles_by_macro.png", bbox_inches="tight")
plt.show()
"""),

md("## 6 — AUROC by Layer per Macro-Group"),

code("""\
fig, ax = plt.subplots(figsize=(13, 5))

for macro, color in MACRO_COLORS.items():
    sub = df[df["macro_group"] == macro]
    if sub["is_hallucination"].nunique() < 2 or len(sub) < 10:
        continue
    y = sub["is_hallucination"].astype(int).values
    mat = np.vstack(sub["d_energy"].values)
    auc_layers = np.array([
        roc_auc_score(y, mat[:, l]) if np.unique(mat[:, l]).size > 1 else 0.5
        for l in range(N_LAYERS)
    ])
    ax.plot(LAYERS, auc_layers, color=color, lw=2, label=macro, alpha=0.85)
    best = int(np.argmax(auc_layers))
    if auc_layers[best] > 0.65:
        ax.annotate(f"L{best}", (best, auc_layers[best]),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=7.5, color=color)

ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="Random")
ax.set_xlabel("Layer index")
ax.set_ylabel(r"AUROC  ($\\Delta E$ at layer $\\ell$)")
ax.set_title(r"Per-layer AUROC of $\\Delta E$ — by Macro-Group", fontsize=12)
ax.set_xticks(range(0, N_LAYERS, 2))
ax.set_ylim(0.25, 0.90)
ax.legend(fontsize=8.5, ncol=2)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_auroc_by_layer_macro.png", bbox_inches="tight")
plt.show()
"""),

md("## 7 — Signal Consistency: Does Signal Strength Correlate with Hallucination Rate?"),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: signal gap vs hal_rate
ax = axes[0]
for macro, color in MACRO_COLORS.items():
    sub = sig_df[sig_df["macro"] == macro]
    if len(sub) == 0:
        continue
    ax.scatter(sub["hal_rate"], sub["gap"],
               color=color, s=sub["n"]*3, alpha=0.75, label=macro, edgecolors="white", lw=0.5)
    for _, row in sub.iterrows():
        if abs(row["gap"]) > sig_df["gap"].abs().quantile(0.75) or row["significant"]:
            ax.annotate(row["category"][:14], (row["hal_rate"], row["gap"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6.5, alpha=0.8)

ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
ax.axvline(50, color="gray", lw=0.7, ls="--", alpha=0.4)
ax.set_xlabel("Hallucination rate (%)")
ax.set_ylabel(r"Energy gap  $\\overline{\\Delta E}^{HAL} - \\overline{\\Delta E}^{OK}$")
ax.set_title("Signal Gap vs Hallucination Base Rate\\n(circle size ∝ n)")
from matplotlib.patches import Patch
handles = [Patch(color=c, label=g) for g, c in MACRO_COLORS.items()]
ax.legend(handles=handles, fontsize=7, loc="upper left")

# Right: mean AUROC vs hal_rate
ax = axes[1]
for macro, color in MACRO_COLORS.items():
    sub = auroc_df[auroc_df["macro"] == macro]
    if len(sub) == 0:
        continue
    hal_rates = sub["category"].map(lambda c: cat_stats.loc[c, "hal_rate"] if c in cat_stats.index else np.nan)
    ax.scatter(hal_rates, sub["mean_auroc"],
               color=color, s=sub["n"]*3, alpha=0.75, label=macro, edgecolors="white", lw=0.5)
    for (_, row), hr in zip(sub.iterrows(), hal_rates):
        if not np.isnan(hr) and (row["mean_auroc"] > 0.62 or row["mean_auroc"] < 0.42):
            ax.annotate(row["category"][:14], (hr, row["mean_auroc"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6.5, alpha=0.8)

ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
ax.set_xlabel("Hallucination rate (%)")
ax.set_ylabel("Mean AUROC (across metrics)")
ax.set_title("Detection Difficulty vs Hallucination Base Rate\\n(categories near 50% HAL are hardest)")
ax.legend(handles=handles, fontsize=7, loc="lower right")

fig.suptitle("Signal Consistency Analysis — Exp-04 Qwen2.5-3B", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_signal_consistency.png", bbox_inches="tight")
plt.show()

from scipy.stats import pearsonr, spearmanr
valid_ = sig_df["significant"].notna()
r_pearson, p_pearson = pearsonr(sig_df["hal_rate"], sig_df["gap"])
r_spearman, p_spearman = spearmanr(sig_df["hal_rate"], sig_df["gap"])
print(f"HAL rate vs energy gap  — Pearson r={r_pearson:.3f} (p={p_pearson:.3f}), "
      f"Spearman r={r_spearman:.3f} (p={p_spearman:.3f})")
"""),

md("## 8 — Summary: Categories Ranked by Detectability"),

code("""\
summary_cat = sig_df.merge(
    auroc_df[["category","mean_auroc"] + metric_cols], on="category", how="left"
).sort_values("mean_auroc", ascending=False)

display_cols = ["category", "macro", "n", "hal_rate", "gap", "p_adj", "significant", "mean_auroc"] + metric_cols
print(summary_cat[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
"""),

code("""\
print("\\n=== HIGH SIGNAL CATEGORIES (mean AUROC > 0.60) ===")
high = summary_cat[summary_cat["mean_auroc"] > 0.60]
print(f"  {len(high)} categories | macro-groups: {high['macro'].value_counts().to_dict()}")

print("\\n=== LOW / INVERTED SIGNAL CATEGORIES (mean AUROC < 0.45) ===")
low = summary_cat[summary_cat["mean_auroc"] < 0.45]
print(f"  {len(low)} categories | macro-groups: {low['macro'].value_counts().to_dict()}")

print("\\n=== STATISTICALLY SIGNIFICANT SIGNAL ===")
sig = summary_cat[summary_cat["significant"] == True]
print(sig[["category","macro","gap","n","p_adj","mean_auroc"]].to_string(index=False))
"""),

]

# Write notebooks
nb3 = nb(NB03_CELLS)
nb4 = nb(NB04_CELLS)

# Fix cell IDs to be unique
import hashlib
def fix_ids(notebook):
    seen = set()
    for i, cell in enumerate(notebook["cells"]):
        cid = hashlib.md5(f"{i}-{cell['source'][:20]}".encode()).hexdigest()[:8]
        while cid in seen:
            cid = hashlib.md5(f"{i}-{cid}".encode()).hexdigest()[:8]
        seen.add(cid)
        cell["id"] = cid
    return notebook

nb3 = fix_ids(nb3)
nb4 = fix_ids(nb4)

p3 = OUT / "03_energy_analysis_exp04.ipynb"
p4 = OUT / "04_category_signal_exp04.ipynb"

p3.write_text(json.dumps(nb3, indent=1, ensure_ascii=False))
p4.write_text(json.dumps(nb4, indent=1, ensure_ascii=False))

print(f"Written: {p3}")
print(f"Written: {p4}")
print(f"  NB03 cells: {len(nb3['cells'])}")
print(f"  NB04 cells: {len(nb4['cells'])}")

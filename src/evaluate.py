"""Evaluation at two levels.

Level 1 - data quality:  did we find the labels we corrupted? (detection_metrics)
                         how clean is the training set now?    (label_quality)
Level 2 - model quality: did fixing the data improve the model? (classification_metrics)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# Fixed color per dataset variant (color follows the entity, never its rank).
DATASET_COLORS = {
    "clean": "#2a78d6",            # blue
    "noisy": "#eb6834",            # orange
    "cleaned_remove": "#1baf7a",   # aqua
    "cleaned_relabel": "#eda100",  # yellow
    "cleaned_hybrid": "#e87ba4",   # magenta
}
DATASET_LABELS = {
    "clean": "Clean (upper bound)",
    "noisy": "Noisy",
    "cleaned_remove": "Cleaned: remove",
    "cleaned_relabel": "Cleaned: relabel",
    "cleaned_hybrid": "Cleaned: hybrid",
}


#  #
# Level 2: model quality
#  #
def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy, precision, recall, F1 (positive class = 1), ROC-AUC and confusion counts."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


#  #
# Level 1: data quality
#  #
def detection_metrics(is_noisy, flagged) -> dict:
    """Compare Cleanlab's flags against the labels we actually corrupted.

    det_precision: of the flagged rows, how many were really corrupted?
    det_recall:    of the corrupted rows, how many were flagged?
    Undefined values (e.g. recall at 0% noise) are NaN, not 0.
    """
    is_noisy, flagged = np.asarray(is_noisy, bool), np.asarray(flagged, bool)
    tp = int((is_noisy & flagged).sum())
    n_flagged, n_true = int(flagged.sum()), int(is_noisy.sum())
    p = tp / n_flagged if n_flagged else np.nan
    r = tp / n_true if n_true else np.nan
    f1 = 2 * p * r / (p + r) if (n_flagged and n_true and (p + r) > 0) else np.nan
    return {
        "n_true_errors": n_true,
        "n_flagged": n_flagged,
        "n_flagged_correct": tp,
        "det_precision": round(p, 4) if p == p else np.nan,
        "det_recall": round(r, 4) if r == r else np.nan,
        "det_f1": round(f1, 4) if f1 == f1 else np.nan,
        "review_reduction": round(1 - n_flagged / len(flagged), 4) if len(flagged) else np.nan,
    }


def label_quality(df: pd.DataFrame) -> dict:
    """How noisy is a (possibly cleaned) training set? Needs `label` and `true_label`."""
    wrong = df["label"].to_numpy() != df["true_label"].to_numpy()
    return {
        "n_train": int(len(df)),
        "n_wrong_labels": int(wrong.sum()),
        "residual_noise": round(float(wrong.mean()), 4) if len(df) else np.nan,
    }


def recovery_rate(clean: float, noisy: float, cleaned: float) -> float:
    """Share of the performance lost to noise that cleaning won back.

    (cleaned - noisy) / (clean - noisy). 1.0 = fully recovered, 0 = no help,
    negative = cleaning hurt. NaN when noise caused no measurable loss.
    """
    lost = clean - noisy
    if not np.isfinite(lost) or lost <= 1e-6:
        return np.nan
    return round((cleaned - noisy) / lost, 4)


#  #
# Summaries
#  #
def with_noise_type(results: pd.DataFrame) -> pd.DataFrame:
    """Results written before noise types existed were all symmetric (0% rows = 'none')."""
    if "noise_type" in results.columns:
        return results
    return results.assign(noise_type=np.where(results["noise_rate"] == 0, "none", "symmetric"))


def summarize(results: pd.DataFrame, metrics=("accuracy", "f1"),
              by=("noise_type", "noise_rate", "dataset")) -> pd.DataFrame:
    """Mean ± std over seeds, as a readable table."""
    results = with_noise_type(results)
    g = results.groupby(list(by))[list(metrics)].agg(["mean", "std"])
    table = pd.DataFrame(index=g.index)
    for m in metrics:
        table[m] = [f"{mu:.4f} ± {0 if sd != sd else sd:.4f}" for mu, sd in zip(g[(m, "mean")], g[(m, "std")])]
    table["n_seeds"] = results.groupby(list(by)).size()
    return table.reset_index()


def recovery_table(results: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Recovery rate per noise type, noise level and cleaning strategy (seed-averaged)."""
    r = with_noise_type(results)
    clean = r[(r["noise_rate"] == 0) & (r["dataset"] == "clean")][metric]
    if clean.empty:
        return pd.DataFrame()
    clean_ref = clean.mean()
    mean = r[r["noise_rate"] > 0].groupby(["noise_type", "noise_rate", "dataset"])[metric].mean()
    rows = []
    for (ntype, rate), grp in mean.groupby(level=[0, 1]):
        g = grp.droplevel([0, 1])
        if "noisy" not in g.index:
            continue
        for col in [c for c in g.index if c.startswith("cleaned_")]:
            rows.append({
                "noise_type": ntype,
                "noise_rate": rate,
                "strategy": col.replace("cleaned_", ""),
                f"clean_{metric}": round(clean_ref, 4),
                f"noisy_{metric}": round(g["noisy"], 4),
                f"cleaned_{metric}": round(g[col], 4),
                "recovery_rate": recovery_rate(clean_ref, g["noisy"], g[col]),
            })
    return pd.DataFrame(rows)


#  #
# Figures
#  #
NOISE_TYPE_COLORS = {"symmetric": "#2a78d6", "instance": "#eb6834"}  # blue, orange
NOISE_TYPE_LABELS = {"symmetric": "Random (symmetric) noise", "instance": "Realistic (instance-dependent) noise"}


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _metric_name(metric: str) -> str:
    return metric.upper() if len(metric) <= 3 else metric.replace("_", " ")


def _clean_reference(ax, r: pd.DataFrame, metric: str) -> None:
    clean = r[(r["noise_rate"] == 0) & (r["dataset"] == "clean")][metric]
    if len(clean):
        y = clean.mean()
        ax.axhline(y, color="#52514e", linewidth=1.2, linestyle="--", zorder=1)
        ax.text(0.995, y, "Clean data (upper bound)  ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.5, color="#52514e")


def _save(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_metric_vs_noise(results: pd.DataFrame, metric: str = "f1", path: str | None = None,
                         noise_type: str = "symmetric"):
    """Line per dataset variant for ONE noise type: metric vs. noise rate, mean ± std over seeds."""
    import matplotlib.pyplot as plt

    r = with_noise_type(results)
    sub = r[r["noise_type"].isin([noise_type, "none"])]
    stats = sub.groupby(["dataset", "noise_rate"])[metric].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    _clean_reference(ax, r, metric)

    for name in [d for d in DATASET_LABELS if d != "clean" and d in set(stats.dataset)]:
        s = stats[stats.dataset == name].sort_values("noise_rate")
        x = s["noise_rate"] * 100
        ax.plot(x, s["mean"], color=DATASET_COLORS[name], linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=DATASET_LABELS[name], zorder=3)
        ax.fill_between(x, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                        color=DATASET_COLORS[name], alpha=0.15, linewidth=0, zorder=2)

    name = _metric_name(metric)
    ax.set_xticks(sorted(sub["noise_rate"].unique() * 100))
    ax.set_xlabel("Injected label noise (%)", color="#52514e")
    ax.set_ylabel(f"Test {name}", color="#52514e")
    ax.set_title(f"Same model, different data quality: test {name}\n{NOISE_TYPE_LABELS.get(noise_type, noise_type)}",
                 loc="left", fontsize=11, color="#0b0b0b")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    return _save(fig, path)


def plot_noise_type_comparison(results: pd.DataFrame, metric: str = "f1", path: str | None = None):
    """Which noise hurts more? Noisy-model metric vs. noise rate, one line per noise type."""
    import matplotlib.pyplot as plt

    r = with_noise_type(results)
    noisy = r[(r["dataset"] == "noisy")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    _clean_reference(ax, r, metric)
    for ntype in [t for t in NOISE_TYPE_COLORS if t in set(noisy.noise_type)]:
        s = noisy[noisy.noise_type == ntype].groupby("noise_rate")[metric].agg(["mean", "std"])
        x = s.index.to_numpy() * 100
        ax.plot(x, s["mean"], color=NOISE_TYPE_COLORS[ntype], linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=NOISE_TYPE_LABELS[ntype], zorder=3)
        ax.fill_between(x, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                        color=NOISE_TYPE_COLORS[ntype], alpha=0.15, linewidth=0, zorder=2)
    name = _metric_name(metric)
    if len(noisy):
        ax.set_xticks(sorted(noisy["noise_rate"].unique() * 100))
    ax.set_xlabel("Injected label noise (%)", color="#52514e")
    ax.set_ylabel(f"Test {name} (trained on noisy labels)", color="#52514e")
    ax.set_title(f"Which label noise hurts BERT more? (test {name})", loc="left", fontsize=11, color="#0b0b0b")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    return _save(fig, path)


def plot_detection(results: pd.DataFrame, path: str | None = None):
    """Cleanlab detection precision (solid) and recall (dashed) vs. noise rate, per noise type."""
    import matplotlib.pyplot as plt

    r = with_noise_type(results)
    det = (r[r["noise_rate"] > 0]
           .drop_duplicates(["noise_type", "noise_rate", "seed"])
           .groupby(["noise_type", "noise_rate"])[["det_precision", "det_recall"]].agg(["mean", "std"]))
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for ntype in [t for t in NOISE_TYPE_COLORS if t in det.index.get_level_values(0)]:
        d = det.loc[ntype]
        x = d.index.to_numpy() * 100
        for col, style, label in [("det_precision", "-", "precision"), ("det_recall", "--", "recall")]:
            mu, sd = d[(col, "mean")], d[(col, "std")].fillna(0)
            ax.plot(x, mu, color=NOISE_TYPE_COLORS[ntype], linewidth=2, linestyle=style, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=f"{ntype}: {label}", zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=NOISE_TYPE_COLORS[ntype], alpha=0.12, linewidth=0)
        ax.set_xticks(x)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Injected label noise (%)", color="#52514e")
    ax.set_ylabel("Score vs. known corrupted labels", color="#52514e")
    ax.set_title("How well Cleanlab finds the labels we corrupted", loc="left", fontsize=11, color="#0b0b0b")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2, handlelength=3.5)
    return _save(fig, path)


def plot_confusion(row: pd.Series | dict, title: str = "", path: str | None = None):
    """Confusion matrix from a results row (tn, fp, fn, tp)."""
    import matplotlib.pyplot as plt

    cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(3.8, 3.4), dpi=150)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#0b0b0b")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig

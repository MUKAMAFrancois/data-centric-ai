"""Cleaning stage: out-of-sample probabilities -> Cleanlab -> label resolution.

This module never trains BERT. It takes a feature matrix (frozen BERT embeddings
on Colab, TF-IDF features in the local dry run) and the noisy labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STRATEGIES = ("none", "remove", "relabel", "hybrid")


def get_oof_probs(features: np.ndarray, labels: np.ndarray, cv_folds: int = 5, seed: int = 42, C: float = 1.0) -> np.ndarray:
    """Out-of-fold predicted probabilities, shape (n, n_classes).

    Every row is scored by a model trained on the OTHER folds, so a wrong label
    can't be memorised. This is the input Cleanlab needs.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return cross_val_predict(clf, features, labels, cv=cv, method="predict_proba")


def find_issues(labels: np.ndarray, pred_probs: np.ndarray, filter_by: str = "prune_by_noise_rate") -> pd.DataFrame:
    """Run Cleanlab. Returns one row per example with:

    is_label_issue   Cleanlab thinks the given label is wrong
    label_quality    0..1, lower = more suspicious
    given_label      the (possibly noisy) label
    suggested_label  the model's most likely class
    suggested_prob   the model's probability for that class
    """
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    labels = np.asarray(labels)
    issues = find_label_issues(labels=labels, pred_probs=pred_probs, filter_by=filter_by, n_jobs=1)
    return pd.DataFrame({
        "is_label_issue": issues,
        "label_quality": get_label_quality_scores(labels, pred_probs),
        "given_label": labels,
        "suggested_label": pred_probs.argmax(axis=1),
        "suggested_prob": pred_probs.max(axis=1),
    })


def resolve_labels(df: pd.DataFrame, issues: pd.DataFrame, strategy: str, hybrid_threshold: float = 0.9) -> tuple[pd.DataFrame, dict]:
    """Build the cleaned training set from the flagged rows.

    none     keep everything as is
    remove   drop every flagged row
    relabel  replace a flagged row's label with the suggested label
    hybrid   relabel if suggested_prob >= hybrid_threshold, otherwise remove
             (stands in for "send to human review" in a real system)

    `df` and `issues` must be row-aligned. Returns (cleaned_df, stats).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if len(df) != len(issues):
        raise ValueError("df and issues must have the same number of rows")

    df = df.reset_index(drop=True).copy()
    flagged = issues["is_label_issue"].to_numpy(bool)
    # only relabel when the suggestion actually differs from the given label
    disagrees = flagged & (issues["suggested_label"].to_numpy() != df["label"].to_numpy())
    confident = issues["suggested_prob"].to_numpy() >= hybrid_threshold

    relabel_mask = np.zeros(len(df), bool)
    remove_mask = np.zeros(len(df), bool)
    if strategy == "remove":
        remove_mask = flagged
    elif strategy == "relabel":
        relabel_mask = disagrees
    elif strategy == "hybrid":
        relabel_mask = disagrees & confident
        remove_mask = flagged & ~relabel_mask

    df.loc[relabel_mask, "label"] = issues.loc[relabel_mask, "suggested_label"].to_numpy()
    cleaned = df.loc[~remove_mask].reset_index(drop=True)
    stats = {"n_relabeled": int(relabel_mask.sum()), "n_removed": int(remove_mask.sum())}

    if "true_label" in df.columns:  # only possible because we injected the noise ourselves
        true = df["true_label"].to_numpy()
        given = issues["given_label"].to_numpy()
        stats["n_fixed"] = int((relabel_mask & (given != true) & (df["label"].to_numpy() == true)).sum())
        stats["n_broken"] = int((relabel_mask & (given == true)).sum())
        stats["n_removed_correct"] = int((remove_mask & (given == true)).sum())
    return cleaned, stats


def save_flagged(df: pd.DataFrame, issues: pd.DataFrame, path: str, max_chars: int = 300) -> pd.DataFrame:
    """Save flagged rows (most suspicious first) for error analysis / human review."""
    flagged = pd.concat([df.reset_index(drop=True), issues.drop(columns="given_label")], axis=1)
    flagged = flagged[flagged["is_label_issue"]].sort_values("label_quality")
    flagged = flagged.assign(text=flagged["text"].str.slice(0, max_chars))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(path, index=False)
    return flagged

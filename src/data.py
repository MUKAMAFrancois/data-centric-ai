"""Data stage: load IMDb, profile it, and inject controlled label noise.

This module knows nothing about BERT or Cleanlab.

Column conventions used across the whole project:
    id          stable row id (position in the original HF split)
    text        review text
    label       the label the model trains on (noisy after inject_noise)
    true_label  the original, correct label (kept as ground truth)
    is_noisy    True if we flipped this row's label
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_NAMES = {0: "negative", 1: "positive"}
_HTML_BREAK = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_SPACES = re.compile(r"\s+")


#  #
# Loading
#  #
def normalize_text(text: str) -> str:
    """Light, label-independent cleanup: IMDb reviews contain '<br />' tags."""
    text = _HTML_BREAK.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def stratified_subset(df: pd.DataFrame, n: int | None, seed: int, label_col: str = "label") -> pd.DataFrame:
    """Return n rows with the same class balance as df (or df itself if n is None/too big)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    subset, _ = train_test_split(df, train_size=n, stratify=df[label_col], random_state=seed)
    return subset.sort_values("id").reset_index(drop=True)


def load_imdb(
    dataset_name: str = "stanfordnlp/imdb",
    train_subset: int | None = None,
    test_subset: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load IMDb train/test splits as DataFrames with columns [id, text, label]."""
    from datasets import load_dataset  # imported lazily so tests don't need network

    ds = load_dataset(dataset_name)
    frames = {}
    for split in ("train", "test"):
        df = ds[split].to_pandas()[["text", "label"]]
        df.insert(0, "id", np.arange(len(df)))
        df["text"] = df["text"].map(normalize_text)
        df["label"] = df["label"].astype(int)
        frames[split] = df

    train = stratified_subset(frames["train"], train_subset, seed)
    test = stratified_subset(frames["test"], test_subset, seed)
    return train, test


#  #
# Profiling
#  #
def profile_data(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> dict:
    """Basic data-quality profile: size, balance, missing/empty, duplicates, lengths."""
    text = df[text_col].fillna("")
    n_words = text.str.split().str.len()
    dup_mask = text.duplicated(keep=False)
    # duplicated texts that carry different labels = guaranteed label conflicts
    conflicts = df[dup_mask].groupby(text_col)[label_col].nunique()

    return {
        "n_rows": int(len(df)),
        "class_counts": {int(k): int(v) for k, v in df[label_col].value_counts().sort_index().items()},
        "positive_rate": round(float(df[label_col].mean()), 4),
        "missing_text": int(df[text_col].isna().sum()),
        "empty_text": int((text.str.strip() == "").sum()),
        "duplicate_rows": int(text.duplicated().sum()),
        "conflicting_duplicates": int((conflicts > 1).sum()),
        "words_mean": round(float(n_words.mean()), 1),
        "words_median": float(n_words.median()),
        "words_p95": float(n_words.quantile(0.95)),
        "words_max": int(n_words.max()),
    }


def split_overlap(train: pd.DataFrame, test: pd.DataFrame, text_col: str = "text") -> int:
    """Number of test reviews whose exact text also appears in train (leakage check)."""
    return int(test[text_col].isin(set(train[text_col])).sum())


#  #
# Controlled noise
#  #
def ambiguity_scores(texts, labels, cv_folds: int = 5, seed: int = 42) -> np.ndarray:
    """How hard each review is for a simple "annotator" model: 1 - P(true label).

    An out-of-fold TF-IDF + LogisticRegression model stands in for a human annotator.
    Reviews it struggles with (mixed sentiment, sarcasm, plot summaries) score high.
    It deliberately uses different features from the Cleanlab detector (frozen BERT),
    so the noise generator and the detector are independent.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    labels = np.asarray(labels)
    clf = make_pipeline(
        TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=4.0, max_iter=2000),
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    probs = cross_val_predict(clf, list(texts), labels, cv=cv, method="predict_proba")
    classes = np.unique(labels)  # predict_proba columns are the sorted classes
    p_true = probs[np.arange(len(labels)), np.searchsorted(classes, labels)]
    return 1.0 - p_true


NOISE_TYPES = ("symmetric", "instance")


def inject_noise(
    df: pd.DataFrame,
    noise_rate: float,
    seed: int,
    label_col: str = "label",
    noise_type: str = "symmetric",
    difficulty: np.ndarray | None = None,
) -> pd.DataFrame:
    """Flip exactly round(noise_rate * n_c) labels in every class c.

    noise_type="symmetric": rows to flip are chosen uniformly at random.
    noise_type="instance":  rows are chosen with probability proportional to
        `difficulty` (see ambiguity_scores), so ambiguous reviews are flipped far
        more often. This mimics real annotator mistakes, which cluster on hard
        examples, and is much harder for models and detectors than random flips.

    Flipping the same fraction per class keeps the class balance unchanged, so any
    performance change comes from wrong labels, not from a shifted class prior.
    The original label is preserved in `true_label`; flipped rows have is_noisy=True.
    The input DataFrame is never modified.
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
    classes = np.sort(df[label_col].unique())
    max_rate = 1 - 1 / max(len(classes), 2)
    if not 0.0 <= noise_rate < max_rate:
        raise ValueError(f"noise_rate must be in [0, {max_rate:.2f}) for {len(classes)} classes, got {noise_rate}")
    if noise_type == "instance":
        if difficulty is None or len(difficulty) != len(df):
            raise ValueError("instance noise needs a `difficulty` array with one score per row")
        weights_all = np.clip(np.asarray(difficulty, dtype=float), 0, None) + 1e-3  # every row stays possible

    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    true = out[label_col].to_numpy().copy()
    noisy = true.copy()

    for c in classes:
        idx = np.flatnonzero(true == c)
        n_flip = int(round(noise_rate * len(idx)))
        if n_flip == 0:
            continue
        if noise_type == "instance":
            w = weights_all[idx]
            chosen = rng.choice(idx, size=n_flip, replace=False, p=w / w.sum())
        else:
            chosen = rng.choice(idx, size=n_flip, replace=False)
        others = classes[classes != c]
        noisy[chosen] = rng.choice(others, size=n_flip)  # binary: always the other class

    out["true_label"] = true
    out[label_col] = noisy
    out["is_noisy"] = noisy != true
    return out

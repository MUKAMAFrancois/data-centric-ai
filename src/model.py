"""Model stage.

BERT (GPU, Colab):
    embed_texts            frozen BERT -> mean-pooled embeddings (for the cheap detector)
    fine_tune_and_predict  the FIXED classifier used in every final comparison

TF-IDF (CPU, local dry run):
    tfidf_features         stands in for embed_texts
    tfidf_train_predict    stands in for fine_tune_and_predict

Every "train_predict" function has the same signature:
    fn(train_df, test_df, model_cfg, seed) -> (y_pred, y_prob_positive)
so the pipeline can swap them without changing anything else.
torch / transformers are imported inside functions so this module imports fine without them.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd


#  #
# BERT: frozen embeddings for the detector
#  #
def embed_texts(
    texts: list[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    batch_size: int = 64,
    cache_path: str | None = None,
) -> np.ndarray:
    """Mean-pooled last-layer embeddings from a frozen (not fine-tuned) BERT.

    Frozen = never sees our labels, so it can't learn the noise. Cached to .npy
    because it's the slowest step of detection and is reused by every run.
    """
    if cache_path and Path(cache_path).exists():
        emb = np.load(cache_path)
        if len(emb) == len(texts):
            return emb

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # sort by length so each batch pads as little as possible, then restore order
    order = np.argsort([len(t) for t in texts])
    chunks = []
    with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
        for start in range(0, len(texts), batch_size):
            batch = [texts[i] for i in order[start:start + batch_size]]
            enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.float().cpu().numpy())

    emb = np.empty((len(texts), chunks[0].shape[1]), dtype=np.float32)
    emb[order] = np.concatenate(chunks)

    del model
    _free_gpu()
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb


#  #
# BERT: the fixed classifier
#  #
def fine_tune_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, model_cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fine-tune `model_cfg['name']` on train_df['label'] and predict test_df.

    Hyperparameters come only from model_cfg, so they are identical across runs.
    Nothing is saved to disk (no checkpoints): only predictions are returned.
    """
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(seed)  # also fixes the random init of the classification head
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["name"], num_labels=2)

    def to_ds(df: pd.DataFrame, with_labels: bool) -> Dataset:
        cols = ["text", "label"] if with_labels else ["text"]
        ds = Dataset.from_pandas(df[cols].reset_index(drop=True))
        ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=model_cfg["max_length"]),
                    batched=True, remove_columns=["text"])
        return ds.rename_column("label", "labels") if with_labels else ds

    # warmup as an integer step count: works on transformers 4.x and 5.x
    # (5.x removed warmup_ratio)
    total_steps = math.ceil(len(train_df) / model_cfg["batch_size"]) * model_cfg["epochs"]
    warmup_steps = int(model_cfg.get("warmup_ratio", 0.0) * total_steps)

    args = TrainingArguments(
        output_dir="outputs/tmp_trainer",
        num_train_epochs=model_cfg["epochs"],
        learning_rate=float(model_cfg["learning_rate"]),
        per_device_train_batch_size=model_cfg["batch_size"],
        per_device_eval_batch_size=model_cfg.get("eval_batch_size", 64),
        weight_decay=model_cfg.get("weight_decay", 0.0),
        warmup_steps=warmup_steps,
        fp16=bool(model_cfg.get("fp16", False)) and torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(train_df, True),
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()

    logits = trainer.predict(to_ds(test_df, False)).predictions
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

    del trainer, model
    _free_gpu()
    return probs.argmax(axis=1), probs[:, 1]


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


#  #
# CPU stand-ins for the local dry run
#  #
def tfidf_features(texts: list[str], n_components: int = 100, seed: int = 42) -> np.ndarray:
    """TF-IDF -> TruncatedSVD dense features (local stand-in for BERT embeddings)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    n_components = min(n_components, X.shape[1] - 1)
    return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(X).astype(np.float32)


def tfidf_train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, model_cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF + LogisticRegression classifier (local stand-in for fine-tuned BERT)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=model_cfg.get("C", 4.0), max_iter=2000, random_state=seed),
    )
    clf.fit(train_df["text"], train_df["label"])
    prob = clf.predict_proba(test_df["text"])[:, 1]
    return (prob >= 0.5).astype(int), prob

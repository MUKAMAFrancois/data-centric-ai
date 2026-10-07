"""Experiment loop: noise -> detect -> clean -> train the SAME model -> evaluate.

The notebook calls run_experiments(); it can also be run from the command line:

    python -m src.pipeline --mode dry-run      # local CPU check, TF-IDF stand-ins, ~2-5 min
    python -m src.pipeline --mode bert         # the real experiment (GPU: Colab/Kaggle)

Runs are resumable: every finished (noise_type, noise_rate, seed, dataset) is appended
to the results CSV, and anything already there is skipped on the next call.

Noise types (config: noise.types):
    symmetric  labels flipped uniformly at random
    instance   ambiguous reviews flipped far more often (like real annotator mistakes)
At 0% noise both types are identical, so those runs are stored once with noise_type="none".
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.cleaning import find_issues, get_oof_probs, resolve_labels, save_flagged
from src.data import ambiguity_scores, inject_noise, load_imdb, profile_data, split_overlap
from src.evaluate import (classification_metrics, detection_metrics, label_quality, plot_detection,
                          plot_metric_vs_noise, plot_noise_type_comparison, recovery_table, summarize)
from src.utils import append_result, load_config, run_exists, set_seed

RESULT_COLUMNS = [
    # what was run
    "noise_type", "noise_rate", "seed", "dataset", "strategy",
    # level 1: data quality
    "n_train", "n_wrong_labels", "residual_noise",
    "n_true_errors", "n_flagged", "n_flagged_correct", "det_precision", "det_recall", "det_f1",
    "review_reduction", "n_relabeled", "n_removed", "n_fixed", "n_broken", "n_removed_correct",
    # level 2: model quality (same test set every time)
    "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp",
    "train_seconds",
]

TrainFn = Callable[[pd.DataFrame, pd.DataFrame, dict, int], tuple]


def migrate_results(path: str) -> None:
    """Upgrade a results CSV written before noise types existed (adds `noise_type`).

    Old rows were all symmetric noise; their 0% rows become noise_type="none".
    """
    path = Path(path)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "noise_type" in df.columns:
        return
    df.insert(0, "noise_type", np.where(df["noise_rate"] == 0, "none", "symmetric"))
    df = df.reindex(columns=RESULT_COLUMNS)
    df.to_csv(path, index=False)


def issues_filename(noise_type: str, noise_rate: float, seed: int) -> str:
    return f"{noise_type}_noise{int(round(noise_rate * 100)):02d}_seed{seed}.csv"


def detect(train_df: pd.DataFrame, features: np.ndarray, noise_rate: float, seed: int, det_cfg: dict,
           noise_type: str = "symmetric", difficulty: np.ndarray | None = None):
    """Inject noise, run Cleanlab on out-of-fold probabilities, score the detection."""
    ntype = "symmetric" if noise_type == "none" else noise_type
    noisy = inject_noise(train_df, noise_rate, seed, noise_type=ntype, difficulty=difficulty)
    probs = get_oof_probs(features, noisy["label"].to_numpy(), det_cfg.get("cv_folds", 5), seed)
    issues = find_issues(noisy["label"].to_numpy(), probs, det_cfg.get("filter_by", "prune_by_noise_rate"))
    det = detection_metrics(noisy["is_noisy"], issues["is_label_issue"])
    return noisy, issues, det


def build_variants(noisy: pd.DataFrame, issues: pd.DataFrame, noise_rate: float, clean_cfg: dict) -> dict:
    """{dataset_name: (train_df, extra_info)}. At 0% noise the 'noisy' set IS the clean set."""
    base = "clean" if noise_rate == 0 else "noisy"
    variants = {base: (noisy, {"strategy": "none", "n_relabeled": 0, "n_removed": 0})}
    for strategy in clean_cfg.get("strategies", ["remove", "relabel"]):
        cleaned, stats = resolve_labels(noisy, issues, strategy, clean_cfg.get("hybrid_threshold", 0.9))
        variants[f"cleaned_{strategy}"] = (cleaned, {"strategy": strategy, **stats})
    return variants


def plan_jobs(noise_cfg: dict) -> list[tuple[str, float, int]]:
    """All (noise_type, noise_rate, seed) combinations; 0% noise appears once as 'none'."""
    types = noise_cfg.get("types", ["symmetric"])
    jobs = []
    for rate in noise_cfg["rates"]:
        for ntype in (["none"] if rate == 0 else types):
            for seed in noise_cfg["seeds"]:
                jobs.append((ntype, rate, seed))
    return jobs


def run_experiments(
    cfg: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: np.ndarray,
    train_fn: TrainFn,
    results_file: str,
    issues_dir: str | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Run every (noise_type, noise_rate, seed, dataset) combination that isn't logged yet."""
    assert len(features) == len(train_df), "features must be row-aligned with train_df"
    assert "true_label" not in test_df.columns, "the test set must never be noised"
    test_labels_before = test_df["label"].to_numpy().copy()
    strategies = cfg["cleaning"].get("strategies", [])
    migrate_results(results_file)

    jobs = plan_jobs(cfg["noise"])
    difficulty = None
    if any(ntype == "instance" for ntype, _, _ in jobs):
        log("[noise] scoring review ambiguity for instance-dependent noise ...")
        difficulty = ambiguity_scores(train_df["text"], train_df["label"], seed=cfg.get("seed", 42))

    for ntype, rate, seed in jobs:
        names = ["clean" if rate == 0 else "noisy"] + [f"cleaned_{s}" for s in strategies]
        keys = {"noise_type": ntype, "noise_rate": rate, "seed": seed}
        if all(run_exists(results_file, **keys, dataset=n) for n in names):
            log(f"[skip] {ntype} noise={rate:.0%} seed={seed}: already done")
            continue

        set_seed(seed)
        noisy, issues, det = detect(train_df, features, rate, seed, cfg["detection"], ntype, difficulty)
        log(f"[detect] {ntype} noise={rate:.0%} seed={seed}: flagged {det['n_flagged']} "
            f"(true errors {det['n_true_errors']}, precision {det['det_precision']}, recall {det['det_recall']})")
        if issues_dir:
            save_flagged(noisy, issues, f"{issues_dir}/{issues_filename(ntype, rate, seed)}")

        for name, (df, extra) in build_variants(noisy, issues, rate, cfg["cleaning"]).items():
            if run_exists(results_file, **keys, dataset=name):
                continue
            t0 = time.time()
            y_pred, y_prob = train_fn(df, test_df, cfg["model"], seed)
            row = {**keys, "dataset": name, **extra, **det,
                   **label_quality(df), **classification_metrics(test_df["label"], y_pred, y_prob),
                   "train_seconds": round(time.time() - t0, 1)}
            append_result(row, results_file, RESULT_COLUMNS)
            log(f"   [train] {name:<16} n={len(df):>6}  residual_noise={row['residual_noise']:.3f}  "
                f"acc={row['accuracy']:.4f}  f1={row['f1']:.4f}  ({row['train_seconds']}s)")

    assert np.array_equal(test_df["label"].to_numpy(), test_labels_before), "test labels changed!"
    return pd.read_csv(results_file)


def report(results: pd.DataFrame, figures_dir: str | None = None) -> None:
    """Print the summary tables and save the main figures."""
    pd.set_option("display.width", 160)
    print("\n=== Model quality (mean ± std over seeds) ===")
    print(summarize(results, metrics=("accuracy", "f1")).to_string(index=False))
    print("\n=== Recovery of F1 lost to noise ===")
    print(recovery_table(results, "f1").to_string(index=False))
    print("\n=== Detection quality vs. known corrupted labels ===")
    det = results[results.noise_rate > 0].drop_duplicates(["noise_type", "noise_rate", "seed"])
    print(det.groupby(["noise_type", "noise_rate"])[["n_true_errors", "n_flagged", "det_precision",
                                                     "det_recall", "det_f1"]].mean().round(3).to_string())
    if figures_dir:
        for ntype in sorted(set(results.noise_type) - {"none"}):
            plot_metric_vs_noise(results, "f1", f"{figures_dir}/f1_vs_noise_{ntype}.png", noise_type=ntype)
        plot_detection(results, f"{figures_dir}/detection_vs_noise.png")
        plot_noise_type_comparison(results, "f1", f"{figures_dir}/noise_type_comparison.png")
        print(f"\nFigures saved to {figures_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["dry-run", "bert"], default="dry-run")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-train", type=int, default=None, help="override data.train_subset")
    parser.add_argument("--n-test", type=int, default=None, help="override data.test_subset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    if args.mode == "dry-run":
        cfg["data"]["train_subset"] = args.n_train or 3000
        cfg["data"]["test_subset"] = args.n_test or 5000
        results_file, figures_dir, issues_dir = "results/dry_run.csv", "results/dry_run_figures", None
    else:
        cfg["data"]["train_subset"] = args.n_train or cfg["data"]["train_subset"]
        cfg["data"]["test_subset"] = args.n_test or cfg["data"]["test_subset"]
        results_file, figures_dir, issues_dir = paths["results_file"], paths["figures_dir"], paths["issues_dir"]

    set_seed(cfg["seed"])
    train_df, test_df = load_imdb(cfg["data"]["dataset_name"], cfg["data"]["train_subset"],
                                  cfg["data"]["test_subset"], cfg["seed"])
    print("train profile:", profile_data(train_df))
    print("test  profile:", profile_data(test_df))
    print("train/test exact-text overlap:", split_overlap(train_df, test_df))

    if args.mode == "dry-run":
        from src.model import tfidf_features, tfidf_train_predict

        features, train_fn = tfidf_features(train_df["text"].tolist(), seed=cfg["seed"]), tfidf_train_predict
    else:
        from src.model import embed_texts, fine_tune_and_predict

        det = cfg["detection"]
        cache = Path(paths["cache_dir"]) / f"emb_{det['embed_model'].replace('/', '_')}_{len(train_df)}_{det['embed_max_length']}.npy"
        features = embed_texts(train_df["text"].tolist(), det["embed_model"], det["embed_max_length"],
                               det.get("embed_batch_size", 64), str(cache))
        train_fn = fine_tune_and_predict

    results = run_experiments(cfg, train_df, test_df, features, train_fn, results_file, issues_dir)
    report(results, figures_dir)


if __name__ == "__main__":
    main()

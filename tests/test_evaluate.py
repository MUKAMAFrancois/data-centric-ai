import numpy as np
import pandas as pd
import pytest

from src.evaluate import (classification_metrics, detection_metrics, label_quality, recovery_rate,
                          recovery_table, summarize)


def test_classification_metrics_known_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = classification_metrics(y_true, y_pred, y_prob=[0.1, 0.6, 0.8, 0.9])
    assert m["accuracy"] == 0.75
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 0, 2)
    assert m["roc_auc"] == 1.0


def test_detection_metrics_known_values():
    is_noisy = [True, True, False, False, False]
    flagged = [True, False, True, False, False]
    d = detection_metrics(is_noisy, flagged)
    assert d["det_precision"] == 0.5
    assert d["det_recall"] == 0.5
    assert d["det_f1"] == 0.5
    assert d["n_flagged"] == 2 and d["n_true_errors"] == 2
    assert d["review_reduction"] == 0.6


def test_detection_metrics_no_true_errors_is_nan():
    d = detection_metrics([False] * 4, [True, False, False, False])
    assert d["det_precision"] == 0.0
    assert np.isnan(d["det_recall"])


def test_label_quality():
    df = pd.DataFrame({"label": [0, 1, 1, 0], "true_label": [0, 1, 0, 0]})
    q = label_quality(df)
    assert q["n_wrong_labels"] == 1 and q["residual_noise"] == 0.25


def test_recovery_rate():
    assert recovery_rate(0.92, 0.85, 0.90) == pytest.approx(0.7143, abs=1e-4)
    assert recovery_rate(0.92, 0.85, 0.85) == 0
    assert np.isnan(recovery_rate(0.90, 0.90, 0.91))


def test_summary_and_recovery_tables():
    rows = []
    for seed in (1, 2):
        rows += [
            {"noise_rate": 0.0, "seed": seed, "dataset": "clean", "accuracy": 0.92, "f1": 0.92},
            {"noise_rate": 0.1, "seed": seed, "dataset": "noisy", "accuracy": 0.85, "f1": 0.85},
            {"noise_rate": 0.1, "seed": seed, "dataset": "cleaned_relabel", "accuracy": 0.90, "f1": 0.90},
        ]
    res = pd.DataFrame(rows)
    assert len(summarize(res)) == 3
    rec = recovery_table(res, "f1")
    assert rec.loc[0, "recovery_rate"] == pytest.approx(0.7143, abs=1e-4)
    assert rec.loc[0, "noise_type"] == "symmetric"   # old results without noise_type still work


def test_figures_render(tmp_path):
    from src.evaluate import plot_detection, plot_metric_vs_noise, plot_noise_type_comparison

    rows = [{"noise_type": "none", "noise_rate": 0.0, "seed": 1, "dataset": "clean", "f1": 0.92,
             "det_precision": np.nan, "det_recall": np.nan}]
    for ntype, drop in [("symmetric", 0.01), ("instance", 0.05)]:
        for rate in (0.1, 0.2):
            for ds, f1 in [("noisy", 0.92 - drop * rate * 10), ("cleaned_remove", 0.915)]:
                rows.append({"noise_type": ntype, "noise_rate": rate, "seed": 1, "dataset": ds, "f1": f1,
                             "det_precision": 0.5, "det_recall": 0.7})
    res = pd.DataFrame(rows)
    plot_metric_vs_noise(res, "f1", str(tmp_path / "a.png"), noise_type="instance")
    plot_noise_type_comparison(res, "f1", str(tmp_path / "b.png"))
    plot_detection(res, str(tmp_path / "c.png"))
    assert len(list(tmp_path.glob("*.png"))) == 3

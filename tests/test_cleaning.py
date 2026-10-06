import numpy as np
import pandas as pd
import pytest

from src.cleaning import find_issues, get_oof_probs, resolve_labels
from src.data import inject_noise
from src.evaluate import detection_metrics, label_quality


@pytest.fixture
def noisy_blobs(blobs):
    X, y = blobs
    df = pd.DataFrame({"id": np.arange(len(y)), "text": "x", "label": y})
    noisy = inject_noise(df, 0.2, seed=42)
    probs = get_oof_probs(X, noisy.label.to_numpy(), cv_folds=5, seed=42)
    issues = find_issues(noisy.label.to_numpy(), probs)
    return noisy, issues, probs


def test_oof_probs_shape_and_range(noisy_blobs):
    _, _, probs = noisy_blobs
    assert probs.shape[1] == 2
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_cleanlab_finds_injected_errors(noisy_blobs):
    noisy, issues, _ = noisy_blobs
    d = detection_metrics(noisy.is_noisy, issues.is_label_issue)
    # well-separated blobs -> detection should be very good
    assert d["det_recall"] > 0.8
    assert d["det_precision"] > 0.8


def test_remove_drops_flagged_rows(noisy_blobs):
    noisy, issues, _ = noisy_blobs
    cleaned, stats = resolve_labels(noisy, issues, "remove")
    assert len(cleaned) == len(noisy) - issues.is_label_issue.sum()
    assert stats["n_removed"] == issues.is_label_issue.sum()
    assert label_quality(cleaned)["residual_noise"] < label_quality(noisy)["residual_noise"]


def test_relabel_keeps_size_and_reduces_noise(noisy_blobs):
    noisy, issues, _ = noisy_blobs
    cleaned, stats = resolve_labels(noisy, issues, "relabel")
    assert len(cleaned) == len(noisy)
    assert stats["n_fixed"] > stats["n_broken"]
    assert label_quality(cleaned)["residual_noise"] < label_quality(noisy)["residual_noise"]


def test_hybrid_and_none(noisy_blobs):
    noisy, issues, _ = noisy_blobs
    hybrid, s = resolve_labels(noisy, issues, "hybrid", hybrid_threshold=0.9)
    assert s["n_relabeled"] + s["n_removed"] == issues.is_label_issue.sum()
    same, s0 = resolve_labels(noisy, issues, "none")
    assert same.label.equals(noisy.label) and s0["n_removed"] == 0


def test_resolve_does_not_mutate_input(noisy_blobs):
    noisy, issues, _ = noisy_blobs
    before = noisy.copy()
    resolve_labels(noisy, issues, "relabel")
    pd.testing.assert_frame_equal(noisy, before)


def test_bad_strategy_raises(noisy_blobs):
    noisy, issues, _ = noisy_blobs
    with pytest.raises(ValueError):
        resolve_labels(noisy, issues, "delete_everything")

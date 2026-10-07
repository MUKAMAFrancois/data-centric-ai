import numpy as np
import pandas as pd
import pytest

from src.data import inject_noise, normalize_text, profile_data, split_overlap, stratified_subset


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.2, 0.3])
def test_flips_exact_fraction_per_class(reviews, rate):
    noisy = inject_noise(reviews, rate, seed=42)
    for c in (0, 1):
        n_c = (reviews.label == c).sum()
        flipped = (noisy.is_noisy & (noisy.true_label == c)).sum()
        assert flipped == round(rate * n_c)


def test_class_balance_preserved(reviews):
    noisy = inject_noise(reviews, 0.2, seed=42)
    assert noisy.label.value_counts().to_dict() == reviews.label.value_counts().to_dict()


def test_true_label_kept_and_flags_consistent(reviews):
    noisy = inject_noise(reviews, 0.2, seed=42)
    assert np.array_equal(noisy.true_label, reviews.label)
    assert np.array_equal(noisy.is_noisy, noisy.label != noisy.true_label)


def test_zero_noise_changes_nothing(reviews):
    noisy = inject_noise(reviews, 0.0, seed=42)
    assert not noisy.is_noisy.any()
    assert np.array_equal(noisy.label, reviews.label)


def test_reproducible_and_seed_dependent(reviews):
    a = inject_noise(reviews, 0.2, seed=42)
    b = inject_noise(reviews, 0.2, seed=42)
    c = inject_noise(reviews, 0.2, seed=7)
    assert a.equals(b)
    assert not np.array_equal(a.is_noisy, c.is_noisy)


def test_input_not_mutated(reviews):
    before = reviews.copy()
    inject_noise(reviews, 0.3, seed=42)
    pd.testing.assert_frame_equal(reviews, before)


@pytest.mark.parametrize("rate", [-0.1, 0.5, 0.9])
def test_invalid_rate_raises(reviews, rate):
    with pytest.raises(ValueError):
        inject_noise(reviews, rate, seed=42)


def test_normalize_text():
    assert normalize_text("Great film.<br /><br />Loved   it") == "Great film. Loved it"


def test_stratified_subset_keeps_balance(reviews):
    sub = stratified_subset(reviews, 100, seed=0)
    assert len(sub) == 100
    assert sub.label.mean() == pytest.approx(reviews.label.mean(), abs=0.02)
    assert stratified_subset(reviews, None, seed=0).shape == reviews.shape


def test_profile_and_overlap():
    df = pd.DataFrame({"id": range(4), "text": ["a b", "a b", "c", "d e f"], "label": [0, 1, 1, 0]})
    p = profile_data(df)
    assert p["n_rows"] == 4
    assert p["duplicate_rows"] == 1
    assert p["conflicting_duplicates"] == 1   # "a b" labelled both 0 and 1
    assert split_overlap(df, pd.DataFrame({"text": ["c", "zzz"]})) == 1


# --------------------------------------------------------------------------- #
# Instance-dependent noise
# --------------------------------------------------------------------------- #
from src.data import ambiguity_scores


def test_ambiguity_scores_range(reviews):
    d = ambiguity_scores(reviews.text, reviews.label, cv_folds=3, seed=0)
    assert d.shape == (len(reviews),)
    assert ((d >= 0) & (d <= 1)).all()


@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3])
def test_instance_noise_exact_count_and_balance(reviews, rate):
    difficulty = np.random.default_rng(0).random(len(reviews))
    noisy = inject_noise(reviews, rate, seed=42, noise_type="instance", difficulty=difficulty)
    for c in (0, 1):
        n_c = (reviews.label == c).sum()
        assert (noisy.is_noisy & (noisy.true_label == c)).sum() == round(rate * n_c)
    assert noisy.label.value_counts().to_dict() == reviews.label.value_counts().to_dict()


def test_instance_noise_targets_hard_examples(reviews):
    # half the rows are "hard" (difficulty 0.9), half "easy" (0.01)
    difficulty = np.where(np.arange(len(reviews)) % 4 < 2, 0.9, 0.01)
    noisy = inject_noise(reviews, 0.2, seed=42, noise_type="instance", difficulty=difficulty)
    hard_share = (difficulty[noisy.is_noisy.to_numpy()] == 0.9).mean()
    assert hard_share > 0.9          # random noise would give ~0.5


def test_instance_noise_reproducible(reviews):
    difficulty = np.random.default_rng(0).random(len(reviews))
    a = inject_noise(reviews, 0.2, seed=1, noise_type="instance", difficulty=difficulty)
    b = inject_noise(reviews, 0.2, seed=1, noise_type="instance", difficulty=difficulty)
    assert a.equals(b)


def test_instance_noise_requires_difficulty(reviews):
    with pytest.raises(ValueError):
        inject_noise(reviews, 0.2, seed=1, noise_type="instance")
    with pytest.raises(ValueError):
        inject_noise(reviews, 0.2, seed=1, noise_type="instance", difficulty=np.ones(3))
    with pytest.raises(ValueError):
        inject_noise(reviews, 0.2, seed=1, noise_type="adversarial")

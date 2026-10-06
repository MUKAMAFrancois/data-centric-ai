"""Shared fixtures: tiny synthetic data so tests run offline in seconds."""
import numpy as np
import pandas as pd
import pytest

POS = ["great", "excellent", "wonderful", "loved", "brilliant", "superb", "enjoyable", "masterpiece"]
NEG = ["awful", "terrible", "boring", "hated", "dull", "waste", "worst", "mess"]
NEUTRAL = ["movie", "film", "plot", "actor", "scene", "story", "director", "ending", "the", "a", "was"]


def make_reviews(n: int, seed: int = 0) -> pd.DataFrame:
    """Balanced fake reviews: sentiment words + filler words."""
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2))
    texts = []
    for y in labels:
        words = list(rng.choice(POS if y else NEG, size=4)) + list(rng.choice(NEUTRAL, size=8))
        rng.shuffle(words)
        texts.append(" ".join(words))
    return pd.DataFrame({"id": np.arange(n), "text": texts, "label": labels})


@pytest.fixture
def reviews():
    return make_reviews(400, seed=0)


@pytest.fixture
def test_reviews():
    return make_reviews(200, seed=1)


@pytest.fixture
def blobs():
    """Two well-separated Gaussian blobs: (features, labels)."""
    rng = np.random.default_rng(0)
    n = 600
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, 10)) + y[:, None] * 3.0
    return X.astype(np.float32), y

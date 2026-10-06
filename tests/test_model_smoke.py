"""Optional BERT smoke test with a tiny model. Skipped unless you opt in:

    pip install torch transformers
    RUN_SLOW=1 pytest tests/test_model_smoke.py        (Windows: set RUN_SLOW=1)
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run")
TINY = "prajjwal1/bert-tiny"  # 4M params: runs on CPU in seconds


def test_embed_texts(tmp_path, reviews):
    pytest.importorskip("torch")
    from src.model import embed_texts

    texts = reviews.text.tolist()[:20]
    cache = tmp_path / "emb.npy"
    emb = embed_texts(texts, TINY, max_length=32, batch_size=8, cache_path=str(cache))
    assert emb.shape[0] == 20 and cache.exists()
    assert np.allclose(emb, embed_texts(texts, TINY, cache_path=str(cache)))  # cache hit


def test_fine_tune_and_predict(reviews, test_reviews):
    pytest.importorskip("transformers")
    from src.model import fine_tune_and_predict

    cfg = {"name": TINY, "max_length": 32, "learning_rate": 5e-4, "epochs": 1, "batch_size": 16, "fp16": False}
    y_pred, y_prob = fine_tune_and_predict(reviews.head(64), test_reviews.head(32), cfg, seed=0)
    assert y_pred.shape == (32,) and ((y_prob >= 0) & (y_prob <= 1)).all()

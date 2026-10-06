"""End-to-end test of the experiment loop on fake reviews with the TF-IDF stand-ins."""
import pandas as pd

from src.model import tfidf_features, tfidf_train_predict
from src.pipeline import RESULT_COLUMNS, run_experiments

CFG = {
    "noise": {"rates": [0.0, 0.2], "seeds": [1, 2]},
    "detection": {"cv_folds": 3, "filter_by": "prune_by_noise_rate"},
    "cleaning": {"strategies": ["remove", "relabel"], "hybrid_threshold": 0.9},
    "model": {"C": 4.0},
}


def test_full_loop_and_resume(tmp_path, reviews, test_reviews):
    out = tmp_path / "results.csv"
    features = tfidf_features(reviews.text.tolist(), n_components=20)
    test_before = test_reviews.copy()

    res = run_experiments(CFG, reviews, test_reviews, features, tfidf_train_predict, str(out),
                          issues_dir=str(tmp_path / "issues"), log=lambda *_: None)

    # 2 rates x 2 seeds x 3 datasets
    assert len(res) == 12
    assert list(res.columns) == RESULT_COLUMNS
    assert set(res.dataset) == {"clean", "noisy", "cleaned_remove", "cleaned_relabel"}
    assert (res[res.dataset == "noisy"].residual_noise == 0.2).all()
    assert (res[res.dataset == "cleaned_relabel"].residual_noise < 0.2).all()
    pd.testing.assert_frame_equal(test_reviews, test_before)  # test set untouched
    assert len(list((tmp_path / "issues").glob("*.csv"))) == 4

    # second call must skip everything (resume after a Colab disconnect)
    calls = []
    def spy(*a):
        calls.append(1)
        return tfidf_train_predict(*a)
    res2 = run_experiments(CFG, reviews, test_reviews, features, spy, str(out), log=lambda *_: None)
    assert len(res2) == 12 and not calls

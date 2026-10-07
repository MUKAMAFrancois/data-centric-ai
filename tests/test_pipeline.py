"""End-to-end test of the experiment loop on fake reviews with the TF-IDF stand-ins."""
import numpy as np
import pandas as pd

from src.model import tfidf_features, tfidf_train_predict
from src.pipeline import RESULT_COLUMNS, migrate_results, plan_jobs, run_experiments

CFG = {
    "seed": 42,
    "noise": {"types": ["symmetric", "instance"], "rates": [0.0, 0.2], "seeds": [1, 2]},
    "detection": {"cv_folds": 3, "filter_by": "prune_by_noise_rate"},
    "cleaning": {"strategies": ["remove", "relabel"], "hybrid_threshold": 0.9},
    "model": {"C": 4.0},
}


def test_plan_jobs_runs_zero_noise_once():
    jobs = plan_jobs(CFG["noise"])
    # 0%: 'none' x 2 seeds;  20%: 2 types x 2 seeds
    assert len(jobs) == 6
    assert {j[0] for j in jobs if j[1] == 0} == {"none"}


def test_full_loop_and_resume(tmp_path, reviews, test_reviews):
    out = tmp_path / "results.csv"
    features = tfidf_features(reviews.text.tolist(), n_components=20)
    test_before = test_reviews.copy()

    res = run_experiments(CFG, reviews, test_reviews, features, tfidf_train_predict, str(out),
                          issues_dir=str(tmp_path / "issues"), log=lambda *_: None)

    # 6 (type, rate, seed) jobs x 3 datasets each
    assert len(res) == 18
    assert list(res.columns) == RESULT_COLUMNS
    assert set(res.noise_type) == {"none", "symmetric", "instance"}
    assert set(res.dataset) == {"clean", "noisy", "cleaned_remove", "cleaned_relabel"}
    assert (res[res.dataset == "noisy"].residual_noise == 0.2).all()
    pd.testing.assert_frame_equal(test_reviews, test_before)  # test set untouched
    assert len(list((tmp_path / "issues").glob("*.csv"))) == 6

    # second call must skip everything (resume after a Colab disconnect)
    calls = []
    def spy(*a):
        calls.append(1)
        return tfidf_train_predict(*a)
    res2 = run_experiments(CFG, reviews, test_reviews, features, spy, str(out), log=lambda *_: None)
    assert len(res2) == 18 and not calls


def test_migrate_old_results_keeps_pilot_runs(tmp_path, reviews, test_reviews):
    """A CSV from before noise types existed is upgraded, and its runs are not repeated."""
    out = tmp_path / "results.csv"
    old_cols = [c for c in RESULT_COLUMNS if c != "noise_type"]
    old = pd.DataFrame([
        {**{c: np.nan for c in old_cols}, "noise_rate": 0.0, "seed": 1, "dataset": "clean", "f1": 0.9},
        {**{c: np.nan for c in old_cols}, "noise_rate": 0.2, "seed": 1, "dataset": "noisy", "f1": 0.8},
    ])[old_cols]
    old.to_csv(out, index=False)

    migrate_results(str(out))
    new = pd.read_csv(out)
    assert list(new.columns) == RESULT_COLUMNS
    assert list(new.noise_type) == ["none", "symmetric"]

    cfg = {**CFG, "noise": {"types": ["symmetric"], "rates": [0.0, 0.2], "seeds": [1]},
           "cleaning": {"strategies": []}}
    calls = []
    def spy(*a):
        calls.append(1)
        return tfidf_train_predict(*a)
    features = tfidf_features(reviews.text.tolist(), n_components=20)
    res = run_experiments(cfg, reviews, test_reviews, features, spy, str(out), log=lambda *_: None)
    assert len(res) == 2 and not calls   # both runs were already done

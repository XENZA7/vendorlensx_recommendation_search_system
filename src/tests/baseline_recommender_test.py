import numpy as np
import pandas as pd

from src.models.baseline_recommender import BaselineRecommender


def make_baseline(vendors, sim_matrix):
    df = pd.DataFrame({
        "id": list(range(len(vendors))),
        "vendor": vendors,
        "title": [f"Product {i}" for i in range(len(vendors))],
    })
    baseline = BaselineRecommender.__new__(BaselineRecommender)
    baseline.df = df
    baseline.sim_matrix = np.array(sim_matrix, dtype=float)
    return baseline


def test_baseline_excludes_self():
    vendors = ["A", "A", "B", "B"]
    sim = np.array([
        [1.0, 0.9, 0.5, 0.3],
        [0.9, 1.0, 0.4, 0.2],
        [0.5, 0.4, 1.0, 0.6],
        [0.3, 0.2, 0.6, 1.0],
    ])
    baseline = make_baseline(vendors, sim)

    result = baseline.recommend(product_id=0, n=3)
    assert 0 not in result["id"].values


def test_baseline_ranks_by_raw_similarity_no_penalty():
    # Same-vendor, high-similarity product should NOT be penalised here —
    # this is exactly what distinguishes BaselineRecommender from Recommender.
    vendors = ["A", "A", "B"]
    sim = np.array([
        [1.0, 0.95, 0.10],   # product 0's similarities
        [0.95, 1.0, 0.05],
        [0.10, 0.05, 1.0],
    ])
    baseline = make_baseline(vendors, sim)

    result = baseline.recommend(product_id=0, n=2)
    # Product 1 (same vendor, 0.95 similarity) should rank first — no penalty applied
    assert result.iloc[0]["id"] == 1


def test_baseline_can_return_single_vendor_dominated_results():
    # Unlike Recommender, BaselineRecommender has no diversity guarantee —
    # if the top matches are all one vendor, that's exactly what should come back.
    vendors = ["A", "A", "A", "B"]
    sim = np.array([
        [1.0, 0.9, 0.8, 0.1],
        [0.9, 1.0, 0.7, 0.1],
        [0.8, 0.7, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ])
    baseline = make_baseline(vendors, sim)

    result = baseline.recommend(product_id=0, n=2)
    assert set(result["vendor"]) == {"A"}


def test_baseline_returns_empty_dataframe_for_unknown_product_id():
    vendors = ["A", "B"]
    baseline = make_baseline(vendors, np.eye(2))
    result = baseline.recommend(product_id=999, n=3)
    assert result.empty


def test_baseline_respects_n():
    vendors = ["A", "B", "C", "D", "E"]
    baseline = make_baseline(vendors, np.eye(5) + 0.1)
    result = baseline.recommend(product_id=0, n=2)
    assert len(result) == 2
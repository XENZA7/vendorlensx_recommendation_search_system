import numpy as np
import pandas as pd
import pytest

from src.models.recommender import Recommender


def make_recommender(vendors, sim_matrix):
    """Build a Recommender against a synthetic in-memory dataset — no CSV/joblib needed."""
    df = pd.DataFrame({
        "id": list(range(len(vendors))),
        "vendor": vendors,
        "title": [f"Product {i}" for i in range(len(vendors))],
    })
    rec = Recommender.__new__(Recommender)
    rec.df = df
    rec.sim_matrix = np.array(sim_matrix, dtype=float)
    return rec


# ── apply_bias_penalty ────────────────────────────────────────────────────────

def test_bias_penalty_applies_only_to_same_vendor_high_similarity():
    vendors = ["A", "A", "B", "B"]
    rec = make_recommender(vendors, np.eye(4))
    scores = np.array([1.0, 0.9, 0.85, 0.5])

    penalised = rec.apply_bias_penalty(scores, source_vendor="A")

    # index 1 is vendor A (same as source) with similarity > 0.8 -> penalised
    assert penalised[1] == pytest.approx(0.9 * 0.75)
    # index 2 is vendor B (different vendor) with similarity > 0.8 -> untouched
    assert penalised[2] == pytest.approx(0.85)
    # index 3 is vendor B with similarity <= 0.8 -> untouched regardless of vendor
    assert penalised[3] == pytest.approx(0.5)


def test_bias_penalty_does_not_mutate_input_array():
    vendors = ["A", "A"]
    rec = make_recommender(vendors, np.eye(2))
    original = np.array([1.0, 0.95])
    scores_copy = original.copy()

    rec.apply_bias_penalty(original, source_vendor="A")

    # Original array must be untouched — apply_bias_penalty should return a copy
    assert np.array_equal(original, scores_copy)


def test_bias_penalty_ignores_products_at_or_below_threshold():
    vendors = ["A", "A"]
    rec = make_recommender(vendors, np.eye(2))
    scores = np.array([1.0, 0.80])  # exactly 0.80 -> not > 0.8, should not be penalised

    penalised = rec.apply_bias_penalty(scores, source_vendor="A")
    assert penalised[1] == pytest.approx(0.80)


# ── enforce_diversity ──────────────────────────────────────────────────────────

def test_enforce_diversity_guarantees_two_vendors_when_possible():
    # Ranked order (best to worst) all point to vendor A except the last two
    vendors = ["A", "A", "A", "A", "B", "C"]
    rec = make_recommender(vendors, np.eye(6))
    ranked_indices = np.array([0, 1, 2, 3, 4, 5])  # indices 0-3 = A, 4 = B, 5 = C

    result = rec.enforce_diversity(ranked_indices, source_vendor="A", n=5)
    result_vendors = {vendors[i] for i in result}

    assert len(result) == 5
    assert len(result_vendors) >= 2  # diversity guarantee held


def test_enforce_diversity_fills_short_results_when_diversity_impossible():
    # Every candidate is the same vendor as the source — diversity is impossible,
    # but we must still return exactly n results (this was BUG: used to return n-1).
    vendors = ["A"] * 10
    rec = make_recommender(vendors, np.eye(10))
    ranked_indices = np.arange(10)

    result = rec.enforce_diversity(ranked_indices, source_vendor="A", n=5)

    assert len(result) == 5  # must not silently return fewer than n


def test_enforce_diversity_respects_n():
    vendors = ["A", "B", "A", "B", "A", "B"]
    rec = make_recommender(vendors, np.eye(6))
    ranked_indices = np.arange(6)

    for n in (1, 3, 6):
        result = rec.enforce_diversity(ranked_indices, source_vendor="A", n=n)
        assert len(result) == n


# ── recommend (integration of both) ────────────────────────────────────────────

def test_recommend_excludes_self():
    vendors = ["A", "A", "B", "B"]
    sim = np.array([
        [1.0, 0.9, 0.5, 0.3],
        [0.9, 1.0, 0.4, 0.2],
        [0.5, 0.4, 1.0, 0.6],
        [0.3, 0.2, 0.6, 1.0],
    ])
    rec = make_recommender(vendors, sim)

    result = rec.recommend(product_id=0, n=3)
    assert 0 not in result["id"].values


def test_recommend_returns_n_results_when_catalog_allows():
    vendors = ["A", "B", "C", "D", "E", "F"]
    rec = make_recommender(vendors, np.eye(6) + 0.1)  # small off-diagonal similarity
    result = rec.recommend(product_id=0, n=3)
    assert len(result) == 3


def test_recommend_returns_empty_dataframe_for_unknown_product_id():
    vendors = ["A", "B"]
    rec = make_recommender(vendors, np.eye(2))
    result = rec.recommend(product_id=999, n=3)
    assert result.empty
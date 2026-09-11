"""
Baseline vs. Enhanced recommender evaluation.

Compares raw cosine-similarity top-N ("baseline") against the vendor-bias-
penalised, diversity-enforced recommender ("enhanced") across the full
catalogue, and logs both as separate MLflow runs so they show up side by
side in the MLflow UI.

Metrics (computed per product, then averaged):
  - vendor_diversity_at_k     avg unique vendors in top-k
  - intra_list_similarity     avg pairwise cosine similarity within top-k
                               (lower = less redundant / repetitive list)
  - catalog_coverage          % of the full catalogue that appears in
                               *any* top-k list across all queries
  - relevance_retention       avg similarity score of top-k items
                               (enhanced's score relative to baseline shows
                               the "cost" of enforcing diversity)
  - price_spread              avg (max - min) discounted_price within top-k

Usage:
    python -m src.evaluate
    python -m src.evaluate --k 5 --sample 500
"""

import argparse
import itertools
import os

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.models.recommender import Recommender
from src.models.recommender import Recommender
from src.models.baseline_recommender import BaselineRecommender

def baseline_recommend(df: pd.DataFrame, sim_matrix: np.ndarray, idx: int, k: int) -> np.ndarray:
    """Pure top-k by raw cosine similarity — no bias penalty, no diversity enforcement."""
    scores = sim_matrix[idx].copy()
    ranked = scores.argsort()[::-1]
    ranked = ranked[ranked != idx]
    return ranked[:k]


def enhanced_recommend_indices(rec: Recommender, df: pd.DataFrame, idx: int, k: int) -> np.ndarray:
    """Re-derive the index array the enhanced recommender picks, for metric reuse."""
    source_vendor = df.iloc[idx]["vendor"]
    raw_scores = rec.sim_matrix[idx].copy()
    penalised = rec.apply_bias_penalty(raw_scores, source_vendor)
    ranked = penalised.argsort()[::-1]
    ranked = ranked[ranked != idx]
    final = rec.enforce_diversity(ranked, source_vendor, k)
    return np.array(final)


def intra_list_similarity(sim_matrix: np.ndarray, indices: np.ndarray) -> float:
    """Average pairwise cosine similarity among the recommended items themselves."""
    if len(indices) < 2:
        return 0.0
    pairs = list(itertools.combinations(indices, 2))
    sims = [sim_matrix[i, j] for i, j in pairs]
    return float(np.mean(sims))


def evaluate_variant(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    sample_indices: np.ndarray,
    k: int,
    variant: str,
    rec: Recommender = None,
) -> dict:
    diversity_scores, ils_scores, relevance_scores, price_spreads = [], [], [], []
    seen_recommended_ids = set()
    n_catalog = len(df)

    for idx in sample_indices:
        if variant == "baseline":
            rec_idx = baseline_recommend(df, sim_matrix, idx, k)
        else:
            rec_idx = enhanced_recommend_indices(rec, df, idx, k)

        if len(rec_idx) == 0:
            continue

        rec_rows = df.iloc[rec_idx]
        seen_recommended_ids.update(df.iloc[rec_idx]["id"].tolist())

        diversity_scores.append(rec_rows["vendor"].nunique())
        ils_scores.append(intra_list_similarity(sim_matrix, rec_idx))
        relevance_scores.append(float(np.mean(sim_matrix[idx][rec_idx])))

        prices = rec_rows["discounted_price"].values
        price_spreads.append(float(prices.max() - prices.min()) if len(prices) else 0.0)

    return {
        "vendor_diversity_at_k": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "intra_list_similarity": float(np.mean(ils_scores)) if ils_scores else 0.0,
        "catalog_coverage_pct": 100.0 * len(seen_recommended_ids) / n_catalog,
        "relevance_retention": float(np.mean(relevance_scores)) if relevance_scores else 0.0,
        "price_spread_pkr": float(np.mean(price_spreads)) if price_spreads else 0.0,
        "num_queries_evaluated": len(diversity_scores),
    }


def run_evaluation(
    k: int = 5,
    sample: int = None,
    experiment_name: str = "VendorLensX-Eval",
    clean_data_path: str = "Data/clean_products.csv",
    matrix_path: str = "models/similarity_matrix.joblib",
):
    df = pd.read_csv(clean_data_path, encoding="utf-8-sig")
    sim_matrix = joblib.load(matrix_path)

    if sample and sample < len(df):
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(df), size=sample, replace=False)
    else:
        sample_indices = np.arange(len(df))

    rec = Recommender.__new__(Recommender)
    rec.df = df
    rec.sim_matrix = sim_matrix

    mlflow.set_experiment(experiment_name)

    results = {}
    for variant in ("baseline", "enhanced"):
        print(f"\nEvaluating '{variant}' over {len(sample_indices)} products (k={k})...")
        metrics = evaluate_variant(df, sim_matrix, sample_indices, k, variant, rec)
        results[variant] = metrics

        with mlflow.start_run(run_name=f"{variant}_k{k}"):
            mlflow.log_param("variant", variant)
            mlflow.log_param("k", k)
            mlflow.log_param("num_products_in_catalog", len(df))
            mlflow.log_param("num_queries_evaluated", metrics["num_queries_evaluated"])
            for name, value in metrics.items():
                if name != "num_queries_evaluated":
                    mlflow.log_metric(name, value)

        print(f"  vendor_diversity_at_{k}:  {metrics['vendor_diversity_at_k']:.2f}")
        print(f"  intra_list_similarity:   {metrics['intra_list_similarity']:.3f}")
        print(f"  catalog_coverage:        {metrics['catalog_coverage_pct']:.1f}%")
        print(f"  relevance_retention:     {metrics['relevance_retention']:.3f}")
        print(f"  price_spread (PKR):      {metrics['price_spread_pkr']:,.0f}")

    print("\n" + "=" * 60)
    print(f"{'Metric':<28}{'Baseline':>15}{'Enhanced':>15}")
    print("=" * 60)
    for name in results["baseline"]:
        if name == "num_queries_evaluated":
            continue
        b, e = results["baseline"][name], results["enhanced"][name]
        print(f"{name:<28}{b:>15.3f}{e:>15.3f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline vs. enhanced recommender")
    parser.add_argument("--k", type=int, default=5, help="Top-k recommendations to evaluate")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of products to sample as queries (default: full catalogue)",
    )
    args = parser.parse_args()

    run_evaluation(k=args.k, sample=args.sample)
    
def baseline_recommend(df: pd.DataFrame, sim_matrix: np.ndarray, idx: int, k: int) -> np.ndarray:
    """Pure top-k by raw cosine similarity, via BaselineRecommender — no bias penalty, no diversity enforcement."""
    product_id = df.iloc[idx]["id"]
    baseline = BaselineRecommender.__new__(BaselineRecommender)
    baseline.df = df
    baseline.sim_matrix = sim_matrix
    result = baseline.recommend(product_id, n=k)
    return df.index.get_indexer(result.index)
    
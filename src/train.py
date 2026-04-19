

import os
import time
import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FIX BUG-21: correct module name is preprocess, not processed
from src.data.processed import load_and_clean
from src.features.extract import run_feature_engineering
from src.utils.helpers import validate_clean_products, timer
from src.models.recommender import Recommender


@timer
def run_training_pipeline(
    experiment_name: str = "VendorLensX",
    max_features: int = 8000,      # FIX BUG-22: raised from 5000
    ngram_range: tuple = (1, 2),
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        start = time.time()

        # ── Locate source data ────────────────────────────────────────────────
        data_dir = os.path.join(os.getcwd(), "Data")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data folder not found at {data_dir}")

        raw_files = [f for f in os.listdir(data_dir)
                     if f.endswith(".csv") and "clean" not in f.lower()]
        if not raw_files:
            raise FileNotFoundError(f"No source CSV found in {data_dir}")

        raw_path   = os.path.join(data_dir, raw_files[0])
        clean_path = os.path.join(data_dir, "clean_products.csv")
        print(f"Source: {raw_files[0]}")

        # ── Prepare data ──────────────────────────────────────────────────────
        df = load_and_clean(raw_path)
        df = run_feature_engineering(df)
        validate_clean_products(df)
        df.to_csv(clean_path, index=False, encoding="utf-8-sig")

        # ── Fit TF-IDF ────────────────────────────────────────────────────────
        print(f"Fitting TF-IDF (max_features={max_features}, ngram={ngram_range})...")
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,              # ignore tokens appearing in only 1 document
            sublinear_tf=True,     # log-normalise TF to reduce dominance of very frequent terms
        )
        tfidf_matrix = tfidf.fit_transform(df["clean_content"].fillna(""))
        print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

        # ── Cosine similarity ─────────────────────────────────────────────────
        print("Computing similarity matrix...")
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # ── Diversity KPI ─────────────────────────────────────────────────────
        # FIX BUG-23: Recommender built with already-loaded objects, no CSV re-read
        rec = Recommender.__new__(Recommender)
        rec.df         = df
        rec.sim_matrix = sim_matrix

        sample_ids  = df["id"].sample(min(100, len(df)), random_state=42).values
        div_scores  = []
        for pid in sample_ids:
            result = rec.recommend(pid, n=5)
            if not result.empty:
                div_scores.append(result["vendor"].nunique())

        # FIX BUG-24: guard against empty div_scores
        avg_diversity = sum(div_scores) / len(div_scores) if div_scores else 0.0

        # ── Save artifacts ────────────────────────────────────────────────────
        os.makedirs("models", exist_ok=True)
        joblib.dump(tfidf,      "models/vectorizer.joblib")
        joblib.dump(sim_matrix, "models/similarity_matrix.joblib")

        # ── Log to MLflow ─────────────────────────────────────────────────────
        duration = time.time() - start

        # FIX BUG-25: log all tunable hyperparameters, not just vectorizer_type
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("max_features",    max_features)
        mlflow.log_param("ngram_range",     str(ngram_range))
        mlflow.log_param("min_df",          2)
        mlflow.log_param("sublinear_tf",    True)
        mlflow.log_param("num_products",    len(df))

        mlflow.log_metric("avg_vendor_diversity_top5", avg_diversity)
        mlflow.log_metric("training_time_seconds",     duration)
        mlflow.log_metric("vocabulary_size",           len(tfidf.vocabulary_))

        mlflow.log_artifact("models/vectorizer.joblib")
        mlflow.log_artifact("models/similarity_matrix.joblib")
        mlflow.log_artifact(clean_path)

        print(f"\nTraining complete in {duration:.1f}s")
        print(f"Avg vendor diversity (top-5): {avg_diversity:.2f}")
        print(f"Vocabulary size: {len(tfidf.vocabulary_)}")


if __name__ == "__main__":
    # Experiment A — title only baseline (set clean_content to title before running)
    # Experiment B — full clean_content (brand × 2 + category + title + specs)
    # Experiment C — full content, max_features=10000
    run_training_pipeline(
        experiment_name="VendorLensX",
        max_features=8000,
        ngram_range=(1, 2),
    )
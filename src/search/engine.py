

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


class SearchEngine:
    def __init__(self, data_path: str, vectorizer_path: str, matrix_path: str):
        self.df = pd.read_csv(data_path, encoding="utf-8-sig")
        self.load_index(vectorizer_path, matrix_path)

    def load_index(self, vectorizer_path: str, matrix_path: str) -> None:
        """
        Load pre-computed TF-IDF vectorizer and matrix.
        Fixed: uses joblib (not pickle), raises on failure instead of silent continue.
        """
        try:
            self.vectorizer   = joblib.load(vectorizer_path)
            self.tfidf_matrix = joblib.load(matrix_path)
            print("Search index loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load search index: {e}\n"
                "Run train.py first to generate models/vectorizer.joblib"
            ) from e

    def search(
        self,
        query: str,
        category: Optional[str] = None, # Changed from str | None
        brand: Optional[str] = None,    # Changed from str | None
        min_price: Optional[float] = None, # Changed from float | None
        max_price: Optional[float] = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        TF-IDF cosine similarity search with metadata filters.
        Returns top-N results sorted by similarity score.
        """
        try:
            query_vec  = self.vectorizer.transform([query.lower()])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            mask = np.ones(len(self.df), dtype=bool)
            if category:  mask &= self.df["category"].str.lower() == category.lower()
            if brand:     mask &= self.df["brand"].str.lower() == brand.lower()
            if min_price is not None: mask &= self.df["discounted_price"] >= min_price
            if max_price is not None: mask &= self.df["discounted_price"] <= max_price

            filtered_idx = np.where(mask)[0]
            if len(filtered_idx) == 0:
                return pd.DataFrame()

            scores      = sim_scores[filtered_idx]
            top_idx     = filtered_idx[scores.argsort()[::-1][:n]]
            results     = self.df.iloc[top_idx].copy()
            results["_score"] = sim_scores[top_idx]
            return results

        except Exception as e:
            print(f"Search error: {e}")
            return pd.DataFrame()
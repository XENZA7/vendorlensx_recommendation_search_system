import numpy as np
import pandas as pd


class BaselineRecommender:
    """
    Naive recommender: pure top-N by raw cosine similarity.

    No vendor-bias penalty, no diversity enforcement — this is the
    "before" model that Recommender (src/models/recommender.py) improves
    on. Kept as its own class, with the same constructor/recommend
    interface as Recommender, so the two can be swapped interchangeably
    in evaluation or API code.
    """

    def __init__(self, data_path: str, sim_matrix: np.ndarray):
        self.df = pd.read_csv(data_path, encoding="utf-8-sig")
        self.sim_matrix = sim_matrix

    def load_matrix(self, matrix_path: str) -> None:
        import joblib
        try:
            self.sim_matrix = joblib.load(matrix_path)
        except Exception as e:
            raise RuntimeError(f"Matrix load failed: {e}") from e

    def recommend(self, product_id: int, n: int = 5) -> pd.DataFrame:
        """
        Return top-N most similar products by raw cosine similarity.

        No same-vendor penalty, no diversity guarantee — results can
        legitimately be dominated by a single vendor if that vendor's
        listings happen to be most similar.
        """
        try:
            matches = self.df[self.df["id"] == product_id]
            if matches.empty:
                raise KeyError(f"product_id {product_id} not found in dataset")

            idx = matches.index[0]
            scores = self.sim_matrix[idx].copy()

            ranked = scores.argsort()[::-1]
            ranked = ranked[ranked != idx]   # exclude self

            top_n = ranked[:n]
            return self.df.iloc[top_n].copy()

        except KeyError as e:
            print(f"BaselineRecommender KeyError: {e}")
            return pd.DataFrame()
        except IndexError as e:
            print(f"BaselineRecommender IndexError — matrix/df shape mismatch: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"BaselineRecommender unexpected error: {e}")
            raise
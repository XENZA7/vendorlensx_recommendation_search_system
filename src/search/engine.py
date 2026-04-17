import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SearchEngine:
    def __init__(self, data_path, model_path, matrix_path):
        self.df = pd.read_csv(data_path)
        self.load_index(model_path, matrix_path)

    def load_index(self, model_path, matrix_path):
        """Loads pre-computed TF-IDF vectorizer and matrix."""
        try:
            with open(model_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(matrix_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            print(" Search Index Loaded Successfully.")
        except Exception as e:
            print(f" Failed to load index: {e}")

    def search(self, query, category=None, min_price=None, max_price=None, brand=None, n=10):
        """Semantic search with strict metadata filtering."""
        try:
            # 1. Vectorize query and get similarity
            query_vec = self.vectorizer.transform([query.lower()])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # 2. Build filter mask
            mask = np.ones(len(self.df), dtype=bool)
            if category: mask &= (self.df['category'].str.lower() == category.lower())
            if brand:    mask &= (self.df['brand'].str.lower() == brand.lower())
            if min_price: mask &= (self.df['discounted_price'] >= min_price)
            if max_price: mask &= (self.df['discounted_price'] <= max_price)
            
            # 3. Apply mask and sort
            filtered_indices = np.where(mask)[0]
            scores = sim_scores[filtered_indices]
            top_indices = filtered_indices[scores.argsort()[::-1][:n]]
            
            return self.df.iloc[top_indices].copy()
        except Exception as e:
            print(f" Search Error: {e}")
            return pd.DataFrame()
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, data_path, matrix_path):
        self.df = pd.read_csv(data_path)
        self.load_matrix(matrix_path)

    def load_matrix(self, matrix_path):
        try:
            with open(matrix_path, 'rb') as f:
                self.sim_matrix = pickle.load(f)
        except Exception as e:
            print(f" Matrix load failed: {e}")

    def apply_bias_penalty(self, scores, source_vendor):
        """Rule: -25% score if same vendor and similarity > 0.8."""
        for i in range(len(scores)):
            current_vendor = self.df.iloc[i]['vendor']
            if current_vendor == source_vendor and scores[i] > 0.8:
                scores[i] *= 0.75
        return scores

    def enforce_diversity(self, ranked_indices, source_vendor, n=5):
        """Guarantees at least 2 unique vendors in the top N."""
        final_results = []
        vendors_found = {source_vendor}
        
        for idx in ranked_indices:
            if len(final_results) >= n: break
            
            # Hard diversity check for the last slot
            if len(final_results) == (n - 1) and len(vendors_found) < 2:
                if self.df.iloc[idx]['vendor'] != source_vendor:
                    final_results.append(idx)
                    vendors_found.add(self.df.iloc[idx]['vendor'])
            else:
                final_results.append(idx)
                vendors_found.add(self.df.iloc[idx]['vendor'])
                
        return final_results

    def recommend(self, product_id, n=5):
        try:
            idx = self.df[self.df['id'] == product_id].index[0]
            source_vendor = self.df.iloc[idx]['vendor']
            
            # 1. Get raw scores from pre-computed matrix
            raw_scores = self.sim_matrix[idx].copy()
            
            # 2. Apply Penalty Logic
            penalized_scores = self.apply_bias_penalty(raw_scores, source_vendor)
            
            # 3. Initial Sort
            ranked_indices = penalized_scores.argsort()[::-1][1:] # Skip self
            
            # 4. Enforce Hard Diversity
            final_indices = self.enforce_diversity(ranked_indices, source_vendor, n)
            
            return self.df.iloc[final_indices]
        except:
            return pd.DataFrame()
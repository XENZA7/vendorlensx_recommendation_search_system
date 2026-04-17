import joblib
import mlflow
import time
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- IMPORT CHECK ---
from src.data.processed import load_and_clean 
from src.features.extract import run_feature_engineering
from src.utils.helpers import validate_clean_products, timer
from src.models.recommender import Recommender

@timer
def run_training_pipeline():
    # 1. Initialize MLflow
    mlflow.set_experiment("Tech_Comparison_Engine")
    
    with mlflow.start_run():
        start_time = time.time()
        
        # 2. Smart Path Logic
        data_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ The 'data' folder is missing at {data_dir}")

        # Find the source CSV (not the clean output)
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'clean' not in f]
        if not all_files:
            raise FileNotFoundError(f"❌ No source CSV found in {data_dir}")
        
        raw_path = os.path.join(data_dir, all_files[0])
        clean_path = os.path.join(data_dir, 'clean_products.csv')

        print(f"📂 Found source data: {all_files[0]}")
        
        # 3. Data Preparation
        df = load_and_clean(raw_path)
        
        # FIX: Ensure 'images' column exists for the Feature Factory
        if 'images' not in df.columns:
            print("⚠️  Column 'images' not found. Adding a placeholder.")
            df['images'] = ""
        
        print("🛠️  Running Feature Engineering...")
        df = run_feature_engineering(df)
        
        # 4. Quality Audit (Threshold adjusted in src/utils/helpers.py previously)
        print("✅ Validating Data Quality...")
        validate_clean_products(df)
        
        # 5. Vectorization (The AI Learning bit)
        print("🧠 Fitting TF-IDF Model...")
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf.fit_transform(df['clean_content'])
        
        # 6. Similarity Math
        print("📐 Computing Similarity Matrix...")
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 7. Save Clean Data (Using utf-8-sig to fix the emoji/special char error)
        df.to_csv(clean_path, index=False, encoding='utf-8-sig') 
        
        # 8. Diversity KPI Calculation
        print("📊 Calculating Diversity...")
        # Recommender needs to use encoding='utf-8-sig' internally as well
        temp_rec = Recommender(clean_path, sim_matrix)
        sample_ids = df['id'].sample(min(100, len(df))).values
        
        # Using 'vendor' as identified in your columns list
        div_scores = [temp_rec.recommend(pid, n=5)['vendor'].nunique() for pid in sample_ids]
        avg_diversity = sum(div_scores) / len(div_scores)
        
        # 9. Save Artifacts
        if not os.path.exists('models'):
            os.makedirs('models')
            
        joblib.dump(tfidf, 'models/vectorizer.joblib')
        joblib.dump(sim_matrix, 'models/similarity_matrix.joblib')
        
        # 10. Log to MLflow
        duration = time.time() - start_time
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_metric("avg_vendor_diversity", avg_diversity)
        mlflow.log_metric("training_time", duration)
        
        mlflow.log_artifact('models/vectorizer.joblib')
        mlflow.log_artifact('models/similarity_matrix.joblib')
        mlflow.log_artifact(clean_path)
        
        print(f"\n✨ SUCCESS! Training Complete.")
        print(f"📊 Diversity Score: {avg_diversity:.2f} (Vendors per Top 5)")
        print(f"⏱️  Total Time: {duration:.2f}s")

if __name__ == "__main__":
    run_training_pipeline()
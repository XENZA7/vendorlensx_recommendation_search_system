import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from typing import List, Optional

# --- CORE LOGIC IMPORTS ---
from src.models.recommender import Recommender

# Global placeholders for models and data
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """6.2 Model Persistence Strategy: Load once at startup"""
    try:
        print("🧠 Loading models and dataset into memory...")
        # Paths (adjust based on your actual folder structure)
        clean_data_path = "data/clean_products.csv"
        matrix_path = "models/similarity_matrix.joblib"
        vectorizer_path = "models/vectorizer.joblib"

        # Load data
        models["df"] = pd.read_csv(clean_data_path, encoding='utf-8-sig')
        models["sim_matrix"] = joblib.load(matrix_path)
        models["vectorizer"] = joblib.load(vectorizer_path)
        
        # Initialize Recommender
        models["recommender"] = Recommender(clean_data_path, models["sim_matrix"])
        
        print(f"✅ Startup complete. Dataset size: {len(models['df'])} products.")
        yield
    finally:
        # Cleanup logic if needed
        models.clear()
        print("🛑 API shutting down...")

app = FastAPI(title="VendorLensX Comparison API", lifespan=lifespan)

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "message": "Welcome to VendorLensX API",
        "docs": "Go to /docs to test the endpoints",
        "status": "Running"
    }

@app.get("/health")
async def health_check():
    """Check API and Model status"""
    return {
        "status": "online",
        "models_loaded": "recommender" in models,
        "dataset_size": len(models.get("df", []))
    }

@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: float = 0,
    max_price: float = 1000000,
    n: int = 10
):
    """6.1 GET /search - Fast keyword search with filters"""
    df = models["df"]
    
    # 1. Filter logic
    mask = (df['title'].str.contains(q, case=False, na=False)) & \
           (df['discounted_price'] >= min_price) & \
           (df['discounted_price'] <= max_price)
    
    if category: mask &= (df['category'] == category)
    if brand: mask &= (df['brand'] == brand)
    
    results = df[mask].head(n)
    
    # Convert to list of dicts with image URLs and vendor links
    return results.to_dict(orient="records")

@app.get("/recommend/{product_id}")
async def recommend(product_id: str, n: int = 5):
    """6.1 GET /recommend/{product_id} - Cross-vendor recommendations"""
    try:
        # Recommender class already handles the similarity math
        recommendations = models["recommender"].recommend(product_id, n=n)
        return recommendations.to_dict(orient="records")
    except Exception:
        raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found.")

@app.get("/product/{product_id}")
async def get_product(product_id: str):
    """6.1 GET /product/{product_id} - Full details and specs"""
    df = models["df"]
    product = df[df['id'] == product_id]
    
    if product.empty:
        raise HTTPException(status_code=404, detail="Product not found.")
    
    return product.iloc[0].to_dict()

@app.get("/filters")
async def get_filters(category: Optional[str] = None):
    """6.1 GET /filters - Dynamic frontend filters"""
    df = models["df"]
    if category:
        df = df[df['category'] == category]
        
    return {
        "brands": df['brand'].dropna().unique().tolist(),
        "min_price": float(df['discounted_price'].min()),
        "max_price": float(df['discounted_price'].max()),
        "categories": models["df"]['category'].unique().tolist()
    }

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)
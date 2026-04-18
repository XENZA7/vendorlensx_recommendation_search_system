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
        print(" Loading models and dataset into memory...")
        # Paths (adjust based on your actual folder structure)
        clean_data_path = "Data/clean_products.csv"
        matrix_path = "models/similarity_matrix.joblib"
        vectorizer_path = "models/vectorizer.joblib"

        # Load data
        models["df"] = pd.read_csv(clean_data_path, encoding='utf-8-sig')
        models["sim_matrix"] = joblib.load(matrix_path)
        models["vectorizer"] = joblib.load(vectorizer_path)
        
        # Initialize Recommender
        models["recommender"] = Recommender(clean_data_path, models["sim_matrix"])
        
        print(f" Startup complete. Dataset size: {len(models['df'])} products.")
        yield
    finally:
        # Cleanup logic if needed
        models.clear()
        print(" API shutting down...")

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
async def recommend(product_id: int, n: int = 5):
    try:
        # 1. Validate that the ID exists in the dataframe first
        if product_id not in models["df"]['id'].values:
            raise HTTPException(status_code=404, detail="Product ID not in database")

        # 2. Get recommendations
        recommendations = models["recommender"].recommend(product_id, n=n)
        
        # 3. Clean the output for JSON (Crucial for 500 error prevention)
        return recommendations.fillna("").to_dict(orient="records")
    except Exception as e:
        print(f" Recommender Crash: {e}")
        # This will tell us if it's an Index error or a Math error
        raise HTTPException(status_code=500, detail=f"Recommendation Error: {str(e)}")
    
@app.get("/product/{product_id}")
async def get_product(product_id: int): # Change str to int here
    df = models["df"]
    
    # Lookup by integer
    product = df[df['id'] == product_id]
    
    if product.empty:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
    
    return product.fillna("").iloc[0].to_dict()

@app.get("/filters")
async def get_filters(category: Optional[str] = None):
    try:
        df = models["df"]
        if category and category != "All":
            df = df[df['category'] == category]
        
        # Use .dropna() and handle empty lists to prevent JSON crashes
        brands = sorted([str(b) for b in df['brand'].unique() if b and str(b) != 'nan'])
        categories = sorted([str(c) for c in df['category'].unique() if c and str(c) != 'nan'])
        
        # Ensure prices are Python floats, not NumPy floats
        min_p = float(df['discounted_price'].min()) if not df.empty else 0
        max_p = float(df['discounted_price'].max()) if not df.empty else 1000000

        return {
            "brands": brands,
            "categories": categories,
            "min_price": min_p,
            "max_price": max_p
        }
    except Exception as e:
        print(f"🚨 Filter Crash: {e}")
        raise HTTPException(status_code=500, detail=f"Filter Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)
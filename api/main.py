

import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from src.models.recommender import Recommender
from src.search.engine import SearchEngine
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) # Add this

# Global model store — loaded once at startup, reused for every request
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data once at startup. Never reload on a request."""
    print("Loading models into memory...")

    clean_data_path  = "Data/clean_products.csv"
    matrix_path      = "models/similarity_matrix.joblib"
    vectorizer_path  = "models/vectorizer.joblib"

    df         = pd.read_csv(clean_data_path, encoding="utf-8-sig")
    sim_matrix = joblib.load(matrix_path)
    vectorizer = joblib.load(vectorizer_path)

    models["df"]          = df
    models["sim_matrix"]  = sim_matrix
    models["vectorizer"]  = vectorizer

    # FIX BUG-17: Recommender constructed once with already-loaded objects
    models["recommender"] = Recommender.__new__(Recommender)
    models["recommender"].df         = df
    models["recommender"].sim_matrix = sim_matrix

    # FIX BUG-16: SearchEngine constructed once — search calls use TF-IDF, not str.contains
    models["search_engine"] = SearchEngine.__new__(SearchEngine)
    models["search_engine"].df          = df
    models["search_engine"].vectorizer  = vectorizer
    models["search_engine"].tfidf_matrix = vectorizer.transform(df["clean_content"].fillna(""))

    print(f"Startup complete. {len(df)} products loaded.")
    yield

    models.clear()
    print("API shut down.")


app = FastAPI(title="VendorLensX API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "VendorLensX API", "docs": "/docs", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "models_loaded": "search_engine" in models and "recommender" in models,
        "dataset_size": len(models.get("df", [])),
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: float = 0,
    max_price: float = 1_000_000,
    n: int = 10,
):
    """
    TF-IDF semantic search with metadata filters.

    Fixed vs original:
      - Original used str.contains() — plain keyword match, TF-IDF never called
      - Now delegates to SearchEngine which uses cosine similarity on the TF-IDF matrix
    """
    try:
        results = models["search_engine"].search(
            query=q,
            category=category,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            n=n,
        )
        return results.fillna("").to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


@app.get("/recommend/{product_id}")
async def recommend(product_id: int, n: int = 5):
    """Return N similar products with vendor diversity guaranteed."""
    if product_id not in models["df"]["id"].values:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    try:
        results = models["recommender"].recommend(product_id, n=n)
        return results.fillna("").to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {e}")


@app.get("/product/{product_id}")
async def get_product(product_id: int):
    product = models["df"][models["df"]["id"] == product_id]
    if product.empty:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return product.fillna("").iloc[0].to_dict()


@app.get("/filters")
async def get_filters(category: Optional[str] = None):
    try:
        df = models["df"].copy()
        if category and category != "All":
            df = df[df["category"] == category]

        brands     = sorted(str(b) for b in df["brand"].dropna().unique() if str(b) != "nan")
        categories = sorted(str(c) for c in models["df"]["category"].dropna().unique())
        min_price  = float(df["discounted_price"].min()) if not df.empty else 0.0
        max_price  = float(df["discounted_price"].max()) if not df.empty else 1_000_000.0

        return {"brands": brands, "categories": categories,
                "min_price": min_price, "max_price": max_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
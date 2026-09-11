# VendorLensX — Vendor-Aware Product Search & Recommendation Engine

VendorLensX is a small end-to-end ML system that cleans a scraped, multi-vendor
e-commerce product catalogue (mobiles, laptops, earbuds, watches) and serves
**TF-IDF semantic search** and **content-based recommendations** through a
FastAPI backend and a Streamlit frontend, with experiments tracked in MLflow.

The dataset is scraped listings from five Pakistani electronics vendors
(MEGA.PK, PriceOye, ComputerZone, TechGlobe, Paklap) — the same phone/laptop
models get listed by multiple vendors at different prices, which is what
makes vendor-aware recommendations useful here.

## What it does

- **Cleans messy scraped data**: normalises prices (`"Rs. 145,000"` → `145000.0`),
  standardises availability strings, drops scraper-only columns.
- **Recovers missing brands and specs**: brand inference from product titles,
  RAM/storage/processor/GPU parsing out of a free-form specifications dict,
  category-specific feature extraction (5G, PTA approval, water resistance, etc.).
- **Builds a TF-IDF search index** over brand + category + title + specs, and
  a cosine-similarity matrix for recommendations.
- **De-biases vendor overlap in recommendations**: near-identical listings
  from the *same* vendor as the source product are penalised, and the top-N
  results are guaranteed to include at least two different vendors where
  possible — so "similar products" doesn't just mean "other things this one
  vendor sells."
- **Validates data quality before training**: asserts brand-null rate,
  RAM-fill rate, and empty-content rate against KPI thresholds and fails
  loudly if the pipeline regresses.
- **Tracks every training run in MLflow**: hyperparameters, vocabulary size,
  training time, and an average-vendor-diversity metric, with the fitted
  vectorizer, similarity matrix, and cleaned dataset logged as artifacts.

## Project structure

```
├── api/
│   └── main.py                # FastAPI app: /search, /recommend, /product, /filters
├── frontend/
│   └── app.py                 # Streamlit UI, talks to the API
├── src/
│   ├── data/
│   │   └── processed.py       # Structural cleaning (prices, availability, dropped cols)
│   ├── features/
│   │   └── extract.py         # Brand recovery, spec parsing, TF-IDF content building
│   ├── models/
│   │   └── recommender.py     # Vendor-bias penalty + diversity-enforced recommender
│   ├── search/
│   │   └── engine.py          # TF-IDF cosine-similarity search with metadata filters
│   ├── utils/
│   │   └── helpers.py         # Timer decorator, KPI validation, path helpers
│   ├── tests/
│   │   └── extract_test.py    # Unit tests for feature extraction
│   └── train.py                # Orchestrates the full training pipeline
├── Notebooks/                  # EDA, cleaning, and prototyping notebooks
├── Data/
│   ├── data.csv                 # Raw scraped catalogue
│   └── clean_products.csv       # Cleaned + feature-engineered output of train.py
├── models/                     # Fitted vectorizer.joblib / similarity_matrix.joblib
├── mlruns/                     # MLflow experiment tracking store
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train the model** (cleans `Data/`, fits TF-IDF, computes the similarity
matrix, writes `models/vectorizer.joblib` and `models/similarity_matrix.joblib`,
and logs a run to MLflow):

```bash
python -m src.train
```

**2. Start the API:**

```bash
uvicorn api.main:app --reload
```

The API loads all models once at startup. Interactive docs are available at
`http://127.0.0.1:8000/docs`.

| Endpoint | Description |
|---|---|
| `GET /search?q=...` | TF-IDF semantic search, with optional `category`, `brand`, `min_price`, `max_price`, `n` |
| `GET /recommend/{product_id}` | Top-N similar products, vendor-diversity enforced |
| `GET /product/{product_id}` | Full record for a single product |
| `GET /filters` | Available brands/categories and price range (optionally scoped to a category) |
| `GET /health` | Model-load status and dataset size |

**3. Start the frontend** (in a separate terminal, with the API already running):

```bash
streamlit run frontend/app.py
```

**4. (Optional) Inspect experiment runs:**

```bash
mlflow ui
```

## Running tests

```bash
pytest src/tests/
```

> **Note:** the feature-extraction tests currently expect an older function
> signature (e.g. integer vs. string return types, brand aliases like
> "iphone" → "Apple" that aren't in the current brand list) and are out of
> sync with `src/features/extract.py`. They need updating to match current
> behaviour before they can be trusted as a regression check.

## Data

Each row is a single vendor's listing for a product, with columns for title,
brand, category (`Mobile` / `Laptop` / `Earbuds` / `Watch`), vendor, prices,
availability, raw specifications, and images. `src/train.py` reads whichever
non-`clean_*` CSV is in `Data/` as the raw source, so a new scrape can be
dropped in without touching code.

## Known limitations

- Search and recommendations are TF-IDF/cosine based — matching is lexical,
  not semantic (e.g. "phone" and "smartphone" won't match unless they share
  tokens).
- The vendor→ZAR price conversion in the frontend uses a fixed exchange rate
  constant rather than a live rate.
- Training artifacts (`models/*.joblib`) and MLflow run history (`mlruns/`)
  are currently checked into the repo rather than gitignored.
  > **Note:** `models/` and `mlruns/` are gitignored. Run `python -m src.train`
> after cloning to regenerate the fitted vectorizer, similarity matrix, and
> local MLflow run history.

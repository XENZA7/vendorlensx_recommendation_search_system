# VendorLensX — Vendor-Aware Product Search & Recommendation Engine

[![Tests](https://github.com/XENZA7/vendorlensx_recommendation_search_system/actions/workflows/tests.yml/badge.svg)](https://github.com/XENZA7/vendorlensx_recommendation_search_system/actions/workflows/tests.yml)

VendorLensX is an end-to-end ML system that cleans a scraped, multi-vendor
e-commerce product catalogue (mobiles, laptops, earbuds, watches) and serves
**TF-IDF semantic search** and **content-based recommendations** through a
FastAPI backend and a Streamlit frontend, with every experiment tracked in
MLflow.

The dataset is scraped listings from five Pakistani electronics vendors
(MEGA.PK, PriceOye, ComputerZone, TechGlobe, Paklap) — the same phone and
laptop models get listed by multiple vendors at different prices. That's the
whole reason vendor-aware recommendations matter here: a naive "similar
products" feature will happily show you five listings of the exact same
phone from the exact same vendor, which is useless if you're trying to
compare prices across the market. The core engineering problem this project
solves is making recommendations *diverse across vendors* without destroying
their relevance — and then actually measuring whether that trade-off is
worth it, rather than assuming it is.

## What it does

- **Cleans messy scraped data** — normalises prices (`"Rs. 145,000"` →
  `145000.0`), standardises availability strings, drops scraper-only
  columns.
- **Recovers missing brands and specs** — brand inference from product
  titles (including mapping marketing sub-brands like "iPhone" and "Redmi"
  to their actual parent manufacturer, Apple and Xiaomi), RAM/storage/
  processor/GPU parsing out of a free-form specifications dict,
  category-specific feature extraction (5G, PTA approval, water
  resistance, etc.).
- **Builds a TF-IDF search index** over brand + category + title + specs,
  and a cosine-similarity matrix for recommendations.
- **De-biases vendor overlap in recommendations** — near-identical listings
  from the *same* vendor as the source product are penalised 25% if their
  similarity exceeds 0.8, and the top-N results are guaranteed to include
  at least two different vendors where possible.
- **Validates data quality before training** — asserts brand-null rate,
  RAM-fill rate, and empty-content rate against KPI thresholds and fails
  loudly if the pipeline regresses.
- **Tracks every training and evaluation run in MLflow** — hyperparameters,
  vocabulary size, training time, and the full evaluation metric suite
  below, with the fitted vectorizer, similarity matrix, and cleaned
  dataset logged as artifacts.
- **Evaluates itself** — a dedicated baseline-vs-enhanced comparison
  quantifies exactly what the diversity enforcement costs and buys you
  (see [Evaluation](#evaluation) below), instead of just asserting the
  feature works.

## Why TF-IDF instead of embeddings?

This was a deliberate choice, not a limitation I didn't know about.
Semantic embeddings (sentence-transformers, OpenAI embeddings, etc.) would
catch synonyms TF-IDF misses — "phone" vs "smartphone" — but they're a
black box: you can't easily explain *why* two products were considered
similar, they need a model download and GPU/CPU budget, and for a catalogue
built from structured fields (brand, category, spec strings) rather than
free-flowing prose, TF-IDF's lexical matching is already close to the
ceiling of what's achievable without embeddings, while staying fully
interpretable and running in milliseconds on a laptop. It's listed as a
known limitation below because it's a real trade-off, not because it was
an oversight.

## Project structure

├── api/
│ └── main.py # FastAPI app: /search, /recommend, /product, /filters
├── frontend/
│ └── app.py # Streamlit UI — search, filters, live FX conversion
├── src/
│ ├── data/
│ │ └── processed.py # Structural cleaning (prices, availability, dropped cols)
│ ├── features/
│ │ └── extract.py # Brand recovery, spec parsing, TF-IDF content building
│ ├── models/
│ │ ├── recommender.py # Vendor-bias penalty + diversity-enforced recommender
│ │ └── baseline_recommender.py # Naive top-N by raw cosine similarity — the "before" model
│ ├── search/
│ │ └── engine.py # TF-IDF cosine-similarity search with metadata filters
│ ├── utils/
│ │ └── helpers.py # Timer decorator, KPI validation, MLflow config, path helpers
│ ├── tests/ # 27 unit tests across extraction, both recommenders, and search
│ ├── train.py # Orchestrates the full training pipeline
│ └── evaluate.py # Baseline vs. enhanced evaluation across 5 metrics
├── .github/workflows/
│ └── tests.yml # CI — runs the full test suite on every push and PR
├── Notebooks/ # EDA, cleaning, and prototyping notebooks
├── Data/
│ ├── data.csv # Raw scraped catalogue
│ └── clean_products.csv # Cleaned + feature-engineered output of train.py
├── models/ # Fitted vectorizer.joblib / similarity_matrix.joblib (gitignored)
├── mlflow.db # MLflow SQLite tracking store (gitignored)
└── requirements.txt


## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`models/` and `mlflow.db` are gitignored — they're generated locally, not
committed, to keep the repo lightweight and avoid merge conflicts on binary
artifacts. Run the training pipeline below to regenerate them after
cloning.

## Usage

**1. Train the model** — cleans `Data/`, fits TF-IDF, computes the
similarity matrix, writes `models/vectorizer.joblib` and
`models/similarity_matrix.joblib`, and logs a run to MLflow:

```bash
python -m src.train
```

**2. Evaluate it** — compares the vendor-diversity-enforced recommender
against a naive baseline across five metrics (see below):

```bash
python -m src.evaluate              # full catalogue
python -m src.evaluate --sample 300 # faster, sampled run
```

**3. Start the API:**

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

**4. Start the frontend** (in a separate terminal, with the API already
running):

```bash
streamlit run frontend/app.py
```

The frontend converts PKR prices to ZAR using a **live exchange rate**
fetched from a no-key public FX API, cached hourly, with a graceful
fallback to a fixed constant if the API is unreachable — so a stale rate
never silently misleads someone comparing prices.

**5. (Optional) Inspect experiment runs:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow tracking runs against a local SQLite database rather than the
plain filesystem store (`./mlruns`), which newer MLflow versions have
deprecated. This was a genuinely necessary fix, not just tidiness — the
filesystem backend throws a hard error on some MLflow releases, and SQLite
keeps run history queryable.

## Evaluation

<a name="evaluation"></a>

The interesting engineering question here isn't "does vendor-diversity
enforcement work" — it's "what does it cost, and is that cost worth
paying." `src/evaluate.py` answers this directly by running two variants
of the recommender across the same queries and comparing them:

- **Baseline** — pure top-N by raw cosine similarity (`BaselineRecommender`).
  No same-vendor penalty, no diversity guarantee.
- **Enhanced** — the production recommender (`Recommender`): 25% same-vendor
  penalty above 0.8 similarity, plus enforced diversity (≥2 vendors) in the
  top-N.

**Results across the full 1,666-product catalogue (k=5):**

| Metric | Baseline | Enhanced | Change |
|---|---:|---:|---:|
| Vendor Diversity@5 | 1.38 | **2.09** | +51% |
| Intra-List Similarity | 0.627 | **0.484** | −23% (less redundant) |
| Catalog Coverage | 97.5% | 96.7% | −0.8pt (negligible) |
| Relevance Retention | 0.676 | 0.604 | −11% |
| Price Spread (PKR) | 51,983 | **65,046** | +25% |

**Reading this table:** enforcing vendor diversity costs about 11% in raw
cosine relevance, and buys a 51% increase in vendor diversity, a 25%
increase in the price range a shopper sees (the actual point of the
feature — real price comparison), and a 23% drop in list redundancy —
all with essentially no loss in how much of the catalogue ever gets
recommended. That's a trade worth making.

**Precision@k (relevance proxy):** the metrics above measure internal
consistency (does the list overlap with itself, does it cover the
catalogue) but not whether recommendations are actually *good*
substitutes. `evaluate.py` also computes `precision_at_k` — the fraction
of top-k items that share the source product's category **and** fall
within ±20% of its price. Category alone was rejected as a relevance
signal because it's baked hard into the TF-IDF content (brand ×2 +
category + title + specs), so it scores near 100% regardless of
recommendation quality and tells you nothing. Requiring a comparable
price band approximates what a shopper actually means by "a relevant
alternative." On a 300-product sample, this showed baseline at 0.129
vs. enhanced at 0.110 — a real but modest relevance cost, consistent
with the story above. Run `python -m src.evaluate` yourself to reproduce
this across the full catalogue.

## Testing

```bash
pytest src/tests/
```

27 tests across four files:
- `extract_test.py` — RAM/storage parsing, brand recovery and aliasing
- `recommender_test.py` — bias-penalty math, diversity-enforcement edge
  cases (including the specific bug this logic was fixed for: silently
  returning fewer than N results when diversity is impossible)
- `baseline_recommender_test.py` — confirms the baseline has *no* penalty
  or diversity behavior, as a contrast to `recommender_test.py`
- `search_engine_test.py` — TF-IDF ranking and metadata filters, using a
  real fitted vectorizer rather than mocks

All tests use small synthetic in-memory datasets rather than the real
catalogue, so they run in seconds and don't require training first. CI
(`.github/workflows/tests.yml`) runs the full suite on every push and pull
request.

## Data

Each row is a single vendor's listing for a product, with columns for
title, brand, category (`Mobile` / `Laptop` / `Earbuds` / `Watch`), vendor,
prices, availability, raw specifications, and images. `src/train.py` reads
whichever non-`clean_*` CSV is in `Data/` as the raw source, so a new
scrape can be dropped in without touching code.

## Known limitations

- Search and recommendations are TF-IDF/cosine based — matching is
  lexical, not semantic (e.g. "phone" and "smartphone" won't match unless
  they share tokens). See [Why TF-IDF instead of embeddings?](#why-tf-idf-instead-of-embeddings)
  above for why this is a deliberate trade-off rather than an oversight.
- `precision_at_k`'s relevance proxy (same category + price within ±20%)
  is an approximation, not ground-truth relevance labels — a genuinely
  rigorous evaluation would need human-labeled relevance judgments, which
  weren't available for this dataset.
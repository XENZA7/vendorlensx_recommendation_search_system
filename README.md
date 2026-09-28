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

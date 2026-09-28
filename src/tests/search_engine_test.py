import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search.engine import SearchEngine


def make_search_engine(df: pd.DataFrame) -> SearchEngine:
    """Build a SearchEngine from a synthetic dataframe with a real fitted TF-IDF index."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["clean_content"])

    engine = SearchEngine.__new__(SearchEngine)
    engine.df = df
    engine.vectorizer = vectorizer
    engine.tfidf_matrix = tfidf_matrix
    return engine


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "title": ["Samsung Galaxy S21", "Samsung Galaxy S22", "HP Laptop Pavilion", "Dell Laptop XPS"],
        "brand": ["Samsung", "Samsung", "HP", "Dell"],
        "category": ["Mobile", "Mobile", "Laptop", "Laptop"],
        "vendor": ["MEGA.PK", "PriceOye", "TechGlobe", "Paklap"],
        "discounted_price": [50000.0, 60000.0, 120000.0, 150000.0],
        "clean_content": [
            "samsung samsung mobile samsung galaxy s21",
            "samsung samsung mobile samsung galaxy s22",
            "hp hp laptop hp pavilion",
            "dell dell laptop dell xps",
        ],
    })


def test_search_returns_relevant_results_ranked_by_similarity(sample_df):
    engine = make_search_engine(sample_df)
    results = engine.search(query="samsung galaxy", n=5)

    assert not results.empty
    # Both Samsung phones should outrank the laptops for a Samsung query
    assert set(results.iloc[:2]["brand"]) == {"Samsung"}


def test_search_category_filter(sample_df):
    engine = make_search_engine(sample_df)
    results = engine.search(query="samsung", category="Laptop", n=5)

    # Category filter should exclude the Samsung phones even though they match the query best
    assert all(results["category"] == "Laptop")


def test_search_brand_filter(sample_df):
    engine = make_search_engine(sample_df)
    results = engine.search(query="laptop", brand="Dell", n=5)

    assert all(results["brand"] == "Dell")


def test_search_price_range_filter(sample_df):
    engine = make_search_engine(sample_df)
    results = engine.search(query="laptop", min_price=130000, max_price=200000, n=5)

    assert all(results["discounted_price"] >= 130000)
    assert all(results["discounted_price"] <= 200000)
    assert 4 in results["id"].values  # Dell XPS at 150000
    assert 3 not in results["id"].values  # HP Pavilion at 120000 excluded


def test_search_respects_n(sample_df):
    engine = make_search_engine(sample_df)
    results = engine.search(query="laptop mobile", n=2)
    assert len(results) <= 2


def test_search_returns_empty_dataframe_when_filters_exclude_everything(sample_df):
    engine = make_search_engine(sample_df)
    results = engine.search(query="laptop", category="Mobile", brand="Dell", n=5)
    assert results.empty


def test_search_query_is_case_insensitive(sample_df):
    engine = make_search_engine(sample_df)
    lower_results = engine.search(query="samsung galaxy", n=5)
    upper_results = engine.search(query="SAMSUNG GALAXY", n=5)

    assert list(lower_results["id"]) == list(upper_results["id"])
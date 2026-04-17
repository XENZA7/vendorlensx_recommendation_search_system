import pytest
from src.features.extract import extract_ram, recover_brand, extract_storage

# --- Test RAM Extraction ---
@pytest.mark.parametrize("input_title, expected", [
    ("Samsung Galaxy 8GB RAM", 8),
    ("HP Laptop 16 GB", 16),
    ("No RAM mentioned phone", 0),
    ("12gb ram specialized", 12),
])
def test_extract_ram(input_title, expected):
    assert extract_ram(input_title) == expected

# --- Test Brand Recovery & Alias Logic ---
def test_recover_brand_aliases():
    # Test 1: Alias 'iphone' should map to 'Apple'
    row_apple = {'title': 'iphone 13 pro', 'brand': 'Unknown'}
    assert recover_brand(row_apple) == 'Apple'
    
    # Test 2: Alias 'redmi' should map to 'Xiaomi'
    row_xiaomi = {'title': 'redmi note 10', 'brand': 'nan'}
    assert recover_brand(row_xiaomi) == 'Xiaomi'

# --- Test Storage Extraction ---
def test_extract_storage():
    assert extract_storage("iPhone 128GB") == "128GB"
    assert extract_storage("MacBook 1TB SSD") == "1TB"
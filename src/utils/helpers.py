import os
import time
import functools
from pathlib import Path

def get_data_path(filename):
    """
    Portable path resolution. 
    Finds the /data folder regardless of if you run from root or /src.
    """
    root = Path(__file__).parent.parent.parent
    return root / "data" / filename

def timer(func):
    """
    Decorator to measure execution time of pipeline steps.
    Useful for monitoring performance in train.py.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"⏱️  {func.__name__!r} finished in {end_time - start_time:.2f}s")
        return result
    return wrapper

def validate_clean_products(df):
    """
    Audits the quality of the cleaned data against project KPIs.
    If these fail, we stop the train.py execution immediately.
    """
    print("🔍 Auditing Data Quality KPIs...")

    # KPI 1: Brand Null Rate < 5%
    brand_null_pct = df['brand'].isna().mean() * 100
    if brand_null_pct > 5:
        raise AssertionError(
            f"KPI FAILURE: Brand Null Rate is {brand_null_pct:.2f}% (Limit: 5%).\n"
            "CHECK: src/features/extract.py -> recover_brand()"
        )

    # KPI 2: RAM Fill Rate > 80% for tech categories
    tech_df = df[df['category'].str.lower().isin(['mobile', 'laptop', 'smartphones', 'notebooks'])]
    if len(tech_df) > 0:
        ram_null_pct = tech_df['ram_gb'].replace(0, float('nan')).isna().mean() * 100
        ram_fill_pct = 100 - ram_null_pct
        
        if ram_fill_pct < 80:
            raise AssertionError(
                f" KPI FAILURE: Tech RAM Fill Rate is only {ram_fill_pct:.2f}% (Limit: 80%).\n"
                "CHECK: src/features/extract.py -> extract_ram()"
            )

    print(f" Audit Passed: Brand Nulls at {brand_null_pct:.1f}%, RAM Fill at {ram_fill_pct:.1f}%")
    return True
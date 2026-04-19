import time
import functools
import logging
from pathlib import Path
 
import pandas as pd
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
 
 
def get_project_root() -> Path:
    """Absolute path to project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent.parent
 
 
def get_data_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    return get_project_root() / relative
 
 
def timer(func):
    """Log execution time of any pipeline step."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__!r} finished in {elapsed:.2f}s")
        return result
    return wrapper
 
 
def validate_clean_products(df: pd.DataFrame) -> bool:
    """
    Assert cleaned data meets project KPIs. Raises AssertionError on failure.
 
    Fixed vs original:
      - ram_fill_pct initialised to 0.0 before the if block — prevents NameError
        when dataset has no Mobile/Laptop rows
      - KPI threshold corrected to 80% (original code checked < 10 but printed '80%')
    """
    logger.info("Auditing data quality KPIs...")
 
    # KPI 1: Brand null rate < 5%
    brand_null_pct = df["brand"].isna().mean() * 100
    assert brand_null_pct <= 5, (
        f"KPI FAIL — brand null rate: {brand_null_pct:.1f}% (target ≤ 5%)\n"
        "Fix: src/features/extract.py → recover_brand()"
    )
 
    # KPI 2: RAM fill rate > 80% for Mobile + Laptop
    ram_fill_pct = 0.0   # FIX: initialised here, not inside if block
    tech_df = df[df["category"].isin(["Mobile", "Laptop"])]
    if len(tech_df) > 0:
        ram_fill_pct = tech_df["ram_gb"].notna().mean() * 100
        assert ram_fill_pct >= 80, (
            f"KPI FAIL — RAM fill rate: {ram_fill_pct:.1f}% (target ≥ 80%)\n"
            "Fix: src/features/extract.py → extract_ram() / extract_specs_features()"
        )
 
    # KPI 3: clean_content has no empty strings
    empty = (df["clean_content"].str.strip() == "").sum()
    assert empty == 0, f"KPI FAIL — {empty} rows have empty clean_content"
 
    logger.info(
        f"All KPIs passed — brand nulls: {brand_null_pct:.1f}%, "
        f"RAM fill (Mobile+Laptop): {ram_fill_pct:.1f}%"
    )
    return True
 














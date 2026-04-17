import pandas as pd
import numpy as np

def drop_useless_columns(df):
    """
    Removes columns that are irrelevant to the recommendation engine 
    to reduce memory footprint and noise.
    """
    # Define columns that typically come from scrapers but aren't used in TF-IDF
    useless_cols = [
        'scraped_at', 'url', 'status_code', 'Unnamed: 0', 
        'index', 'meta_description', 'currency'
    ]
    # Only drop if they exist in the current dataframe
    cols_to_drop = [c for c in useless_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)

def normalise_prices(df):
    """
    Converts price strings (e.g., 'Rs. 145,000') into clean floats.
    Handles '0' or NaN for missing prices.
    """
    price_cols = ['actual_price', 'discounted_price']
    
    for col in price_cols:
        if col in df.columns:
            # 1. Convert to string and remove commas/currency markers
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # 2. Convert to numeric, turning empty strings into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 3. Fill NaN with 0 to prevent downstream calculation errors
            df[col] = df[col].fillna(0).astype(float)
            
    return df

def clean_availability(df):
    """
    Standardises stock status into 'In Stock' or 'Out of Stock'.
    Essential for filtering results in the UI.
    """
    if 'availability' not in df.columns:
        df['availability'] = 'In Stock' # Default assumption
        return df

    # Map various scraper outputs to a binary standard
    df['availability'] = df['availability'].astype(str).str.lower().str.strip()
    
    stock_map = {
        'instock': 'In Stock',
        'in stock': 'In Stock',
        'available': 'In Stock',
        'out of stock': 'Out of Stock',
        'outofstock': 'Out of Stock',
        'sold out': 'Out of Stock'
    }
    
    df['availability'] = df['availability'].map(stock_map).fillna('Out of Stock')
    return df

def load_and_clean(file_path):
    """
    The Orchestrator: Runs the structural cleaning pipeline.
    This is the primary entry point for train.py.
    """
    print(f" Starting Structural Cleaning for: {file_path}")
    
    # Load raw data
    df = pd.read_csv(file_path)
    
    # Execute Pipeline
    df = (df.pipe(drop_useless_columns)
            .pipe(normalise_prices)
            .pipe(clean_availability))
    
    print(f" Structural Cleaning Complete. {len(df)} rows processed.")
    return df
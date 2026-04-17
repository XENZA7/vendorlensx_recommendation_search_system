import re
import pandas as pd

# --- 1. GLOBAL CONFIGURATION ---
# Centralizing mappings to keep functions pure and data-driven
KEY_ALIASES = {
    "brands": {
        "apple": "Apple", "iphone": "Apple", "samsung": "Samsung", 
        "hp": "HP", "dell": "Dell", "lenovo": "Lenovo", "infinix": "Infinix",
        "tecno": "Tecno", "xiaomi": "Xiaomi", "redmi": "Xiaomi"
    },
    "categories": {
        "mobiles": "Mobile", "smartphones": "Mobile", 
        "laptops": "Laptop", "notebooks": "Laptop"
    }
}

# --- 2. CORE EXTRACTION WORKERS ---

def recover_brand(row):
    """
    Priority: 1. Key Alias Match, 2. Existing Brand Col, 3. Title Extraction.
    Ensures 'iphone' becomes 'Apple' and 'redmi' becomes 'Xiaomi'.
    """
    title = str(row.get('title', '')).lower()
    existing_brand = str(row.get('brand', '')).lower()
    
    # Check aliases in title first
    for alias, formal_name in KEY_ALIASES["brands"].items():
        if alias in title or alias in existing_brand:
            return formal_name
            
    # Fallback to the first word of the title if brand is 'Unknown' or NaN
    if existing_brand in ['unknown', 'nan', '']:
        return title.split()[0].capitalize()
        
    return existing_brand.capitalize()

def extract_ram(text):
    if not isinstance(text, str): 
        return None
    
    # This pattern catches: 
    # "8GB", "8 GB", "8gb", "8 gb", "8-GB", "8 giga", "RAM: 8GB"
    pattern = r'(\d+)\s*(?:GB|gb|Gb|giga|Giga|GB\s+RAM|gb\s+ram)'
    
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            val = int(match.group(1))
            # Sanity check: Laptops/Phones usually have 1GB to 128GB RAM
            if 1 <= val <= 128:
                return val
        except ValueError:
            return None
    return None

# Internal alias as requested
extract_gb = extract_ram

def parse_specs(title):
    """
    General parser for common hardware flags (SSD, Generation, PTA Status).
    Returns a space-separated string of discovered specs.
    """
    specs = []
    t = title.lower()
    if 'pta' in t: specs.append('PTA-Approved')
    if 'ssd' in t: specs.append('SSD')
    if 'generation' in t or 'gen' in t:
        gen_match = re.search(r'(\d+)(?:th|rd|st)\s*gen', t)
        if gen_match: specs.append(f"{gen_match.group(1)}Gen")
    return " ".join(specs)

def extract_first_img(img_data):
    """
    Handles messy scraper image data. 
    Can take a stringified list '[url1, url2]' or a single URL.
    """
    if pd.isna(img_data) or img_data == "":
        return "placeholder.jpg"
    
    img_str = str(img_data)
    # If it looks like a list/array string, pull the first link
    if img_str.startswith('['):
        urls = re.findall(r'(https?://[^\s\'"\]]+)', img_str)
        return urls[0] if urls else "placeholder.jpg"
    
    return img_str

def build_clean_content(row):
    """
    The 'Search Fuel'. Order is critical: Brand -> Category -> Title.
    This creates the rich string the TF-IDF vectorizer will digest.
    """
    parts = [
        str(row.get('brand', '')),
        str(row.get('category', '')),
        str(row.get('title', '')),
        str(row.get('extracted_specs', ''))
    ]
    # Filter out empty strings and join
    return " ".join([p.lower().strip() for p in parts if p])

# --- 3. THE PIPELINE ORCHESTRATOR ---

def run_feature_engineering(df):
    """
    The main factory entry point. 
    Functions are called in a specific sequence to maintain data integrity.
    """
    print("🛠️  Initiating Feature Factory...")
    
    # Step 1: Recover Brand (Base requirement for content)
    df['brand'] = df.apply(recover_brand, axis=1)
    
    # Step 2: Extract Hardware Specs
    df['ram_gb'] = df['title'].apply(extract_ram)
    df['extracted_specs'] = df['title'].apply(parse_specs)
    
    # Step 3: Clean Images
    df['primary_image'] = df['images'].apply(extract_first_img)
    
    # Step 4: Final Construction (Depends on Step 1 and 2)
    df['clean_content'] = df.apply(build_clean_content, axis=1)
    
    print(f" Feature Engineering Complete. Generated {len(df.columns)} features.")
    return df
def extract_storage(text):
    """Detects Storage (e.g., '256GB', '1TB')."""
    if not isinstance(text, str): return None
    tb_match = re.search(r'(\d+)\s*(?:tb|TB)', text, re.IGNORECASE)
    if tb_match: return f"{tb_match.group(1)}TB"
    gb_match = re.search(r'(\d+)\s*(?:gb|GB)', text, re.IGNORECASE)
    if gb_match: return f"{gb_match.group(1)}GB"
    return None
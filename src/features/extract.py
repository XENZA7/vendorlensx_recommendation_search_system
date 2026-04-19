import ast
import re
import pandas as pd
from typing import Optional, Union, List

# ── Brand configuration ───────────────────────────────────────────────────────
KNOWN_BRANDS = [
    "Red Magic", "VGO TEL", "OnePlus",                          # multi-word first
    "Apple", "Samsung", "Oppo", "Vivo", "Realme", "Infinix",
    "Tecno", "Nokia", "Huawei", "Xiaomi", "Motorola", "Sparx",
    "Hisense", "Haier", "Honor", "Redmi", "Poco", "Itel",
    "Blackberry", "Google", "Nothing",
    "Microsoft", "Lenovo", "Dell", "Asus", "Acer", "MSI",
    "Razer", "HP", "Toshiba", "LG", "Panasonic",
    "Sony", "JBL", "Anker", "QCY", "Sennheiser", "Bose", "Audionic",
    "Fitbit", "Garmin", "Amazfit",
    "TCL", "Dawlance", "Orient", "PEL", "Kenwood",
]

BRAND_NORMALISE = {
    "hp": "HP", "Hp": "HP", "OPPO": "Oppo",
    "oppo": "Oppo", "DELL": "Dell", "SAMSUNG": "Samsung",
}

def _word_boundary_match(brand: str, text: str) -> bool:
    """Word-boundary match — prevents 'hp' matching inside 'cheap'."""
    pattern = r"(?<![a-zA-Z])" + re.escape(brand) + r"(?![a-zA-Z])"
    return bool(re.search(pattern, text, re.IGNORECASE))

def recover_brand(row: pd.Series) -> str:
    """Priority: 1) trust existing brand, 2) word-boundary title scan, 3) first cap word."""
    existing = row.get("brand", None)
    if pd.notna(existing) and str(existing).strip().lower() not in ("nan", "unknown", "none", ""):
        brand = str(existing).strip()
        return BRAND_NORMALISE.get(brand, brand)

    title = str(row.get("title", "")).strip()
    for brand in KNOWN_BRANDS:
        if _word_boundary_match(brand, title):
            return brand

    if not title:
        return "Unknown"
    for word in title.split():
        if word and word[0].isupper() and len(word) > 1:
            return word
    return "Unknown"

# ── Spec parsing ──────────────────────────────────────────────────────────────

KEY_ALIASES = {
    "Memory": "RAM", "Installed RAM": "RAM", "System Memory": "RAM",
    "Internal Memory": "RAM", "RAM Size": "RAM",
    "Internal Storage": "Storage", "Hard drive size": "Storage",
    "Hard Disk": "Storage", "HDD": "Storage", "SSD": "Storage",
    "Hard Drive": "Storage",
    "Processor Type": "Processor", "Processor Speed": "Processor", "CPU": "Processor",
    "Graphics memory": "GPU", "Graphics Memory": "GPU", "Graphics Card": "GPU",
    "Screen Size": "Screen", "Display Size": "Screen",
}

def parse_specs(spec_str: str) -> dict:
    """Parse specifications column (Python dict stored as string) with normalised keys."""
    try:
        raw = ast.literal_eval(str(spec_str))
        if not isinstance(raw, dict):
            return {}
        return {KEY_ALIASES.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}

def _parse_gb(value: str) -> Optional[int]:
    """Parse '8GB', '8 gb', '1TB' → integer GB."""
    if not value or not isinstance(value, str):
        return None
    value = value.split(",")[0].strip()
    m = re.search(r"(\d+)\s*[Tt][Bb]", value)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"(\d+)\s*[Gg][Bb]", value, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

VALID_RAM_GB     = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}
VALID_STORAGE_GB = {16, 32, 64, 128, 256, 512, 1024, 2048}

def extract_ram(specs_or_text) -> Optional[int]:
    """Extract RAM as integer GB from specs dict (primary) or raw string (fallback)."""
    if isinstance(specs_or_text, dict):
        val = _parse_gb(specs_or_text.get("RAM", ""))
        return val if val and val in VALID_RAM_GB else None
    if not isinstance(specs_or_text, str):
        return None
    m = re.search(r"(\d+)\s*[Gg][Bb]\s*(?:RAM|ram|memory)?", specs_or_text)
    if m:
        val = int(m.group(1))
        return val if val in VALID_RAM_GB else None
    return None

# FIX: Renamed from duplicate extract_ram to extract_storage
def extract_storage(specs_or_text) -> Optional[int]:
    """Extract storage as integer GB."""
    if isinstance(specs_or_text, dict):
        val = _parse_gb(specs_or_text.get("Storage", ""))
        return val if val and val in VALID_STORAGE_GB else None
    if not isinstance(specs_or_text, str):
        return None
    m = re.search(r"(\d+)\s*[Tt][Bb]", specs_or_text)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"(\d+)\s*[Gg][Bb]", specs_or_text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return val if val in VALID_STORAGE_GB else None
    return None

extract_gb = extract_ram

def extract_first_img(imgs_str: str) -> str:
    """Parse imgs column → first URL or placeholder."""
    PLACEHOLDER = "https://placehold.co/270x270?text=No+Image"
    try:
        imgs_list = ast.literal_eval(str(imgs_str))
        if isinstance(imgs_list, list) and imgs_list:
            return imgs_list[0]
    except Exception:
        pass
    urls = re.findall(r"(https?://[^\s'\"\\]]+)", str(imgs_str))
    return urls[0] if urls else PLACEHOLDER

# ── Text preparation ──────────────────────────────────────────────────────────

NOISE_PHRASES = [
    "PTA Approved", "Non PTA", "Non-PTA", "Price in Pakistan",
    "PTA approved", "pta approved", "non pta", "non-pta",
    "(Official)", "- Pakistan",
]

def clean_title(title: str) -> str:
    t = str(title)
    for phrase in NOISE_PHRASES:
        t = t.replace(phrase, " ")
    return " ".join(t.split())

def build_clean_content(row: pd.Series) -> str:
    """TF-IDF input: brand (×2) + category + cleaned title + specs."""
    brand    = str(row.get("brand", "")).strip()
    category = str(row.get("category", "")).strip()
    title    = clean_title(str(row.get("title", "")))
    specs    = str(row.get("extracted_specs", "")).strip()
    parts    = [brand, brand, category, title]
    if specs:
        parts.append(specs)
    return " ".join(p.lower() for p in parts if p).strip()

# ── Category-specific extraction ──────────────────────────────────────────────

def extract_specs_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured fields from specifications JSON per category."""
    df = df.copy()
    parsed = df["specifications"].apply(parse_specs)

    mob = df["category"] == "Mobile"
    df.loc[mob, "ram_gb"]       = parsed[mob].apply(extract_ram)
    df.loc[mob, "storage_gb"]   = parsed[mob].apply(extract_storage)
    df.loc[mob, "has_5g"]       = parsed[mob].apply(lambda s: str(s.get("5G Support", "")).lower() in ["yes", "true", "1"])
    df.loc[mob, "pta_approved"] = parsed[mob].apply(lambda s: "pta" in " ".join(str(v) for v in s.values()).lower())
    df.loc[mob, "screen_size"]  = parsed[mob].apply(lambda s: s.get("Screen") or s.get("Screen Size"))

    lap = df["category"] == "Laptop"
    df.loc[lap, "ram_gb"]      = parsed[lap].apply(extract_ram)
    df.loc[lap, "storage_gb"]  = parsed[lap].apply(extract_storage)
    df.loc[lap, "processor"]   = parsed[lap].apply(lambda s: s.get("Processor") or s.get("Processor Type"))
    df.loc[lap, "gpu"]         = parsed[lap].apply(lambda s: s.get("GPU") or s.get("Graphics Memory"))
    df.loc[lap, "screen_size"] = parsed[lap].apply(lambda s: s.get("Screen") or s.get("Screen Size"))

    ear = df["category"] == "Earbuds"
    df.loc[ear, "playtime_hrs"]  = parsed[ear].apply(lambda s: _parse_gb(str(s.get("Playtime", "") or s.get("Battery Life", ""))))
    df.loc[ear, "bluetooth_ver"] = parsed[ear].apply(lambda s: s.get("Bluetooth Version") or s.get("Bluetooth"))
    df.loc[ear, "waterproof"]    = parsed[ear].apply(lambda s: str(s.get("Waterproof", "")).lower() in ["yes", "true", "ipx4", "ipx5", "ipx7", "ipx8"])

    wat = df["category"] == "Watch"
    df.loc[wat, "water_resistant"] = parsed[wat].apply(lambda s: str(s.get("Water Resistant", "")).lower() in ["yes", "true"])
    df.loc[wat, "nfc"]             = parsed[wat].apply(lambda s: str(s.get("NFC", "")).lower() in ["yes", "true"])
    df.loc[wat, "sim_support"]     = parsed[wat].apply(lambda s: str(s.get("SIM Support", "")).lower() in ["yes", "true"])

    df["extracted_specs"] = parsed.apply(lambda s: " ".join(filter(None, [
        s.get("Processor", ""), s.get("GPU", ""), s.get("Screen", "")
    ])))
    return df

# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brand"]         = df.apply(recover_brand, axis=1)
    df                  = extract_specs_features(df)
    df["clean_content"] = df.apply(build_clean_content, axis=1)

    img_col = "imgs" if "imgs" in df.columns else "images" if "images" in df.columns else None
    df["first_img"] = df[img_col].apply(extract_first_img) if img_col else \
                      "https://placehold.co/270x270?text=No+Image"

    print(f"Feature engineering complete. {len(df.columns)} columns.")
    return df
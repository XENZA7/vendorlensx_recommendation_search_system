import requests
import streamlit as st
import pandas as pd
from typing import Union, Optional, List, Dict

# ── 1. CONFIGURATION & CONSTANTS ──────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"
FX_API_URL = "https://open.er-api.com/v6/latest/PKR"
PKR_TO_ZAR_FALLBACK = 0.065  # used only if the live rate can't be fetched

VENDOR_COLOURS = {
    "PriceOye":     "#7C3AED",
    "MEGA.PK":      "#D97706",
    "ComputerZone": "#0369A1",
    "TechGlobe":    "#059669",
    "Paklap":       "#DC2626",
}
DEFAULT_VENDOR_COLOUR = "#64748B"
PLACEHOLDER_IMG = "https://placehold.co/400x300?text=No+Image"

st.set_page_config(
    page_title="VendorLensX — Smart Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.vlx-header { padding: 1.5rem 0 1rem; border-bottom: 2px solid #7C3AED; margin-bottom: 1.5rem; }
.vlx-logo { font-size: 2rem; font-weight: 800; color: #7C3AED; letter-spacing: -1px; }
.vlx-logo span { color: #D97706; }
.vlx-tagline { font-size: 0.95rem; color: #64748B; margin-top: 2px; }
.vlx-img-wrap { width: 100%; height: 200px; overflow: hidden; background: #F8FAFC; display: flex; align-items: center; justify-content: center; }
.vlx-img-wrap img { width: 100%; height: 100%; object-fit: contain; }
.vlx-title { font-size: 1rem; font-weight: 800;color: #FFFFFF !important; line-height: 1.4; min-height: 3rem; margin-bottom: 0.6rem;display: -webkit-box; }
.vlx-badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem; }
.vlx-price { font-size: 1.1rem; font-weight: 700; color: #7C3AED; }
.vlx-price-sub { font-size: 0.75rem; color: #94A3B8; }
.vlx-result-bar { font-size: 0.85rem; color: #64748B; margin-bottom: 1rem; padding: 0.4rem 0; border-bottom: 1px solid #F1F5F9; }
.vlx-status { display: flex; align-items: center; gap: 6px; font-size: 0.78rem; }
.vlx-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.vlx-dot-green { background: #10B981; }
.vlx-dot-red { background: #EF4444; }
.vlx-rec-row { display: flex; align-items: flex-start; gap: 0.5rem; padding: 0.5rem 0; border-bottom: 1px solid #F1F5F9; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── 3. HELPERS ────────────────────────────────────────────────────────────────
def safe_float(val, default=0.0) -> float:
    try: return float(val)
    except (TypeError, ValueError): return default

def format_price(pkr_price: float, show_zar: bool) -> str:
    if show_zar: return f"R {pkr_price * PKR_TO_ZAR:,.0f}"
    return f"PKR {pkr_price:,.0f}"

def vendor_badge_html(vendor: str) -> str:
    colour = VENDOR_COLOURS.get(vendor, DEFAULT_VENDOR_COLOUR)
    return f'<span class="vlx-badge" style="background:{colour}">{vendor}</span>'

def truncate(text: str, limit: int = 55) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"

def api_get(path: str, params: Optional[Dict] = None):
    """
    GET against the API, returning (data, error) instead of silently
    swallowing failures. error is None on success (HTTP 200), otherwise a
    short dict describing what went wrong so the UI can show something
    other than an unexplained empty result.
    """
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=8)
        if r.status_code == 200:
            return r.json(), None
        return None, {"type": "http", "detail": f"API returned {r.status_code} for {path}"}
    except requests.exceptions.Timeout:
        return None, {"type": "timeout", "detail": f"Request to {path} timed out after 8s"}
    except requests.exceptions.ConnectionError:
        return None, {"type": "connection", "detail": "Could not reach the API — is it running?"}
    except Exception as e:
        return None, {"type": "unknown", "detail": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_fx_rate():
    """
    Live PKR->ZAR rate from open.er-api.com (no key required, updates daily).
    Falls back to PKR_TO_ZAR_FALLBACK on any failure so the app keeps working
    even if the FX API is unreachable.

    Returns (rate, as_of_str, is_live).
    """
    try:
        r = requests.get(FX_API_URL, timeout=5)
        data = r.json()
        if data.get("result") == "success" and "ZAR" in data.get("rates", {}):
            rate = float(data["rates"]["ZAR"])
            as_of = data.get("time_last_update_utc", "")
            return rate, as_of, True
    except Exception:
        pass
    return PKR_TO_ZAR_FALLBACK, None, False

PKR_TO_ZAR, FX_AS_OF, FX_IS_LIVE = fetch_live_fx_rate()

# ── 4. DATA LOADING & STATE ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_filters():
    data, error = api_get("/filters")
    if error:
        return {"brands": [], "categories": [], "min_price": 0, "max_price": 1000000, "_error": error}
    return data

def check_api_health() -> bool:
    data, _ = api_get("/health")
    return bool(data and data.get("status") == "online")

if "results" not in st.session_state: st.session_state.results = []
if "last_query_key" not in st.session_state: st.session_state.last_query_key = ""
if "recs_cache" not in st.session_state: st.session_state.recs_cache = {}

# ── 5. UI HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="vlx-header">
  <div class="vlx-logo">Vendor<span>Lens</span>X</div>
  <div class="vlx-tagline">by off_WRLD</div>           
  <div class="vlx-tagline">Intelligent Cross-Vendor Market Analytics</div>
</div>
""", unsafe_allow_html=True)

# ── 6. SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    api_ok = check_api_health()
    dot, label = ("vlx-dot-green", "API Online") if api_ok else ("vlx-dot-red", "API Offline")
    st.markdown(f'<div class="vlx-status"><span class="vlx-dot {dot}"></span>{label}</div>', unsafe_allow_html=True)

    if FX_IS_LIVE:
        st.caption(f"💱 Live rate: 1 PKR = {PKR_TO_ZAR:.4f} ZAR · updated {FX_AS_OF}")
        st.caption("Rates by [ExchangeRate-API](https://www.exchangerate-api.com)")
    else:
        st.caption(f"💱 Using fallback rate: 1 PKR = {PKR_TO_ZAR:.4f} ZAR (live rate unavailable)")
    st.markdown("---")

    filters = load_filters()
    if filters.get("_error"):
        st.warning(f"Couldn't load filters: {filters['_error']['detail']}")
    category = st.selectbox("Category", ["All"] + filters.get("categories", []))
    show_zar = st.toggle("Show prices in ZAR (R)", value=False)
    selected_brands = st.multiselect("Brand", filters.get("brands", []))

    min_p = safe_float(filters.get("min_price"), 0.0)
    max_p = safe_float(filters.get("max_price"), 1000000.0)

    if show_zar:
        pr = st.slider("Price range (R)", min_p*PKR_TO_ZAR, max_p*PKR_TO_ZAR, (min_p*PKR_TO_ZAR, max_p*PKR_TO_ZAR), format="R %d")
        p_min, p_max = pr[0]/PKR_TO_ZAR, pr[1]/PKR_TO_ZAR
    else:
        pr = st.slider("Price range (PKR)", min_p, max_p, (min_p, max_p), format="PKR %d")
        p_min, p_max = pr[0], pr[1]

    ram_filter = st.select_slider("Min RAM (GB)", [2,4,8,12,16,32,64], 2) if category in ("Mobile", "Laptop") else None
    n_results = st.select_slider("Results count", [6, 9, 12, 15, 20], 12)

# ── 7. SEARCH LOGIC ───────────────────────────────────────────────────────────
query = st.text_input("Search products...", placeholder="e.g. Gaming laptop", label_visibility="collapsed")
query_key = f"{query}|{category}|{selected_brands}|{p_min:.0f}|{p_max:.0f}|{ram_filter}|{n_results}"

if query and query_key != st.session_state.last_query_key:
    with st.spinner("Fetching market data..."):
        params = {"q": query, "min_price": p_min, "max_price": p_max, "n": n_results}
        if category != "All": params["category"] = category
        if ram_filter: params["min_ram"] = ram_filter

        errors = []
        if selected_brands:
            all_res, seen = [], set()
            for b in selected_brands:
                data, error = api_get("/search", {**params, "brand": b})
                if error:
                    errors.append(error)
                elif data:
                    for item in data:
                        if item['id'] not in seen: all_res.append(item); seen.add(item['id'])
            st.session_state.results = all_res
        else:
            data, error = api_get("/search", params)
            if error:
                errors.append(error)
            st.session_state.results = data if data else []

        if errors:
            # Show the most specific error once — no need to repeat per-brand duplicates
            first = errors[0]
            if first["type"] == "timeout":
                st.error("The search took too long to respond. Try again in a moment.")
            elif first["type"] == "connection":
                st.error("Can't reach the API — make sure it's running (`uvicorn api.main:app --reload`).")
            else:
                st.error(f"Search failed: {first['detail']}")

        st.session_state.last_query_key = query_key
        st.session_state.recs_cache = {}

# ── 8. DISPLAY ────────────────────────────────────────────────────────────────
results = st.session_state.results

if query:
    if results:
        # Result Bar
        vendor_count = len({r.get("vendor") for r in results})
        st.markdown(f'<div class="vlx-result-bar"><strong>{len(results)}</strong> products found across <strong>{vendor_count}</strong> vendors.</div>', unsafe_allow_html=True)

        # --- MARKET INSIGHTS (With KeyError Fix) ---
        with st.expander("📊 View Market Insights & Vendor Stats", expanded=False):
            df_res = pd.DataFrame(results)
            
            # Fix missing discount column
            if 'discount_pct' not in df_res.columns:
                df_res['discount_pct'] = df_res.apply(
                    lambda x: ((x.get('original_price', x['discounted_price']) - x['discounted_price']) 
                               / x.get('original_price', x['discounted_price']) * 100) 
                    if safe_float(x.get('original_price')) > 0 else 0, axis=1
                )

            avg_prices = df_res.groupby('vendor')['discounted_price'].mean()
            avg_discounts = df_res.groupby('vendor')['discount_pct'].mean()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Price Leader", avg_prices.idxmin(), f"Avg: R {avg_prices.min() * PKR_TO_ZAR:,.0f}")
            m2.metric("Discount King", avg_discounts.idxmax(), f"{avg_discounts.max():.1f}% Off")
            m3.metric("Market Leader", df_res['vendor'].value_counts().idxmax(), f"{df_res['vendor'].value_counts().max()} Items")
            
            st.write("**Average Price Comparison (ZAR)**")
            chart_df = (avg_prices * PKR_TO_ZAR).reset_index()
            chart_df.columns = ['Vendor', 'Avg Price (ZAR)']
            st.bar_chart(chart_df.set_index('Vendor'))

        # Product Gallery
        n_cols = min(3, len(results))
        cols = st.columns(n_cols if n_cols > 0 else 1)
        for i, item in enumerate(results):
            with cols[i % n_cols]:
                st.markdown(f'<div class="vlx-img-wrap"><img src="{item.get("first_img") or PLACEHOLDER_IMG}"/></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="vlx-card-body"><div class="vlx-title">{truncate(item["title"])}</div>{vendor_badge_html(item["vendor"])}<div class="vlx-price">{format_price(item["discounted_price"], show_zar)}</div></div>', unsafe_allow_html=True)
                
                with st.expander("Details & Similar"):
                    st.write(f"**Specs:** {item.get('extracted_specs', 'N/A')}")
                    if item["id"] not in st.session_state.recs_cache:
                        rec_data, rec_error = api_get(f"/recommend/{item['id']}")
                        st.session_state.recs_cache[item["id"]] = rec_data or []
                        if rec_error:
                            st.caption(f"⚠️ Couldn't load similar products: {rec_error['detail']}")
                    for r in st.session_state.recs_cache[item["id"]]:
                        st.markdown(f'<div class="vlx-rec-row"><span>{r["vendor"]}</span><span style="flex:1">{truncate(r["title"], 40)}</span><b>{format_price(r["discounted_price"], show_zar)}</b></div>', unsafe_allow_html=True)
    else:
        st.info("No products found for this search.")
else:
    st.markdown('<div class="vlx-empty"><h3>VendorLensX Analytics</h3>Enter a search term to compare the Pakistani electronics market.</div>', unsafe_allow_html=True)
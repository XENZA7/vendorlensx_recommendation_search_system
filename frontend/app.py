import streamlit as st
import requests
import pandas as pd

# 1 PKR to ZAR conversion (Current 2026 estimate)
PKR_TO_ZAR = 0.065

# ── 1. CONFIGURATION & STATE ──────────────────────────────────────────────────
st.set_page_config(page_title="VendorLensX", layout="wide", page_icon="🔍")

API_BASE_URL = "http://127.0.0.1:8000"

@st.cache_resource
def get_filter_data():
    try:
        response = requests.get(f"{API_BASE_URL}/filters")
        if response.status_code == 200:
            return response.json()
    except:
        return {"brands": [], "categories": [], "min_price": 0, "max_price": 1000000}

# ── 2. SIDEBAR - DYNAMIC FILTERS ──────────────────────────────────────────────
st.sidebar.header("🛠️ Search Filters")
filters = get_filter_data()

category = st.sidebar.selectbox("Category", ["All"] + filters["categories"])
brand = st.sidebar.multiselect("Brands", filters["brands"])

# Price Slider (Displayed in ZAR)
price_range_zar = st.sidebar.slider(
    "Price Range (ZAR)",
    min_value=float(filters["min_price"]) * PKR_TO_ZAR,
    max_value=float(filters["max_price"]) * PKR_TO_ZAR,
    value=(float(filters["min_price"]) * PKR_TO_ZAR, float(filters["max_price"]) * PKR_TO_ZAR),
    format="R %d"
)
ram_filter = None
if category in ["Mobile", "Laptop"]:
    ram_filter = st.sidebar.select_slider(
        "Minimum RAM (GB)", 
        options=[2, 4, 8, 12, 16, 32, 64],
        value=2
    )
# ── 3. MAIN AREA - SEARCH ─────────────────────────────────────────────────────
st.title("VendorLensX")
st.markdown("### Intelligent Cross-Vendor Search & Recommendations")

query = st.text_input("What are you looking for?", placeholder="e.g. Gaming laptop")

# ── 4. RESULTS ENGINE ─────────────────────────────────────────────────────────
if query:
    with st.spinner("Searching..."):
        # FIX: Convert ZAR back to PKR for the API request
        params = {
            "q": query,
            "min_price": price_range_zar[0] / PKR_TO_ZAR,
            "max_price": price_range_zar[1] / PKR_TO_ZAR,
            "n": 12
        }
        if category != "All": params["category"] = category
        if brand: params["brand"] = brand[0]

        try:
            res = requests.get(f"{API_BASE_URL}/search", params=params)
            results = res.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            results = []

    if results:
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                st.image(item["first_img"], use_container_width=True)
                
                # Convert PKR price to ZAR for the display
                price_zar = item["discounted_price"] * PKR_TO_ZAR
                
                vendor_color = "orange" if "Mega" in item["vendor"] else "blue"
                st.markdown(f"**{item['title'][:50]}...**")
                # Updated Label to R (ZAR)
                st.markdown(f":{vendor_color}[**{item['vendor']}**] | **R {price_zar:,.2f}**")
                
                with st.expander("View Details & Similar"):
                    st.write(f"**Specs:** {item.get('extracted_specs', 'N/A')}")
                    
                    rec_res = requests.get(f"{API_BASE_URL}/recommend/{item['id']}")
                    if rec_res.status_code == 200:
                        recs = rec_res.json()
                        st.markdown("---")
                        st.markdown(f"##### 🤖 Similar (from {len(set(r['vendor'] for r in recs))} Vendors)")
                        for r in recs:
                            rec_zar = r['discounted_price'] * PKR_TO_ZAR
                            st.write(f"🔹 {r['title']} - **R {rec_zar:,.2f}**")
    else:
        st.warning("No products found.")
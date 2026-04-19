import streamlit as st
import requests
import pandas as pd

# ── 1. CONFIGURATION & STATE ──────────────────────────────────────────────────
st.set_page_config(page_title="VendorLensX", layout="wide", page_icon="🔍")

API_BASE_URL = "http://127.0.0.1:8000"

@st.cache_resource
def get_filter_data():
    """Fetch initial filter bounds from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/filters")
        if response.status_code == 200:
            return response.json()
    except:
        return {"brands": [], "categories": [], "min_price": 0, "max_price": 1000000}
    return None

# ── 2. SIDEBAR - DYNAMIC FILTERS ──────────────────────────────────────────────
st.sidebar.header("🛠️ Search Filters")
filters = get_filter_data()

category = st.sidebar.selectbox("Category", ["All"] + filters["categories"])
brand = st.sidebar.multiselect("Brands", filters["brands"])

# Price Slider (Log scale feel handled by streamlit range)
price_range = st.sidebar.slider(
    "Price Range (PKR)",
    min_value=float(filters["min_price"]),
    max_value=float(filters["max_price"]),
    value=(float(filters["min_price"]), float(filters["max_price"]))
)

# Tech-specific filters (only show for Mobile/Laptop)
ram_filter = None
if category in ["Mobile", "Laptop"]:
    ram_filter = st.sidebar.select_slider("Minimum RAM (GB)", options=[2, 4, 8, 12, 16, 32, 64])

# ── 3. MAIN AREA - SEARCH ─────────────────────────────────────────────────────
st.title(" VendorLensX")
st.markdown("### Intelligent Cross-Vendor Product Search & Comparison")

query = st.text_input("What are you looking for?", placeholder="e.g. Gaming laptop under 200k")

# ── 4. RESULTS ENGINE ─────────────────────────────────────────────────────────
if query:
    with st.spinner("Searching across vendors..."):
        # Build API params
        params = {
            "q": query,
            "min_price": price_range[0],
            "max_price": price_range[1],
            "n": 12
        }
        if category != "All": params["category"] = category
        if brand: params["brand"] = brand[0] # API takes single brand or handle list

        try:
            res = requests.get(f"{API_BASE_URL}/search", params=params)
            results = res.json()
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
            results = []

    if results:
        # ── 5. PRODUCT GALLERY ────────────────────────────────────────────────
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                st.image(item["first_img"], use_container_width=True)
                
                # VENDOR TRANSPARENCY: The Badge
                vendor_color = "orange" if "Mega" in item["vendor"] else "blue"
                st.markdown(f"**{item['title'][:50]}...**")
                st.markdown(f":{vendor_color}[**{item['vendor']}**] | Rs. {item['discounted_price']:,}")
                
                # EXPAND FOR DETAILS & RECOMMENDATIONS
                with st.expander("View Details & Similar"):
                    st.write(f"**Specs:** {item.get('extracted_specs', 'N/A')}")
                    
                    # Call Recommendations
                    rec_res = requests.get(f"{API_BASE_URL}/recommend/{item['id']}")
                    if rec_res.status_code == 200:
                        recs = rec_res.json()
                        st.markdown(f"---")
                        st.markdown(f"##### 🤖 Similar Products from {len(set(r['vendor'] for r in recs))} Vendors")
                        
                        for r in recs:
                            st.write(f"🔹 {r['title']} - **{r['vendor']}** (Rs. {r['discounted_price']:,})")
    else:
        st.warning("No products found matching those criteria.")
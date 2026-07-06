import streamlit as st
import requests
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Recommendation Hub",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS
st.markdown("""
<style>
    /* Gradient Title background */
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px 0 rgba(0,0,0,0.1);
        text-align: center;
    }
    .title-container h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
    }
    .title-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Metrics and cards styling */
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Custom connection badges */
    .status-badge {
        font-weight: 600;
        padding: 0.4rem 0.8rem;
        border-radius: 50px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .status-connected {
        background-color: #c6f6d5;
        color: #22543d;
    }
    .status-disconnected {
        background-color: #fed7d7;
        color: #742a2a;
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration (defaults to localhost, but supports custom host for Docker/Cloud deployments)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Sample data for ease of exploration
SAMPLE_USERS = {
    "Warm User 1 (Rich History)": "0000366f3b9a7992bf8c76cfdf3221e2",
    "Warm User 2 (Rich History)": "0000b849f77a49e4a4ce2b2a4ca5be3f",
    "Warm User 3 (Rich History)": "0000f46a3911fa3c0805444483337064",
    "New/Cold-Start User": "nonexistent_new_customer_99"
}

SAMPLE_PRODUCTS = {
    "Watches/Gifts Product": "53b36df67ebb7c41585e8d54d6772e08",
    "Furniture/Decor Product": "aca2eb7d00ea1a7b8ebd4e68314663af",
    "Garden/Tools Product": "422879e10f46682990de24d770e7f83d",
    "Bed/Bath/Table Product": "99a4788cb24856965c36a24e339b6058"
}

BRAZIL_STATES = ["Global (No State Filter)", "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "CE", "DF"]

# App header
st.markdown("""
<div class="title-container">
    <h1>🛍️ E-Commerce Recommendation Hub</h1>
    <p>Personalized product suggestions & content similarity API client</p>
</div>
""", unsafe_allow_html=True)

# Helper function to check health status of the API
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy" and data.get("models_loaded", False)
    except requests.exceptions.RequestException:
        pass
    return False

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ System Status")
api_online = check_api_health()

if api_online:
    st.sidebar.markdown('<span class="status-badge status-connected">● API & Models Online</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge status-disconnected">● API Offline / Unreachable</span>', unsafe_allow_html=True)
    st.sidebar.warning("Ensure the FastAPI server is running in the background by calling: `uvicorn api.app:app --reload`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Control Panel")
recommendation_limit = st.sidebar.slider("Number of recommendations (n)", min_value=1, max_value=20, value=10)

# Main App View Tabs
tab1, tab2 = st.tabs(["👤 Personalized User Recommendations", "🏷️ Similar Products Lookup"])

# TAB 1: User Recommendations
with tab1:
    st.subheader("Compute User Recommendations")
    st.markdown("Generates a personalized list of recommendations. If the customer has past transactions, the system serves collaborative candidates. Otherwise, it serves popularity-based fallback items.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Preloaded user select
        user_select_opt = st.selectbox(
            "Quick-select a test user:", 
            options=list(SAMPLE_USERS.keys()),
            help="Select one of our pre-identified customer IDs to see how the system handles returning users vs. new users."
        )
        custom_user_id = st.text_input("Or enter a custom User Unique ID:", value=SAMPLE_USERS[user_select_opt])
        
    with col2:
        state_select = st.selectbox(
            "State filter (cold-start regional fallback):",
            options=BRAZIL_STATES,
            index=0
        )
        state_code = None if state_select == "Global (No State Filter)" else state_select

    if st.button("Generate Recommendations", type="primary", disabled=not api_online):
        with st.spinner("Fetching recommendations from backend API..."):
            query_params = {
                "user_id": custom_user_id,
                "n": recommendation_limit
            }
            if state_code:
                query_params["state"] = state_code
                
            try:
                response = requests.get(f"{API_BASE_URL}/recommend", params=query_params)
                if response.status_code == 200:
                    res_json = response.json()
                    recs = res_json.get("recommendations", [])
                    
                    if recs:
                        df_recs = pd.DataFrame(recs)
                        
                        # Clean columns names for better visual appeal
                        df_recs.columns = ["Product ID", "Category", "Average Price ($)", "Avg Review Score (1-5)", "Total Sales Count"]
                        
                        # Detect model used (simplified heuristic)
                        is_cold = custom_user_id == SAMPLE_USERS["New/Cold-Start User"] or "Cold-start" in response.text
                        user_type_label = "🆕 Cold Start User (Popularity fallbacks)" if is_cold else "🔥 Returning Warm User (ALS Collaborative Filter)"
                        
                        st.markdown(f"#### **Recommendation Segment:** `{user_type_label}`")
                        
                        # Display recommendations Table
                        st.dataframe(df_recs.style.format({
                            "Average Price ($)": "${:.2f}",
                            "Avg Review Score (1-5)": "{:.2f} ⭐",
                            "Total Sales Count": "{:,} sales"
                        }), use_container_width=True)
                        
                        # Plot charts
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            st.write("**Recommended Product Prices**")
                            st.bar_chart(df_recs.set_index("Product ID")["Average Price ($)"])
                        with chart_col2:
                            st.write("**Recommended Product Ratings**")
                            st.bar_chart(df_recs.set_index("Product ID")["Avg Review Score (1-5)"])
                            
                    else:
                        st.warning("No recommendations returned from the server.")
                else:
                    st.error(f"Error fetching recommendations: {response.text}")
            except Exception as e:
                st.error(f"Failed to communicate with API server: {e}")

# TAB 2: Similar Products Lookup
with tab2:
    st.subheader("Find Content-Based Similar Products")
    st.markdown("Searches for products with similar descriptions, price ranges, and review profiles. Used on product detail pages to prompt cross-selling.")
    
    col_prod1, col_prod2 = st.columns([2, 1])
    
    with col_prod1:
        prod_select_opt = st.selectbox(
            "Quick-select a test product ID:", 
            options=list(SAMPLE_PRODUCTS.keys()),
            help="Select one of the sample product IDs to find similar items in the catalog."
        )
        custom_prod_id = st.text_input("Or enter a custom Product ID:", value=SAMPLE_PRODUCTS[prod_select_opt])
        
    if st.button("Find Similar Products", type="primary", disabled=not api_online):
        with st.spinner("Fetching similar items from backend API..."):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/recommend/similar", 
                    params={"product_id": custom_prod_id, "n": recommendation_limit}
                )
                if response.status_code == 200:
                    res_json = response.json()
                    sim_items = res_json.get("recommendations", [])
                    
                    if sim_items:
                        df_sim = pd.DataFrame(sim_items)
                        df_sim.columns = ["Product ID", "Category", "Average Price ($)", "Avg Review Score (1-5)", "Total Sales Count"]
                        
                        st.markdown(f"#### **Top Similar Products for Product ID:** `{custom_prod_id}`")
                        
                        st.dataframe(df_sim.style.format({
                            "Average Price ($)": "${:.2f}",
                            "Avg Review Score (1-5)": "{:.2f} ⭐",
                            "Total Sales Count": "{:,} sales"
                        }), use_container_width=True)
                        
                        # Compare attributes
                        st.write("**Price Comparison**")
                        st.area_chart(df_sim.set_index("Product ID")["Average Price ($)"])
                    else:
                        st.warning("No similar products found.")
                elif response.status_code == 404:
                    st.error(f"Product ID not found in database: {response.json().get('detail')}")
                else:
                    st.error(f"Error fetching similar products: {response.text}")
            except Exception as e:
                st.error(f"Failed to communicate with API server: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Ronin Wallet Clustering Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Try to load models; handle saved label arrays (not estimators) gracefully and load optional agg NN mapper."""
    base_dirs = ["saved_models", "models"]

    def try_load(path):
        try:
            if os.path.exists(path):
                return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
        return None

    def is_estimator(obj):
        return hasattr(obj, 'predict') or hasattr(obj, 'fit_predict')

    saved_label_arrays = {}
    kmeans = agg = dbscan = mb_kmeans = scaler = feature_names = agg_nn = None

    search_paths = [
        ("kmeans", [os.path.join(d, p) for d, p in [("saved_models", "ronin_kmeans_model.pkl"), ("saved_models","kmeans_model.pkl"), ("models","kmeans_model.pkl")]]),
        ("agg", [os.path.join(d, p) for d, p in [("saved_models", "ronin_agg_model.pkl"), ("saved_models","agg_model.pkl"), ("models","agg_model.pkl")]]),
        ("dbscan", [os.path.join(d, p) for d, p in [("saved_models", "ronin_dbscan_model.pkl"), ("saved_models","dbscan_model.pkl"), ("models","dbscan_model.pkl")]]),
        ("mb_kmeans", [os.path.join(d, p) for d, p in [("saved_models", "ronin_mb_kmeans_model.pkl"), ("saved_models","mb_kmeans_model.pkl"), ("models","mb_kmeans_model.pkl")]]),
        ("scaler", [os.path.join(d, p) for d, p in [("saved_scalers","ronin_scaler.pkl"), ("saved_scalers","scaler.pkl"), ("models","scaler.pkl")]]),
        ("feature_names", [os.path.join(d, p) for d, p in [("saved_scalers","ronin_scaled_features.pkl"), ("saved_scalers","feature_names.pkl"), ("models","feature_names.pkl")]])
    ]

    for key, paths in search_paths:
        for p in paths:
            obj = try_load(p)
            if obj is None:
                continue
            if key in ("kmeans", "agg", "dbscan", "mb_kmeans") and not is_estimator(obj):
                saved_label_arrays[key] = obj
                st.warning(f"{key}: loaded saved labels instead of an estimator; prediction disabled for this model.")
                obj = None

            if key == "kmeans":
                kmeans = obj
            elif key == "agg":
                agg = obj
            elif key == "dbscan":
                dbscan = obj
            elif key == "mb_kmeans":
                mb_kmeans = obj
            elif key == "scaler":
                scaler = obj
            elif key == "feature_names":
                feature_names = obj

            # stop searching this key once a value is found (estimator or saved labels)
            break

    # Try to load an agg NN mapper explicitly
    agg_nn = try_load(os.path.join('saved_models', 'agg_nn.pkl')) or try_load(os.path.join('models', 'agg_nn.pkl'))

    # Try to load explicit agg labels array if present
    agg_label_array = try_load(os.path.join('saved_models', 'agg_labels.pkl')) or try_load(os.path.join('saved_models', 'ronin_agg_labels.pkl'))
    if agg_label_array is not None:
        saved_label_arrays['agg'] = agg_label_array
        st.info('Loaded saved Agglomerative label array for NN mapping.')

    if all(v is None for v in [kmeans, agg, dbscan, mb_kmeans, scaler]):
        st.warning("Some or all model files were not found. Prediction features will be disabled until models are available.")

    return kmeans, agg, dbscan, mb_kmeans, scaler, feature_names, saved_label_arrays, agg_nn

@st.cache_data
def load_data():
    possible_paths = [
        'data/ronin_clusters_clean.csv',
        'ronin_clusters_clean.csv',
        'dune_wallet_data.csv',
        r"C:\Users\DELL\Desktop\ronin_app\ronin_clusters_clean.csv",
        r"C:\Users\DELL\.vscode\extensions\dune_wallet_data.csv"
    ]

    for p in possible_paths:
        if not os.path.exists(p):
            continue
        try:
            data = pd.read_csv(p)

            # Normalize column names and ensure both churn variants exist
            if 'Cluster' in data.columns and 'cluster' not in data.columns:
                data = data.rename(columns={'Cluster':'cluster'})
            if 'days_since_last_transactions' in data.columns and 'days_since_last_transaction' not in data.columns:
                data['days_since_last_transaction'] = data['days_since_last_transactions']
            if 'days_since_last_transaction' in data.columns and 'days_since_last_transactions' not in data.columns:
                data['days_since_last_transactions'] = data['days_since_last_transaction']

            return data
        except Exception as e:
            st.error(f"Error loading data from {p}: {e}")
            return None

    st.error("Data file not found. Place 'ronin_clusters_clean.csv' in the app folder or update the path.")
    return None

# Sidebar
st.sidebar.title("🎮 Ronin Wallet Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Cluster Explorer", "🎯 User Predictor", "💡 Business Intelligence"]
)

# Load everything
data = load_data()
kmeans_model, agg_model, dbscan_model, mb_kmeans_model, scaler, feature_names, saved_label_arrays, agg_nn = load_models()

if data is None:
    st.error("Failed to load data. Please add the CSV and reload the app.")
    st.stop()

# PAGE: Dashboard
if page == "📊 Dashboard":
    st.markdown('<p class="main-header">🎮 Ronin Wallet Clustering Dashboard</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Wallets", f"{len(data):,}")
    with col2:
        st.metric("Total Value", f"{data['value_sent'].sum():,.0f} RON")
    with col3:
        st.metric("Avg Transactions", f"{data['transaction_count'].mean():.0f}")
    with col4:
        if 'cluster' in data.columns:
            st.metric("User Segments", f"{data['cluster'].nunique()}")

# PAGE: Cluster Explorer
elif page == "🔍 Cluster Explorer":
    st.markdown('<p class="main-header"> Deep Dive: Cluster Analysis</p>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select Clustering Model", ["K-Means", "Agglomerative", "DBSCAN", "Mini-Batch K-Means"])

    # Prefer explicit cluster columns if present; fallback to generic 'cluster'
    candidate_map = {
        'K-Means': ['kmeans_cluster', 'cluster'],
        'Agglomerative': ['agg_cluster_labels', 'cluster'],
        'DBSCAN': ['dbscan_cluster', 'cluster'],
        'Mini-Batch K-Means': ['mb_kmeans_cluster', 'cluster']
    }

    candidates = candidate_map.get(model_choice, ['cluster'])
    cluster_col = next((c for c in candidates if c in data.columns), 'cluster')
    if cluster_col not in data.columns:
        st.warning(f"Column '{cluster_col}' not found; falling back to 'cluster'.")
        cluster_col = 'cluster'

    clusters = sorted(data[cluster_col].unique())
    selected = st.selectbox('Select Cluster to Explore', clusters)
    cluster_data = data[data[cluster_col] == selected]

    st.subheader('Cluster Overview')
    st.write(cluster_data.describe().T)

# PAGE: User Predictor
elif page == "🎯 User Predictor":
    st.markdown('<p class="main-header"> Predict User Segment</p>', unsafe_allow_html=True)
    transaction_count = st.number_input("Transaction Count", min_value=0, value=150)
    days_active = st.number_input("Days Active", min_value=0, value=20)
    days_since_last = st.number_input("Days Since Last Transaction", min_value=0, value=2)
    value_sent = st.number_input("Total Value Sent (RON)", min_value=0.0, value=100.0)
    avg_value = st.number_input("Average Value per Transaction (RON)", min_value=0.0, value=1.0)
    unique_addresses = st.number_input("Unique Addresses", min_value=0, value=10)
    avg_gas = st.number_input("Average Gas Used", min_value=0, value=100000)
    wallet_age = st.number_input("Wallet Age (Days)", min_value=0, value=30)

    if st.button("🔮 Predict Segment"):
        if scaler is None or kmeans_model is None:
            st.error("Models not loaded. Prediction is unavailable.")
        else:
            arr = np.array([[transaction_count, days_active, days_since_last, value_sent, avg_value, unique_addresses, avg_gas, wallet_age]])
            scaled = scaler.transform(arr)
            pred = kmeans_model.predict(scaled)[0]
            st.metric("K-Means", f"Cluster {pred}")

            # Agglomerative mapping (if available)
            if agg_nn is not None and saved_label_arrays.get('agg') is not None:
                try:
                    idx = agg_nn.kneighbors(scaled, return_distance=False)[0][0]
                    agg_pred = int(saved_label_arrays['agg'][idx])
                    st.metric("Agglomerative", f"Cluster {agg_pred}")
                except Exception as e:
                    st.error(f"Agglomerative mapping failed: {e}")
            elif agg_model is not None:
                st.warning("Agglomerative: model available but cannot predict new samples reliably (no 'predict' method). Provide an NN mapping to enable predictions.")

# PAGE: Business Intelligence
elif page == "💡 Business Intelligence":
    st.markdown('<p class="main-header"> Business Intelligence & Alerts</p>', unsafe_allow_html=True)
    whale_threshold = st.slider("Whale Threshold (RON)", 1000, 10000, 5000, 500)
    whales = data[data['value_sent'] >= whale_threshold]
    st.metric("Total Whales", f"{len(whales):,}")

# Footer
st.markdown('---')
st.markdown("<div style='text-align: center; color: gray; padding: 1rem;'><p>🎮 Ronin Wallet Clustering Dashboard | Built with Streamlit</p></div>", unsafe_allow_html=True)

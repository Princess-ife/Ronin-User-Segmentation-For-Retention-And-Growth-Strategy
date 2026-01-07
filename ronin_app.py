import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Ronin Wallet Clustering Dashboard",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
from pathlib import Path

@st.cache_resource
def load_models():
    """Attempt to load models and optional mapping assets from common locations.
    Returns (kmeans, agg, dbscan, mb_kmeans, scaler, feature_names, saved_label_arrays, agg_nn)
    """
    base = Path(__file__).resolve().parent

    def try_load(paths):
        for p in paths:
            p = Path(p)
            try:
                if p.exists():
                    return joblib.load(p)
            except Exception as e:
                st.warning(f"Failed to load {p}: {e}")
        return None

    def is_estimator(obj):
        return hasattr(obj, 'predict') or hasattr(obj, 'fit_predict')

    saved_label_arrays = {}

    kmeans = try_load([
        base / 'saved_models' / 'ronin_kmeans_model.pkl',
        base / 'saved_models' / 'kmeans_model.pkl',
        base / 'models' / 'kmeans_model.pkl'
    ])
    if kmeans is not None and not is_estimator(kmeans):
        saved_label_arrays['kmeans'] = kmeans
        st.warning('K-Means: loaded saved labels instead of an estimator; prediction disabled for this model.')
        kmeans = None

    agg = try_load([
        base / 'saved_models' / 'ronin_agg_model.pkl',
        base / 'saved_models' / 'agg_model.pkl',
        base / 'models' / 'agg_model.pkl'
    ])
    if agg is not None and not is_estimator(agg):
        saved_label_arrays['agg'] = agg
        st.warning('Agglomerative: loaded saved labels instead of an estimator; prediction disabled for this model.')
        agg = None

    dbscan = try_load([
        base / 'saved_models' / 'ronin_dbscan_model.pkl',
        base / 'saved_models' / 'dbscan_model.pkl',
        base / 'models' / 'dbscan_model.pkl'
    ])
    if dbscan is not None and not is_estimator(dbscan):
        saved_label_arrays['dbscan'] = dbscan
        st.warning('DBSCAN: loaded saved labels instead of an estimator; prediction disabled for this model.')
        dbscan = None

    mb_kmeans = try_load([
        base / 'saved_models' / 'ronin_mb_kmeans_model.pkl',
        base / 'saved_models' / 'mb_kmeans_model.pkl',
        base / 'models' / 'mb_kmeans_model.pkl'
    ])
    if mb_kmeans is not None and not is_estimator(mb_kmeans):
        saved_label_arrays['mb_kmeans'] = mb_kmeans
        st.warning('Mini-Batch KMeans: loaded saved labels instead of an estimator; prediction disabled for this model.')
        mb_kmeans = None

    scaler = try_load([
        base / 'saved_scalers' / 'ronin_scaler.pkl',
        base / 'saved_scalers' / 'scaler.pkl',
        base / 'models' / 'scaler.pkl'
    ])

    feature_names = try_load([
        base / 'saved_scalers' / 'ronin_scaled_features.pkl',
        base / 'saved_scalers' / 'feature_names.pkl',
        base / 'models' / 'feature_names.pkl'
    ])

    # Optional: Agglomerative nearest-neighbor mapper & explicit agg labels
    agg_nn = try_load([
        base / 'saved_models' / 'agg_nn.pkl',
        base / 'models' / 'agg_nn.pkl'
    ])

    # Also look for saved agg label arrays by common names
    agg_label_array = try_load([
        base / 'saved_models' / 'agg_labels.pkl',
        base / 'saved_models' / 'ronin_agg_labels.pkl'
    ])
    if agg_label_array is not None:
        saved_label_arrays['agg'] = agg_label_array
        st.info('Loaded saved Agglomerative label array for NN mapping.')

    if all(v is None for v in [kmeans, agg, dbscan, mb_kmeans, scaler]):
        st.warning("Some or all model files were not found. Prediction features will be disabled until models are available.")

    return kmeans, agg, dbscan, mb_kmeans, scaler, feature_names, saved_label_arrays, agg_nn


@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent

    # Try likely local and absolute paths
    possible_paths = [
        base / 'data' / 'ronin_clusters_clean.csv',
        base / 'ronin_clusters_clean.csv',
        base / 'dune_wallet_data.csv',
        base / 'ronin_clusters_clean.csv',
        Path(r"C:\Users\DELL\Desktop\ronin_app\ronin_clusters_clean.csv"),
        Path(r"C:\Users\DELL\.vscode\extensions\dune_wallet_data.csv")
    ]

    for p in possible_paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            data = pd.read_csv(p)

            # Normalize known / inconsistent column names
            if 'days_since_last_transactions' in data.columns and 'days_since_last_transaction' not in data.columns:
                data = data.rename(columns={'days_since_last_transactions': 'days_since_last_transaction'})
            if 'Cluster' in data.columns and 'cluster' not in data.columns:
                data = data.rename(columns={'Cluster': 'cluster'})

            # Ensure both variants exist to avoid KeyErrors elsewhere
            if 'days_since_last_transaction' in data.columns and 'days_since_last_transactions' not in data.columns:
                data['days_since_last_transactions'] = data['days_since_last_transaction']

            return data
        except Exception as e:
            st.error(f"Error loading data from {p}: {e}")
            return None

    st.error("Data file not found in expected locations. Please place 'ronin_clusters_clean.csv' in the project folder or provide a valid path.")
    return None

# Cluster descriptions
DEFAULT_CLUSTER_INFO = {
    0: {"name": "Casual Gamers", "color": "#3498db"},
    1: {"name": "Mass Market Users", "color": "#2ca02c"},
    2: {"name": "Bot/Grinder",  "color": "#d62728"},
    3: {"name": "High-Value Players",  "color": "#9467bd"},
    4: {"name": "Moderate Players", "color": "#ff7f0e"}
}

CLUSTER_DESCRIPTIONS = {
    'K-Means': DEFAULT_CLUSTER_INFO,
    'Agglomerative': DEFAULT_CLUSTER_INFO,
    'DBSCAN': DEFAULT_CLUSTER_INFO,
    'Mini-Batch K-Means': DEFAULT_CLUSTER_INFO
}


def format_cluster_label(model_name, cluster_id):
    """Return a human-friendly label for a cluster..."""
    descr = CLUSTER_DESCRIPTIONS.get(model_name, {})
    info = descr.get(cluster_id)
    
    if isinstance(info, dict):
        return info.get('name', f"Cluster {cluster_id}")
    if isinstance(info, str):
        return f"Cluster {cluster_id}"
    

def get_cluster_color(model_name, cluster_id):
    info = CLUSTER_DESCRIPTIONS.get(model_name, {}).get(cluster_id)
    if isinstance(info, dict):
        return info.get('color')
    return None

# Sidebar
st.sidebar.title("Ronin Wallet Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Cluster Explorer", "User Predictor", "Business Intelligence"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**
- Analyze Ronin wallet behavior
- Identify user segments
- Predict churn risk
- Track business metrics
""")

# Load data
data = load_data()
kmeans_model, agg_model, dbscan_model, mb_kmeans_model, scaler, feature_names, saved_label_arrays, agg_nn = load_models()

# Validate feature consistency between models, scaler, and saved features to prevent runtime ValueErrors
def _check_and_disable_if_mismatch(model, model_name):
    if model is None:
        return None
    model_n = getattr(model, 'n_features_in_', None)
    scaler_n = getattr(scaler, 'n_features_in_', None) if scaler is not None else None
    feature_list_n = len(feature_names) if feature_names is not None else None

    # If model reports a feature count and it conflicts with scaler or saved feature list, disable it
    if model_n is not None and ((scaler_n is not None and model_n != scaler_n) or (feature_list_n is not None and model_n != feature_list_n)):
        conflict_ref = scaler_n if scaler_n is not None else feature_list_n
        st.warning(f"{model_name} feature mismatch: model expects {model_n} features but scaler/feature list indicates {conflict_ref}. Prediction will be disabled for {model_name}. Please re-run training/save cells to refresh artifacts.")
        return None
    return model

kmeans_model = _check_and_disable_if_mismatch(kmeans_model, 'K-Means')
mb_kmeans_model = _check_and_disable_if_mismatch(mb_kmeans_model, 'Mini-Batch K-Means')
if agg_nn is not None:
    # ensure the NN mapper matches the scaler/features as well
    nn_n = getattr(agg_nn, 'n_features_in_', None)
    if nn_n is not None and ((scaler is not None and getattr(scaler, 'n_features_in_', None) is not None and nn_n != getattr(scaler, 'n_features_in_', None)) or (feature_names is not None and nn_n != len(feature_names))):
        st.warning(f"Agglomerative NN mapping feature mismatch: expects {nn_n} but scaler/features differ. Disabling agg_nn mapping.")
        agg_nn = None

# Display model/data status in the sidebar so you can see which assets were found
with st.sidebar.expander("Model & Data Status", expanded=True):
    st.write("**Data file:**")
    if data is None:
        st.error("Missing: 'ronin_clusters_clean.csv' — please run the notebook to create it or place it in the project folder.")
    else:
        st.success("Found: data loaded successfully")
        st.write(f"Rows: {len(data):,}")

    st.write("---")
    st.write("**Saved models / scalers:**")
    def status(name, model_obj, label_key):
        if model_obj is not None:
            try:
                cls = model_obj.__class__.__name__
            except Exception:
                cls = "estimator"
            st.write(f"{name}: ✅ (estimator: {cls})")
        elif saved_label_arrays.get(label_key) is not None:
            st.write(f"{name}: ⚠️ saved labels found (model not available)")
        else:
            st.write(f"{name}: ❌ missing")

    status('K-Means', kmeans_model, 'kmeans')
    status('Agglomerative', agg_model, 'agg')
    status('Agglomerative (NN mapping)', agg_nn, 'agg_nn')
    status('DBSCAN', dbscan_model, 'dbscan')
    status('Mini-Batch KMeans', mb_kmeans_model, 'mb_kmeans')
    status('Scaler', scaler, 'scaler')
    if feature_names is not None:
        st.write(f"Feature names: {', '.join(feature_names[:10])}...")

    # Diagnostics: show feature counts to help debug mismatches
    st.markdown('---')
    st.write('**Model <-> Scaler Feature Diagnostics**')

    def _show_diag(name, model_obj):
        model_n = getattr(model_obj, 'n_features_in_', None) if model_obj is not None else None
        scaler_n = getattr(scaler, 'n_features_in_', None) if scaler is not None else None
        feat_n = len(feature_names) if feature_names is not None else None
        status = 'enabled' if model_obj is not None else 'disabled'
        st.write(f"{name}: status={status}; model_n={model_n}; scaler_n={scaler_n}; saved_feat_n={feat_n}")

    _show_diag('K-Means', kmeans_model)
    _show_diag('Mini-Batch KMeans', mb_kmeans_model)
    if agg_nn is not None:
        st.write(f"Agglomerative NN mapper: n_features_in_={getattr(agg_nn,'n_features_in_',None)}")

    if st.button('🔁 Reload models from disk'):
        try:
            load_models.clear()
            st.experimental_rerun()
        except Exception as e:
            st.warning('Unable to programmatically clear cache — please restart the Streamlit app to reload models.')

if data is None:
    st.error("Failed to load data. Please ensure 'ronin_clusters_clean.csv' exists.")
    st.stop()


# PAGE 1: DASHBOARD

if page == "Dashboard":
    st.markdown('<p class="main-header">Ronin Wallet Clustering Dashboard</p>', unsafe_allow_html=True)
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Wallets", f"{len(data):,}")
    
    with col2:
        total_value = data['value_sent'].sum()
        st.metric("Total Value", f"{total_value:,.0f} RON")
    
    with col3:
        avg_txns = data['transaction_count'].mean()
        st.metric("Avg Transactions", f"{avg_txns:.0f}")
    
    with col4:
        if 'cluster' in data.columns:
            n_clusters = data['cluster'].nunique()
            st.metric("User Segments", f"{n_clusters}")
    
    st.markdown("---")
    
    # Cluster Distribution
    st.subheader("User Segment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'cluster' in data.columns:
            cluster_counts = data['cluster'].value_counts().sort_index()
            labels = [format_cluster_label('K-Means', int(i)) for i in cluster_counts.index]
            fig = px.pie(
                values=cluster_counts.values,
                names=labels,
                title="Distribution by K-Means Cluster",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'cluster' in data.columns:
            cluster_value = data.groupby('cluster')['value_sent'].sum().sort_index()
            labels = [format_cluster_label('K-Means', int(i)) for i in cluster_value.index]
            fig = px.bar(
                x=labels,
                y=cluster_value.values,
                title="Total Value by Cluster",
                labels={'x': 'Cluster', 'y': 'Total Value (RON)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Key Metrics by Cluster
    st.subheader("Key Metrics by Segment")
    
    if 'cluster' in data.columns:
        metrics_df = data.groupby('cluster').agg({
            'transaction_count': ['mean', 'median'],
            'value_sent': ['mean', 'median'],
            'days_active': ['mean', 'median'],
            'unique_address': ['mean', 'median']
        }).round(2)
        
        st.dataframe(metrics_df, use_container_width=True)
    
    # Churn Risk Overview
    st.markdown("---")
    st.subheader(" Churn Risk Overview")
    
    # Create churn risk only if we have the column
    if 'days_since_last_transaction' in data.columns:
        data['churn_risk'] = pd.cut(
            data['days_since_last_transaction'],
            bins=[-1, 3, 7, float('inf')],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
    else:
        # Set unknown churn risk when column is unavailable
        data['churn_risk'] = pd.Categorical(['Unknown'] * len(data))
    
    col1, col2 = st.columns(2)
    
    with col1:
        churn_counts = data['churn_risk'].value_counts()
        fig = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title="Churn Risk Distribution",
            labels={'x': 'Risk Level', 'y': 'Number of Wallets'},
            color=churn_counts.index,
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'orange',
                'High Risk': 'red',
                'Unknown': 'gray'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        high_risk = (data['churn_risk'] == 'High Risk').sum()
        high_risk_pct = (high_risk / len(data)) * 100
        
        st.metric(
            "High Risk Wallets",
            f"{high_risk:,}",
            f"{high_risk_pct:.1f}% of total",
            delta_color="inverse"
        )
        
        if 'cluster' in data.columns and high_risk > 0:
            high_risk_clusters = data[data['churn_risk'] == 'High Risk']['cluster'].value_counts()
            st.write("**High Risk by Cluster:**")
            for cluster, count in high_risk_clusters.items():
                label = format_cluster_label('K-Means', int(cluster))
                st.write(f"- {label}: {count} wallets")


# PAGE 2: CLUSTER EXPLORER

elif page == "Cluster Explorer":
    st.markdown('<p class="main-header">Deep Dive: Cluster Analysis</p>', unsafe_allow_html=True)
    
    # Model selector
    model_choice = st.selectbox(
        "Select Clustering Model",
        ["K-Means", "Agglomerative", "DBSCAN", "Mini-Batch K-Means"]
    )
    
    cluster_col_map = {
        "K-Means": "cluster",
        "Agglomerative": "agg_cluster_labels",
        "DBSCAN": "dbscan_cluster",
        "Mini-Batch K-Means": "mb_kmeans_cluster"
    }
    
    cluster_col = cluster_col_map.get(model_choice, "cluster")
    
    if cluster_col not in data.columns:
        st.warning(f"Column '{cluster_col}' not found in data. Using 'cluster' instead.")
        cluster_col = "cluster"
    
    # Cluster selector
    clusters = sorted(data[cluster_col].unique())
    selected_cluster = st.selectbox(
    "Select Cluster to Explore",
    options=clusters,
    format_func=lambda c: format_cluster_label(model_choice, int(c)) if str(c).strip('-').isdigit() else c
)
    
    cluster_data = data[data[cluster_col] == selected_cluster]
    
    st.markdown("---")
    
    # Cluster Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Wallets in Cluster", f"{len(cluster_data):,}")
        st.caption(f"{len(cluster_data)/len(data)*100:.1f}% of total")
    
    with col2:
        avg_value = cluster_data['value_sent'].mean()
        st.metric("Avg Value Sent", f"{avg_value:,.2f} RON")
    
    with col3:
        avg_txns = cluster_data['transaction_count'].mean()
        st.metric("Avg Transactions", f"{avg_txns:.0f}")
    
    with col4:
        avg_days = cluster_data['days_active'].mean()
        st.metric("Avg Days Active", f"{avg_days:.1f}")
    
    # Detailed Statistics
    st.markdown("---")
    st.subheader(" Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Transaction Behavior:**")
        st.write(f"- Median transactions: {cluster_data['transaction_count'].median():.0f}")
        st.write(f"- Avg value per transaction: {cluster_data['average_value_sent'].mean():.2f} RON")
        st.write(f"- Unique addresses: {cluster_data['unique_address'].mean():.0f}")
    
    with col2:
        st.write("**Engagement Metrics:**")
        st.write(f"- Median days active: {cluster_data['days_active'].median():.0f}")
        if 'days_since_last_transaction' in cluster_data.columns:
            st.write(f"- Days since last transaction: {cluster_data['days_since_last_transaction'].mean():.1f}")
        else:
            st.write("- Days since last transaction: N/A")
        st.write(f"- Wallet age: {cluster_data['wallet_age_days'].mean():.0f} days")
    
    # Distribution Charts
    st.markdown("---")
    st.subheader(" Feature Distributions")
    
    feature_to_plot = st.selectbox(
        "Select Feature to Visualize",
        ['transaction_count', 'value_sent', 'days_active', 'unique_address']
    )
    
    fig = px.histogram(
        cluster_data,
        x=feature_to_plot,
        nbins=50,
        title=f"Distribution of {feature_to_plot.replace('_', ' ').title()}",
        labels={feature_to_plot: feature_to_plot.replace('_', ' ').title()}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare with other clusters
    st.markdown("---")
    st.subheader("Compare with Other Clusters")
    
    comparison_feature = st.selectbox(
        "Select Feature for Comparison",
        ['transaction_count', 'value_sent', 'days_active', 'unique_address'],
        key='comparison'
    )
    
    fig = px.box(
        data,
        x=cluster_col,
        y=comparison_feature,
        title=f"{comparison_feature.replace('_', ' ').title()} by Cluster",
        labels={cluster_col: 'Cluster', comparison_feature: comparison_feature.replace('_', ' ').title()}
    )
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: USER PREDICTOR

elif page == "User Predictor":
    st.markdown('<p class="main-header"> Predict User Segment</p>', unsafe_allow_html=True)
    
    st.write("Enter wallet features to predict which segment they belong to:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_count = st.number_input("Transaction Count", min_value=0, value=150)
        days_active = st.number_input("Days Active", min_value=0, value=20)
        days_since_last = st.number_input("Days Since Last Transaction", min_value=0, value=2)
        value_sent = st.number_input("Total Value Sent (RON)", min_value=0.0, value=100.0)
    
    with col2:
        avg_value = st.number_input("Average Value per Transaction (RON)", min_value=0.0, value=1.0)
        unique_addresses = st.number_input("Unique Addresses", min_value=0, value=10)
        avg_gas = st.number_input("Average Gas Used", min_value=0, value=100000)
        wallet_age = st.number_input("Wallet Age (Days)", min_value=0, value=30)
    
    if st.button("🔮 Predict Segment", type="primary"):
        if scaler is None or kmeans_model is None:
            st.error("Models not loaded properly. Please check your model files.")
        else:
            # Validate feature compatibility before attempting predict
            k_n = getattr(kmeans_model, 'n_features_in_', None)
            sc_n = getattr(scaler, 'n_features_in_', None) if scaler is not None else None
            feat_n = len(feature_names) if feature_names is not None else None
            if k_n is not None and ((sc_n is not None and k_n != sc_n) or (feat_n is not None and k_n != feat_n)):
                st.error(f"K-Means feature mismatch: model expects {k_n} features but scaler/features indicate {sc_n or feat_n}. Please reload models or re-run the training cells and restart the app.")
            else:
                # Prepare input
                input_data = np.array([[
                    transaction_count, days_active, days_since_last, value_sent,
                    avg_value, unique_addresses, avg_gas, wallet_age
                ]])
                
                # Scale input
                scaled_input = scaler.transform(input_data)
                
                # Predict with all models
                st.markdown("---")
                st.subheader(" Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                kmeans_pred = kmeans_model.predict(scaled_input)[0]
                k_label = format_cluster_label('K-Means', int(kmeans_pred))
                st.metric("K-Means", k_label)
            
            with col2:
                # Prefer nearest-neighbour mapping (trained on the clustering's training data) for out-of-sample Agglomerative prediction
                if 'agg_nn' in locals() and agg_nn is not None and saved_label_arrays.get('agg') is not None:
                    try:
                        idx = agg_nn.kneighbors(scaled_input, return_distance=False)[0][0]
                        agg_pred = int(saved_label_arrays['agg'][idx])
                        agg_label = format_cluster_label('Agglomerative', agg_pred)
                        st.metric("Agglomerative", agg_label)
                    except Exception as e:
                        st.error(f"Agglomerative mapping failed: {e}")
                elif agg_model is not None:
                    st.warning(
                        "Agglomerative: model available but cannot predict new samples reliably (no 'predict' method). Provide a mapping (NearestNeighbors) to enable predictions."
                    )
                elif saved_label_arrays.get('agg') is not None:
                    st.warning("Agglomerative: saved cluster labels exist but no mapping model was found to predict on new samples.")
                else:
                    st.write("Agglomerative: model not available")
            
            with col3:
                if dbscan_model is not None:
                    dbscan_pred = dbscan_model.fit_predict(scaled_input)[0]
                    if dbscan_pred == -1:
                        st.metric("DBSCAN", "Outlier ")
                    else:
                        db_label = format_cluster_label('DBSCAN', int(dbscan_pred))
                        st.metric("DBSCAN", db_label)
            
            with col4:
                if mb_kmeans_model is not None:
                    mb_pred = mb_kmeans_model.predict(scaled_input)[0]
                    mb_label = format_cluster_label('Mini-Batch K-Means', int(mb_pred))
                    st.metric("Mini-Batch", mb_label)
            
            # Detailed analysis
            st.markdown("---")
            st.subheader("Segment Analysis")
            
            # Get similar users from the predicted cluster
            similar_users = data[data['cluster'] == kmeans_pred]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Users in this segment:** {len(similar_users):,}")
                st.write(f"**Segment size:** {len(similar_users)/len(data)*100:.1f}% of total")
                
                avg_value_cluster = similar_users['value_sent'].mean()
                st.write(f"**Avg value in segment:** {avg_value_cluster:,.2f} RON")
            
            with col2:
                # Churn risk assessment
                if days_since_last > 7:
                    risk = "High"
                elif days_since_last > 3:
                    risk = "Medium"
                else:
                    risk = " Low"
                
                st.write(f"**Churn Risk:** {risk}")
                
                # Value tier
                if value_sent > 5000:
                    tier = " Whale"
                elif value_sent > 1000:
                    tier = " Power User"
                elif value_sent > 100:
                    tier = " Regular"
                else:
                    tier = " Casual"
                
                st.write(f"**User Tier:** {tier}")

# PAGE 4: BUSINESS INTELLIGENCE

elif page == "Business Intelligence":
    st.markdown('<p class="main-header"> Business Intelligence & Alerts</p>', unsafe_allow_html=True)
    
    # Whale Monitoring
    st.subheader("Whale Monitoring")
    
    # Define whales
    whale_threshold = st.slider("Whale Threshold (RON)", 1000, 10000, 5000, 500)
    whales = data[data['value_sent'] >= whale_threshold]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Whales", f"{len(whales):,}")
        st.caption(f"{len(whales)/len(data)*100:.1f}% of users")
    
    with col2:
        total_whale_value = whales['value_sent'].sum()
        st.metric("Total Whale Value", f"{total_whale_value:,.0f} RON")
        st.caption(f"{total_whale_value/data['value_sent'].sum()*100:.1f}% of total value")
    
    with col3:
        if 'days_since_last_transaction' in whales.columns:
            at_risk_whales = whales[whales['days_since_last_transaction'] > 7]
        else:
            at_risk_whales = pd.DataFrame(columns=whales.columns)
        
        st.metric("At-Risk Whales", f"{len(at_risk_whales)}")
        st.caption("Inactive >7 days")
    
    # At-risk whales list
    if len(at_risk_whales) > 0:
        st.warning(f" {len(at_risk_whales)} high-value users haven't transacted in over 7 days!")
        
        with st.expander("View At-Risk Whales"):
            display_cols = [c for c in ['transaction_count', 'value_sent', 'days_since_last_transaction', 'days_active'] if c in at_risk_whales.columns]
            st.dataframe(
                at_risk_whales[display_cols].sort_values('value_sent', ascending=False).head(20),
                use_container_width=True
            )
    
    # Churn Analysis
    st.markdown("---")
    st.subheader(" Churn Risk Analysis")
    
    if 'days_since_last_transaction' in data.columns:
        high_risk = data[data['days_since_last_transaction'] > 7]
    else:
        high_risk = pd.DataFrame(columns=data.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High Risk Users", f"{len(high_risk):,}")
        potential_loss = high_risk['value_sent'].sum()
        st.caption(f"Potential value at risk: {potential_loss:,.0f} RON")
    
    with col2:
        if 'cluster' in data.columns and len(high_risk) > 0:
            risk_by_cluster = high_risk['cluster'].value_counts()
            fig = px.pie(
                values=risk_by_cluster.values,
                names=[f"Cluster {i}" for i in risk_by_cluster.index],
                title="High Risk Users by Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("Actionable Recommendations")
    
    recommendations = []
    
    # Bot detection
    bots = data[(data['transaction_count'] > 1000) & (data['average_value_sent'] < 0.1)]
    if len(bots) > 0:
        recommendations.append({
            "priority": "High",
            "issue": "Bot Detection",
            "description": f"{len(bots)} potential bots detected (high frequency, low value)",
            "action": "Review and implement CAPTCHA or rate limiting"
        })
    
    # High churn risk
    if len(high_risk) > len(data) * 0.1:
        recommendations.append({
            "priority": "Medium",
            "issue": "High Churn Risk",
            "description": f"{len(high_risk)/len(data)*100:.1f}% of users at risk",
            "action": "Launch re-engagement campaign with targeted offers"
        })
    
    # Single-game dependency
    low_diversity = data[data['unique_address'] < 5]
    if len(low_diversity) > len(data) * 0.3:
        recommendations.append({
            "priority": "Medium",
            "issue": "Platform Dependency",
            "description": f"{len(low_diversity)/len(data)*100:.1f}% interact with <5 addresses",
            "action": "Cross-promote games and implement multi-game rewards"
        })
    
    # Whale retention
    if len(at_risk_whales) > 0:
        recommendations.append({
            "priority": "High",
            "issue": "Whale Retention",
            "description": f"{len(at_risk_whales)} high-value users inactive >7 days",
            "action": "Immediate outreach with VIP support and exclusive offers"
        })
    
    # Display recommendations
    for rec in recommendations:
        with st.expander(f"{rec['priority']} - {rec['issue']}"):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Recommended Action:** {rec['action']}")
    
    # Export functionality
    st.markdown("---")
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Whale List"):
            whale_csv = whales.to_csv(index=False)
            st.download_button(
                label="Download Whales CSV",
                data=whale_csv,
                file_name="whales.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export At-Risk Users"):
            risk_csv = high_risk.to_csv(index=False)
            st.download_button(
                label="Download At-Risk CSV",
                data=risk_csv,
                file_name="at_risk_users.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Ronin Wallet Clustering Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
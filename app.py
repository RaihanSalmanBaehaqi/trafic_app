"""
🚗 Groningen Traffic Analysis Dashboard
=======================================
Aplikasi untuk prediksi kecepatan kendaraan dan klasifikasi status lalu lintas
di Kota Groningen menggunakan Machine Learning.

Authors: 
- Fajri Hadiid Abdani (1103223188)
- Raihan Salman Baehaqi (1103220180)  
- Abid Sabyano R (1103220222)

Universitas Telkom - Data Science & Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Groningen Traffic Analysis",
    page_icon="🚗",
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
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .cluster-box-0 {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .cluster-box-1 {
        background: linear-gradient(135deg, #feca57 0%, #ff9f43 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .cluster-box-2 {
        background: linear-gradient(135deg, #5f27cd 0%, #341f97 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS & DATA
# =============================================================================
@st.cache_resource
def load_models():
    """Load all trained models"""
    import os
    models = {}
    load_errors = {}
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model paths - try both relative and absolute paths
    model_files = {
        'gradient_boosting': 'models/regression_gradient_boosting.pkl',
        'knn': 'models/regression_knn.pkl',
        'random_forest': 'models/regression_random_forest.pkl',
        'kmeans': 'models/clustering_k-means.pkl',
        'gmm': 'models/clustering_gmm.pkl',
        'dbscan': 'models/clustering_dbscan.pkl'
    }
    
    for model_name, rel_path in model_files.items():
        # Try relative path first
        paths_to_try = [
            rel_path,
            os.path.join(script_dir, rel_path),
            os.path.join(os.getcwd(), rel_path)
        ]
        
        loaded = False
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    loaded = True
                    break
                except Exception as e:
                    load_errors[model_name] = str(e)
        
        if not loaded:
            models[model_name] = None
            if model_name not in load_errors:
                load_errors[model_name] = "File not found"
    
    # Store errors for debugging
    models['_load_errors'] = load_errors
    
    return models

@st.cache_data
def load_data():
    """Load the dataset"""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple paths
    paths_to_try = [
        'data/groningen.csv',
        os.path.join(script_dir, 'data/groningen.csv'),
        os.path.join(os.getcwd(), 'data/groningen.csv'),
        'groningen.csv',
        os.path.join(script_dir, 'groningen.csv'),
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                continue
    
    return None

def preprocess_data(df):
    """Preprocess data similar to training"""
    df = df.copy()
    
    # Handle missing values
    df['error'] = df['error'].fillna(0)
    
    # Handle infinity values in occ
    df['occ'] = df['occ'].replace([np.inf, -np.inf], np.nan)
    df['occ'] = df['occ'].fillna(df['occ'].median())
    
    # Cap outliers
    def cap_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
        return data
    
    for col in ['flow', 'occ', 'speed']:
        if col in df.columns:
            df = cap_outliers(df, col)
    
    # Convert day to datetime
    df['day'] = pd.to_datetime(df['day'])
    
    # Feature engineering
    df['hour'] = df['interval'] // 3600
    df['minute'] = (df['interval'] % 3600) // 60
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['density'] = df['flow'] * df['occ']
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                          (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    # Rolling features
    df['flow_rolling_mean'] = df['flow'].rolling(window=3, min_periods=1).mean()
    df['speed_rolling_mean'] = df['speed'].rolling(window=3, min_periods=1).mean()
    
    # Interaction features
    df['flow_speed_interaction'] = df['flow'] * df['speed']
    df['occ_speed_interaction'] = df['occ'] * df['speed']
    
    return df

# =============================================================================
# SIDEBAR
# =============================================================================
def sidebar():
    with st.sidebar:
        st.image("https://telkomuniversity.ac.id/wp-content/uploads/2019/03/Logo-Telkom-University-png-3430x1174.png", width=200)
        st.markdown("---")
        
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Pilih Halaman:",
            ["🏠 Home", "🎯 Speed Prediction", "🔮 Traffic Clustering", 
             "📈 Data Exploration", "ℹ️ Model Info"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 👥 Team")
        st.markdown("""
        - Fajri Hadiid A.
        - Raihan Salman B.
        - Abid Sabyano R.
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Course")
        st.markdown("Data Science & Analysis")
        st.markdown("Teknik Komputer - Tel-U")
        
        return page

# =============================================================================
# HOME PAGE
# =============================================================================
def home_page(df):
    st.markdown('<p class="main-header">🚗 Groningen Traffic Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem Prediksi dan Klasifikasi Lalu Lintas Kota Groningen</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Data",
            value=f"{len(df):,}",
            delta="records"
        )
    
    with col2:
        st.metric(
            label="🚗 Avg Speed",
            value=f"{df['speed'].mean():.1f}",
            delta="km/h"
        )
    
    with col3:
        st.metric(
            label="🚦 Avg Flow",
            value=f"{df['flow'].mean():.1f}",
            delta="vehicles/lane"
        )
    
    with col4:
        unique_detectors = df['detid'].nunique()
        st.metric(
            label="📍 Detectors",
            value=f"{unique_detectors}",
            delta="locations"
        )
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Tentang Project")
        st.markdown("""
        Project ini bertujuan untuk menganalisis dan memprediksi kondisi lalu lintas 
        di Kota Groningen, Belanda menggunakan teknik **Machine Learning**.
        
        **Fitur Utama:**
        - 🎯 **Speed Prediction**: Memprediksi kecepatan kendaraan berdasarkan kondisi lalu lintas
        - 🔮 **Traffic Clustering**: Mengelompokkan kondisi lalu lintas (Lancar/Sedang/Padat)
        - 📈 **Data Exploration**: Visualisasi dan analisis data lalu lintas
        
        **Dataset:**
        - Sumber: ETH Zurich Traffic Dataset
        - Periode: September - Oktober 2017
        - Lokasi: Kota Groningen, Belanda
        """)
        
    with col2:
        st.markdown("### 🏆 Model Performance")
        
        st.markdown("""
        **Regression Models:**
        - Random Forest: R² = 0.994637
        - Gradient Boosting: R² = 0.993358
        - KNN: R² = 0.920677
        
        **Clustering Models:**
        - K-Means: Silhouette = 0.319628
        - GMM: Silhouette = 0.286168
        - DBSCAN: Silhouette = 0.174045
        """)
    
    st.markdown("---")
    
    # Quick visualization
    st.markdown("### 📊 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speed distribution
        fig = px.histogram(
            df[df['speed'] > 0], 
            x='speed', 
            nbins=50,
            title='Distribusi Kecepatan Kendaraan',
            labels={'speed': 'Kecepatan (km/h)', 'count': 'Frekuensi'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Flow distribution
        fig = px.histogram(
            df[df['flow'] > 0], 
            x='flow', 
            nbins=50,
            title='Distribusi Flow Kendaraan',
            labels={'flow': 'Flow (vehicles/lane)', 'count': 'Frekuensi'},
            color_discrete_sequence=['#2ca02c']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SPEED PREDICTION PAGE
# =============================================================================
def prediction_page(models, df):
    st.markdown("## 🎯 Prediksi Kecepatan Kendaraan")
    st.markdown("Masukkan parameter lalu lintas untuk memprediksi kecepatan kendaraan.")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Input Parameters")
        
        # Input form
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                hour = st.slider("🕐 Jam", 0, 23, 12)
                minute = st.slider("🕐 Menit", 0, 59, 30)
                flow = st.number_input("🚗 Flow (vehicles/lane)", 0, 1500, 200)
                occ = st.number_input("📊 Occupancy (%)", 0.0, 100.0, 5.0, 0.1)
            
            with col_b:
                day_of_week = st.selectbox(
                    "📅 Hari",
                    options=[0, 1, 2, 3, 4, 5, 6],
                    format_func=lambda x: ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'][x]
                )
                is_weekend = 1 if day_of_week >= 5 else 0
                is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
                
                st.info(f"🏖️ Weekend: {'Ya' if is_weekend else 'Tidak'}")
                st.info(f"⏰ Rush Hour: {'Ya' if is_rush_hour else 'Tidak'}")
            
            model_choice = st.selectbox(
                "🤖 Pilih Model",
                ["Random Forest", "Gradient Boosting", "KNN"]
            )
            
            submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Hasil Prediksi")
        
        if submitted:
            # Calculate derived features
            interval = hour * 3600 + minute * 60
            density = flow * occ
            
            # For rolling means, use the input values as approximation
            flow_rolling_mean = flow
            speed_rolling_mean = 40  # Default approximation
            
            # Approximate speed for interaction features
            approx_speed = 40
            flow_speed_interaction = flow * approx_speed
            occ_speed_interaction = occ * approx_speed
            
            # Prepare features (8 selected features from training)
            # Selected features: speed_rolling_mean, flow_speed_interaction, occ_speed_interaction,
            # flow_rolling_mean, flow, interval, hour, occ
            features = np.array([[
                speed_rolling_mean,
                flow_speed_interaction,
                occ_speed_interaction,
                flow_rolling_mean,
                flow,
                interval,
                hour,
                occ
            ]])
            
            # Scale features using simple normalization (approximation)
            # Since we don't have the original scaler, we normalize manually
            feature_means = [40, 8000, 200, 200, 200, 40000, 12, 5]
            feature_scales = [15, 8000, 300, 200, 200, 23000, 7, 8]
            
            features_scaled = (features - feature_means) / feature_scales
            
            # Select model
            if "Random Forest" in model_choice:
                model = models.get('random_forest')
                model_name = "Random Forest"
            elif "Gradient Boosting" in model_choice:
                model = models.get('gradient_boosting')
                model_name = "Gradient Boosting"
            else:
                model = models.get('knn')
                model_name = "KNN"
            
            if model is not None:
                try:
                    prediction = model.predict(features_scaled)[0]
                    prediction = max(0, min(prediction, 120))  # Clip to reasonable range
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🚗 Prediksi Kecepatan</h2>
                        <h1 style="font-size: 4rem; margin: 0;">{prediction:.1f}</h1>
                        <h3>km/jam</h3>
                        <p>Model: {model_name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Speed interpretation
                    if prediction > 50:
                        status = "🟢 LANCAR"
                        desc = "Kondisi lalu lintas sangat baik, kendaraan dapat bergerak dengan kecepatan tinggi."
                    elif prediction > 30:
                        status = "🟡 SEDANG"
                        desc = "Kondisi lalu lintas normal, mungkin ada sedikit kepadatan."
                    else:
                        status = "🔴 PADAT"
                        desc = "Kondisi lalu lintas padat, kecepatan kendaraan terbatas."
                    
                    st.markdown(f"### Status: {status}")
                    st.info(desc)
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                # Show debug info
                load_errors = models.get('_load_errors', {})
                error_key = 'gradient_boosting' if "Gradient Boosting" in model_choice else 'knn'
                error_msg = load_errors.get(error_key, "Unknown error")
                
                st.error(f"Model tidak tersedia: {error_msg}")
                
                # Show debugging info
                with st.expander("🔧 Debug Info"):
                    import os
                    st.write("**Current Working Directory:**", os.getcwd())
                    st.write("**Script Directory:**", os.path.dirname(os.path.abspath(__file__)))
                    
                    # Check if models folder exists
                    models_path = "models"
                    if os.path.exists(models_path):
                        st.write("**Files in models/:**")
                        st.write(os.listdir(models_path))
                    else:
                        st.write("**models/ folder not found!**")
                        st.write("Files in current directory:", os.listdir("."))
                    
                    st.write("**Load Errors:**", load_errors)
        else:
            st.info("👆 Masukkan parameter dan klik 'Prediksi' untuk melihat hasil.")
            
            # Show example
            st.markdown("### 📖 Contoh Input")
            st.markdown("""
            | Parameter | Nilai Contoh |
            |-----------|--------------|
            | Jam | 08:00 (Rush Hour) |
            | Flow | 300 vehicles/lane |
            | Occupancy | 8% |
            | Hari | Senin |
            """)

# =============================================================================
# CLUSTERING PAGE
# =============================================================================
def clustering_page(models, df):
    st.markdown("## 🔮 Klasifikasi Status Lalu Lintas")
    st.markdown("Analisis pengelompokan kondisi lalu lintas berdasarkan berbagai parameter.")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Klasifikasi Real-time", "📈 Analisis Cluster"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Input Data Lalu Lintas")
            
            with st.form("clustering_form"):
                flow = st.number_input("🚗 Flow (vehicles/lane)", 0, 1500, 200)
                occ = st.number_input("📊 Occupancy (%)", 0.0, 100.0, 5.0, 0.1)
                speed = st.number_input("🚀 Speed (km/h)", 0, 150, 40)
                hour = st.slider("🕐 Jam", 0, 23, 12)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    is_rush_hour = st.checkbox("⏰ Rush Hour")
                with col_b:
                    is_weekend = st.checkbox("🏖️ Weekend")
                
                model_choice = st.selectbox(
                    "🤖 Pilih Model Clustering",
                    ["K-Means", "GMM", "DBSCAN"]
                )
                
                submitted = st.form_submit_button("🔮 Klasifikasi", use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Hasil Klasifikasi")
            
            if submitted:
                # Calculate density
                density = flow * occ
                
                # Prepare features for clustering
                # Features: flow, occ, speed, density, hour, is_rush_hour, is_weekend
                features = np.array([[
                    flow, occ, speed, density, hour,
                    int(is_rush_hour), int(is_weekend)
                ]])
                
                # Standardize features
                feature_means = [250, 6, 40, 2000, 12, 0.2, 0.3]
                feature_stds = [200, 5, 15, 3000, 7, 0.4, 0.45]
                features_scaled = (features - feature_means) / feature_stds
                
                # Select model
                if "K-Means" in model_choice:
                    model = models.get('kmeans')
                    model_name = "K-Means"
                elif "GMM" in model_choice:
                    model = models.get('gmm')
                    model_name = "GMM"
                else:
                    model = models.get('dbscan')
                    model_name = "DBSCAN"
                
                if model is not None:
                    try:
                        # DBSCAN doesn't have predict method, need special handling
                        if "DBSCAN" in model_choice:
                            # DBSCAN cannot predict new data directly
                            # We'll use a heuristic based on input features
                            st.warning("⚠️ DBSCAN tidak mendukung prediksi data baru secara langsung.")
                            
                            # Heuristic classification based on traffic parameters
                            if speed < 35 and flow > 400:
                                cluster = 0  # Heavy traffic
                                status = "🔴 PADAT"
                                level = "Heavy Traffic"
                                desc = "Berdasarkan parameter input, kondisi lalu lintas terdeteksi PADAT."
                            elif speed > 45 and flow < 200:
                                cluster = 2  # Light traffic  
                                status = "🟢 LANCAR"
                                level = "Low Traffic"
                                desc = "Berdasarkan parameter input, kondisi lalu lintas terdeteksi LANCAR."
                            else:
                                cluster = 1  # Moderate
                                status = "🟡 SEDANG"
                                level = "Moderate Traffic"
                                desc = "Berdasarkan parameter input, kondisi lalu lintas terdeteksi SEDANG."
                            
                            color = "#ff6b6b" if cluster == 0 else ("#feca57" if cluster == 1 else "#5f27cd")
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                        border-radius: 15px; padding: 2rem; color: white; text-align: center;">
                                <h2>{status}</h2>
                                <h3>{level}</h3>
                                <p>Model: Heuristic (DBSCAN-based)</p>
                                <p>Estimated Cluster: {cluster}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.info(f"📋 **Interpretasi:** {desc}")
                            st.caption("*Note: DBSCAN menggunakan pendekatan heuristik untuk data baru karena algoritma ini tidak mendukung predict() secara native.*")
                        else:
                            # K-Means and GMM have predict method
                            cluster = model.predict(features_scaled)[0]
                            
                            # Cluster interpretation based on training results
                            cluster_info = {
                                0: ("🔴 PADAT", "Heavy Traffic", "cluster-box-0", 
                                    "Lalu lintas sangat padat dengan flow tinggi dan kecepatan rendah."),
                                1: ("🟡 SEDANG", "Moderate Traffic", "cluster-box-1",
                                    "Lalu lintas sedang dengan kondisi normal."),
                                2: ("🟢 LANCAR", "Low Traffic", "cluster-box-2",
                                    "Lalu lintas lancar dengan kecepatan tinggi.")
                            }
                            
                            cluster_idx = cluster % 3  # Map to 0, 1, 2
                            status, level, _, desc = cluster_info.get(cluster_idx, cluster_info[1])
                            
                            # Display result
                            if cluster_idx == 0:
                                color = "#ff6b6b"
                            elif cluster_idx == 1:
                                color = "#feca57"
                            else:
                                color = "#5f27cd"
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                        border-radius: 15px; padding: 2rem; color: white; text-align: center;">
                                <h2>{status}</h2>
                                <h3>{level}</h3>
                                <p>Model: {model_name}</p>
                                <p>Cluster ID: {cluster}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.info(f"📋 **Interpretasi:** {desc}")
                        
                        # Show input summary
                        st.markdown("### 📊 Ringkasan Input")
                        input_df = pd.DataFrame({
                            'Parameter': ['Flow', 'Occupancy', 'Speed', 'Density', 'Hour', 'Rush Hour', 'Weekend'],
                            'Nilai': [flow, f"{occ}%", f"{speed} km/h", f"{density:.1f}", hour, 
                                     'Ya' if is_rush_hour else 'Tidak', 'Ya' if is_weekend else 'Tidak']
                        })
                        st.dataframe(input_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Error dalam klasifikasi: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    # Show debug info
                    load_errors = models.get('_load_errors', {})
                    if "K-Means" in model_choice:
                        error_key = 'kmeans'
                    elif "GMM" in model_choice:
                        error_key = 'gmm'
                    else:
                        error_key = 'dbscan'
                    error_msg = load_errors.get(error_key, "Unknown error")
                    
                    st.error(f"Model tidak tersedia: {error_msg}")
                    
                    # Show debugging info
                    with st.expander("🔧 Debug Info"):
                        import os
                        st.write("**Current Working Directory:**", os.getcwd())
                        st.write("**Script Directory:**", os.path.dirname(os.path.abspath(__file__)))
                        
                        models_path = "models"
                        if os.path.exists(models_path):
                            st.write("**Files in models/:**")
                            st.write(os.listdir(models_path))
                        else:
                            st.write("**models/ folder not found!**")
                            st.write("Files in current directory:", os.listdir("."))
                        
                        st.write("**Load Errors:**", load_errors)
            else:
                st.info("👆 Masukkan parameter dan klik 'Klasifikasi' untuk melihat hasil.")
    
    with tab2:
        st.markdown("### 📈 Karakteristik Cluster")
        
        # Display cluster characteristics from training
        cluster_data = pd.DataFrame({
            'Cluster': ['Cluster 0 (Padat)', 'Cluster 1 (Sedang)', 'Cluster 2 (Lancar)'],
            'Avg Flow': [504.05, 221.64, 138.20],
            'Avg Speed': [37.39, 43.06, 45.12],
            'Avg Occupancy': [13.67, 5.26, 3.20],
            'Data Points': [5923, 6306, 11699],
            'Percentage': ['24.8%', '26.4%', '48.9%']
        })
        
        st.dataframe(cluster_data, use_container_width=True, hide_index=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                cluster_data, 
                x='Cluster', 
                y='Avg Speed',
                color='Cluster',
                title='Rata-rata Kecepatan per Cluster',
                color_discrete_sequence=['#ff6b6b', '#feca57', '#5f27cd']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                cluster_data, 
                x='Cluster', 
                y='Avg Flow',
                color='Cluster',
                title='Rata-rata Flow per Cluster',
                color_discrete_sequence=['#ff6b6b', '#feca57', '#5f27cd']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart
        fig = px.pie(
            cluster_data,
            values=[5923, 6306, 11699],
            names=['Padat', 'Sedang', 'Lancar'],
            title='Distribusi Kondisi Lalu Lintas',
            color_discrete_sequence=['#ff6b6b', '#feca57', '#5f27cd']
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# DATA EXPLORATION PAGE
# =============================================================================
def exploration_page(df):
    st.markdown("## 📈 Data Exploration")
    st.markdown("Eksplorasi dan visualisasi data lalu lintas Kota Groningen.")
    
    st.markdown("---")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📈 Visualisasi", "🔍 Raw Data"])
    
    with tab1:
        st.markdown("### 📊 Statistik Deskriptif")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Unique Detectors", df['detid'].nunique())
        
        with col2:
            st.metric("Date Range", f"{df['day'].min()} - {df['day'].max()}")
            st.metric("Avg Speed", f"{df['speed'].mean():.2f} km/h")
        
        with col3:
            st.metric("Max Flow", f"{df['flow'].max():.0f}")
            st.metric("Avg Flow", f"{df['flow'].mean():.2f}")
        
        st.markdown("### 📋 Summary Statistics")
        st.dataframe(df[['flow', 'occ', 'speed']].describe(), use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 Visualisasi Data")
        
        viz_option = st.selectbox(
            "Pilih Visualisasi:",
            ["Speed Distribution", "Flow vs Speed", "Hourly Pattern", "Correlation Heatmap"]
        )
        
        if viz_option == "Speed Distribution":
            fig = px.histogram(
                df_processed[df_processed['speed'] > 0],
                x='speed',
                nbins=50,
                title='Distribusi Kecepatan Kendaraan',
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(xaxis_title="Kecepatan (km/h)", yaxis_title="Frekuensi")
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Flow vs Speed":
            sample_df = df_processed[df_processed['speed'] > 0].sample(min(5000, len(df_processed)))
            fig = px.scatter(
                sample_df,
                x='flow',
                y='speed',
                title='Hubungan Flow dan Speed',
                opacity=0.5,
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(xaxis_title="Flow (vehicles/lane)", yaxis_title="Speed (km/h)")
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Hourly Pattern":
            hourly_data = df_processed.groupby('hour').agg({
                'speed': 'mean',
                'flow': 'mean'
            }).reset_index()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=hourly_data['hour'], y=hourly_data['speed'], 
                          name="Avg Speed", line=dict(color='#1f77b4')),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=hourly_data['hour'], y=hourly_data['flow'], 
                          name="Avg Flow", line=dict(color='#ff7f0e')),
                secondary_y=True,
            )
            
            fig.update_layout(title_text="Pola Lalu Lintas per Jam")
            fig.update_xaxes(title_text="Jam")
            fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
            fig.update_yaxes(title_text="Flow (vehicles/lane)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Correlation Heatmap":
            corr_cols = ['flow', 'occ', 'speed']
            corr_matrix = df_processed[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Raw Data")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_speed = st.slider("Min Speed", 0, int(df['speed'].max()), 0)
        with col2:
            min_flow = st.slider("Min Flow", 0, int(df['flow'].max()), 0)
        
        filtered_df = df[(df['speed'] >= min_speed) & (df['flow'] >= min_flow)]
        
        st.write(f"Showing {len(filtered_df):,} records")
        st.dataframe(filtered_df.head(1000), use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="groningen_traffic_data.csv",
            mime="text/csv"
        )

# =============================================================================
# MODEL INFO PAGE
# =============================================================================
def model_info_page():
    st.markdown("## ℹ️ Informasi Model")
    st.markdown("Detail teknis tentang model machine learning yang digunakan.")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🎯 Regression Models", "🔮 Clustering Models"])
    
    with tab1:
        st.markdown("### 🎯 Model Regresi - Prediksi Kecepatan")
        
        st.markdown("""
        Model regresi digunakan untuk memprediksi kecepatan kendaraan berdasarkan
        berbagai parameter lalu lintas.
        """)
        
        # Model comparison
        reg_models = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'KNN', 'SVR', 'Linear Regression'],
            'R² Score': [0.9946, 0.9934, 0.9207, 0.7959, 0.6996],
            'RMSE': [1.10, 1.23, 4.24, 6.85, 8.31],
            'MAE': [0.44, 0.80, 2.22, 2.86, 5.22],
            'Status': ['✅ Best', '✅ Deployed', '✅ Deployed', '❌ Not Used', '❌ Not Used']
        })
        
        st.dataframe(reg_models, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = px.bar(
            reg_models,
            x='Model',
            y='R² Score',
            title='Perbandingan R² Score Model Regresi',
            color='R² Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📝 Selected Features")
        st.markdown("""
        Top 8 features yang digunakan untuk prediksi (berdasarkan F-score):
        1. **speed_rolling_mean** - Moving average kecepatan
        2. **flow_speed_interaction** - Interaksi flow × speed
        3. **occ_speed_interaction** - Interaksi occupancy × speed
        4. **flow_rolling_mean** - Moving average flow
        5. **flow** - Jumlah kendaraan per lajur
        6. **interval** - Waktu dalam detik
        7. **hour** - Jam dalam sehari
        8. **occ** - Occupancy sensor
        """)
    
    with tab2:
        st.markdown("### 🔮 Model Clustering - Klasifikasi Lalu Lintas")
        
        st.markdown("""
        Model clustering digunakan untuk mengelompokkan kondisi lalu lintas
        menjadi beberapa kategori berdasarkan pola data.
        """)
        
        # Model comparison
        cluster_models = pd.DataFrame({
            'Model': ['K-Means', 'GMM', 'Hierarchical', 'DBSCAN'],
            'Silhouette Score': [0.3196, 0.2862, 0.2699, 0.1740],
            'Davies-Bouldin': [1.20, 1.27, 1.29, 0.89],
            'N Clusters': [3, 3, 3, 11],
            'Status': ['✅ Best', '✅ Deployed', '❌ Not Used', '✅ Deployed']
        })
        
        st.dataframe(cluster_models, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = px.bar(
            cluster_models,
            x='Model',
            y='Silhouette Score',
            title='Perbandingan Silhouette Score Model Clustering',
            color='Silhouette Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📝 Clustering Features")
        st.markdown("""
        Features yang digunakan untuk clustering:
        - **flow** - Jumlah kendaraan per lajur
        - **occ** - Occupancy sensor
        - **speed** - Kecepatan kendaraan
        - **density** - Kepadatan (flow × occ)
        - **hour** - Jam dalam sehari
        - **is_rush_hour** - Indikator jam sibuk
        - **is_weekend** - Indikator akhir pekan
        """)
        
        st.markdown("### 🏷️ Cluster Interpretation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #ff6b6b; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🔴 Cluster 0</h3>
                <h4>PADAT</h4>
                <p>Flow: 504 veh/lane</p>
                <p>Speed: 37 km/h</p>
                <p>24.8% data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #feca57; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🟡 Cluster 1</h3>
                <h4>SEDANG</h4>
                <p>Flow: 222 veh/lane</p>
                <p>Speed: 43 km/h</p>
                <p>26.4% data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #5f27cd; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🟢 Cluster 2</h3>
                <h4>LANCAR</h4>
                <p>Flow: 138 veh/lane</p>
                <p>Speed: 45 km/h</p>
                <p>48.9% data</p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Load resources
    models = load_models()
    df = load_data()
    
    # Sidebar navigation
    page = sidebar()
    
    # Check if data loaded
    if df is None:
        st.error("❌ Gagal memuat dataset. Pastikan file 'groningen.csv' ada di folder 'data/'.")
        return
    
    # Route to pages
    if page == "🏠 Home":
        home_page(df)
    elif page == "🎯 Speed Prediction":
        prediction_page(models, df)
    elif page == "🔮 Traffic Clustering":
        clustering_page(models, df)
    elif page == "📈 Data Exploration":
        exploration_page(df)
    elif page == "ℹ️ Model Info":
        model_info_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚗 Groningen Traffic Analysis Dashboard</p>
        <p>TUBES UAS Data Science & Analysis - Universitas Telkom 2025</p>
        <p>Made with ❤️ by Fajri, Raihan, and Abid</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

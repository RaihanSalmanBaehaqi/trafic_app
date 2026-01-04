# 🚗 Groningen Traffic Analysis Dashboard

**Link:** : https://trafic-groningen.streamlit.app/

Aplikasi web untuk **prediksi kecepatan kendaraan** dan **klasifikasi status lalu lintas** di Kota Groningen menggunakan Machine Learning.

## 📋 Deskripsi Project

Project ini merupakan TUBES UAS mata kuliah **Data Science dan Analisis** yang bertujuan untuk:
1. Memprediksi kecepatan kendaraan berdasarkan kondisi lalu lintas (Regression)
2. Mengklasifikasikan status lalu lintas (Lancar/Sedang/Padat) menggunakan Clustering

## 👥 Tim Pengembang

| Nama | NIM |
|------|-----|
| Fajri Hadiid Abdani | 1103223188 |
| Raihan Salman Baehaqi | 1103220180 |
| Abid Sabyano R | 1103220222 |

**Program Studi:** S1 Teknik Komputer  
**Fakultas:** Teknik Elektro  
**Universitas:** Telkom University

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **ML Libraries:** Scikit-learn
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy

## 📊 Model yang Digunakan

### Regression Models (Prediksi Kecepatan)
| Model | R² Score | RMSE | Status |
|-------|----------|------|--------|
| Random Forest | 0.9946 | 1.10 | ✅ Best & Deployed |
| Gradient Boosting | 0.9934 | 1.23 | ✅ Deployed |
| KNN | 0.9207 | 4.24 | ✅ Deployed |

### Clustering Models (Klasifikasi Lalu Lintas)
| Model | Silhouette Score | N Clusters | Status |
|-------|------------------|------------|--------|
| K-Means | 0.3196 | 3 | ✅ Best & Deployed |
| GMM | 0.2862 | 3 | ✅ Deployed |
| DBSCAN | 0.1740 | 11 | ✅ Deployed |

## 📁 Struktur Folder

```
streamlit_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── data/
│   └── groningen.csv     # Dataset
└── models/
    ├── regression_random_forest.pkl
    ├── regression_gradient_boosting.pkl
    ├── regression_knn.pkl
    ├── clustering_k-means.pkl
    ├── clustering_gmm.pkl
    └── clustering_dbscan.pkl
```

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone <repository-url>
cd streamlit_app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 4. Buka Browser
Aplikasi akan berjalan di `http://localhost:8501`

## 📱 Fitur Aplikasi

### 🏠 Home
- Overview project dan statistik data
- Quick visualization data

### 🎯 Speed Prediction
- Input parameter lalu lintas
- Prediksi kecepatan dengan Gradient Boosting atau KNN
- Interpretasi hasil (Lancar/Sedang/Padat)

### 🔮 Traffic Clustering
- Klasifikasi kondisi lalu lintas real-time
- Pilihan model: K-Means, GMM, DBSCAN
- Analisis karakteristik cluster

### 📈 Data Exploration
- Statistik deskriptif
- Visualisasi interaktif (Histogram, Scatter, Heatmap)
- Download data CSV

### ℹ️ Model Info
- Detail teknis model regression
- Detail teknis model clustering
- Feature importance

## 📊 Dataset

- **Sumber:** ETH Zurich Traffic Dataset
- **Periode:** September - Oktober 2017
- **Lokasi:** Kota Groningen, Belanda
- **Total Records:** 28,378 rows
- **Features:** 8 columns (day, interval, detid, flow, occ, error, city, speed)

## 🔧 Deployment

### Deploy ke Streamlit Cloud
1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Deploy ke Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

## 📝 License

MIT License - Universitas Telkom 2025

## 🙏 Acknowledgments

- ETH Zurich untuk dataset lalu lintas
- Dosen pembimbing mata kuliah Data Science
- Streamlit team untuk framework yang luar biasa

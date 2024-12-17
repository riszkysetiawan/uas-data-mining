import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Membaca data
@st.cache_data
def load_data():
    # Gantilah dengan path data Anda jika menggunakan file lokal
    data = pd.read_excel('assets/sales_clean.xlsx')
    return data.copy()  # Pastikan data tidak dimutasi setelah dimuat

# Menampilkan dataset dan statistik
data = load_data()

# Header Dashboard
st.title("Dashboard Clustering Data Penjualan")
st.sidebar.title("Pengaturan")

# Menampilkan dataset
st.subheader("Dataset:")
st.write(data.head())

# Dropdown untuk memilih clustering
clustering_option = st.sidebar.selectbox(
    'Pilih Grafik Clustering:',
    ['Kuantitas, Profit, Cost Produksi', 'Total, Harga per Unit', 'Profit, Bulan']
)

# Slider untuk memilih jumlah klaster
num_clusters = st.sidebar.slider('Pilih jumlah klaster (K)', min_value=2, max_value=10, value=2, step=1)

# Fungsi untuk melakukan clustering dan visualisasi
def perform_clustering(columns, n_clusters):
    X = data[columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Melakukan fitting model KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualisasi Scatter Plot Interaktif dengan Plotly
    if len(columns) == 2:
        # Visualisasi 2D
        fig = px.scatter(
            data, 
            x=columns[0], 
            y=columns[1], 
            color='Cluster', 
            title=f'Clustering: {columns[0]} vs {columns[1]}',
            labels={columns[0]: columns[0], columns[1]: columns[1]},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)

    elif len(columns) == 3:
        # Visualisasi 3D
        fig = px.scatter_3d(
            data, 
            x=columns[0], 
            y=columns[1], 
            z=columns[2], 
            color='Cluster', 
            title=f'Clustering 3D: {columns[0]}, {columns[1]}, {columns[2]}',
            labels={columns[0]: columns[0], columns[1]: columns[1], columns[2]: columns[2]},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)

    # Menampilkan Silhouette Score untuk evaluasi
    silhouette_avg = silhouette_score(X_scaled, data['Cluster'])
    st.subheader(f"Silhouette Score untuk {columns[0]} dan {columns[1]}: {silhouette_avg:.2f}")

# Memilih kolom berdasarkan pilihan clustering
if clustering_option == 'Kuantitas, Profit, Cost Produksi':
    perform_clustering(['Kuantitas', 'Profit', 'Cost Produksi'], num_clusters)
elif clustering_option == 'Total, Harga per Unit':
    perform_clustering(['Harga per Unit', 'Total'], num_clusters)
elif clustering_option == 'Profit, Bulan':
    perform_clustering(['Profit', 'Bulan'],num_clusters)
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Fungsi untuk membaca dataset
def load_data(file_path):
    return pd.read_excel(file_path, engine='openpyxl')

# Fungsi untuk melakukan clustering
def perform_clustering(data, n_clusters):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    return data, kmeans

# Load dataset
file_path = "assets/sales.xlsx"  # Pastikan file ini ada di direktori
try:
    data = load_data(file_path)
except Exception as e:
    st.error(f"Error membaca file: {e}")
    st.stop()

# Sidebar untuk konfigurasi clustering
st.sidebar.header("Konfigurasi Clustering")
n_clusters = st.sidebar.slider('Jumlah cluster', min_value=2, max_value=10, value=3)

# Kolom yang digunakan
columns_to_use = ['Kuantitas', 'Total']
if not all(col in data.columns for col in columns_to_use):
    st.error("Kolom 'Kuantitas' dan 'Total' tidak ditemukan dalam dataset.")
    st.stop()

# Clustering
if len(data) < n_clusters:
    st.error("Jumlah data lebih kecil dari jumlah cluster. Kurangi jumlah cluster atau perbanyak data.")
else:
    clustered_data, kmeans_model = perform_clustering(data[columns_to_use], n_clusters)

    # Tampilkan data hasil clustering
    st.header("Data Setelah Clustering")
    st.dataframe(clustered_data)

    # Visualisasi scatter plot dengan warna berbeda untuk tiap cluster
    st.subheader("Visualisasi Scatter Plot dengan Cluster")
    fig_all = px.scatter(
        clustered_data,
        x='Kuantitas',
        y='Total',
        color='Cluster',  # Parameter color untuk membedakan warna setiap cluster
        color_discrete_sequence=px.colors.qualitative.Set1,  # Warna berbeda
        title='Visualisasi Semua Cluster',
        labels={'Cluster': 'Cluster'},
        hover_data=clustered_data.columns
    )
    st.plotly_chart(fig_all)

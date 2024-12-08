import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Fungsi untuk membaca dataset
def load_data(file_path):
    return pd.read_excel(file_path)

# Fungsi untuk melakukan clustering
def perform_clustering(data, n_clusters):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    return data, kmeans

# Load dataset
file_path = "assets/sales.xlsx"
data = load_data(file_path)

# Sidebar untuk pengaturan clustering
st.sidebar.header("Clustering Configuration")
columns_to_use = ['Kuantitas', 'Jumlah Per Baris']
n_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=10, value=3)

# Sidebar untuk filter tambahan
st.sidebar.header("Filters")
min_kuantitas = st.sidebar.number_input("Minimum Kuantitas", value=int(data['Kuantitas'].min()))
max_kuantitas = st.sidebar.number_input("Maximum Kuantitas", value=int(data['Kuantitas'].max()))
min_jumlah = st.sidebar.number_input("Minimum Jumlah Per Baris", value=int(data['Jumlah Per Baris'].min()))
max_jumlah = st.sidebar.number_input("Maximum Jumlah Per Baris", value=int(data['Jumlah Per Baris'].max()))

st.title("Dashboard Clustering")

# Filter data berdasarkan input
filtered_data = data[(data['Kuantitas'] >= min_kuantitas) &
                     (data['Kuantitas'] <= max_kuantitas) &
                     (data['Jumlah Per Baris'] >= min_jumlah) &
                     (data['Jumlah Per Baris'] <= max_jumlah)]

# Pastikan ada data yang lolos filter
if filtered_data.empty:
    st.warning("Tidak ada data yang sesuai dengan filter. Silakan ubah nilai filter.")
else:
    # Clustering
    clustered_data, kmeans_model = perform_clustering(filtered_data[columns_to_use], n_clusters)

    # Tampilkan data yang telah dikelompokkan
    st.header("Clustered Data")
    st.dataframe(clustered_data)

    # Visualisasi clustering
    st.header("Cluster Visualization")
    fig, ax = plt.subplots()
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for cluster in range(n_clusters):
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        ax.scatter(cluster_data['Kuantitas'], cluster_data['Jumlah Per Baris'],
                   color=colors[cluster % len(colors)], label=f'Cluster {cluster}')

    # Centroid
    ax.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1],
               color='black', marker='*', label='Centroids')
    ax.set_xlabel("Kuantitas")
    ax.set_ylabel("Jumlah Per Baris")
    ax.legend()
    st.pyplot(fig)

    # Visualisasi dengan Plotly
    fig = px.scatter(clustered_data, x='Kuantitas', y='Jumlah Per Baris', color='Cluster',
                     title='Cluster Visualization',
                     labels={'Cluster': 'Cluster'})
    st.plotly_chart(fig)

    # Evaluasi jumlah cluster menggunakan Elbow Method
    st.header("Elbow Method")
    k_rng = range(1, 11)
    sse = []

    for k in k_rng:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(filtered_data[columns_to_use])
        sse.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_rng, sse, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Sum of Squared Errors (SSE)')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)
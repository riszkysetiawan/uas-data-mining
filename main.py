import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

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

# Sidebar untuk pengaturan clustering
st.sidebar.header("Clustering Configuration")
n_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=10, value=2)

# Pastikan kolom 'Kuantitas' dan 'Total' ada
columns_to_use = ['Kuantitas', 'Total']

if not all(col in data.columns for col in columns_to_use):
    st.error("Kolom 'Kuantitas' dan 'Total' tidak ditemukan dalam dataset.")
    st.stop()

# Clustering
if len(data) < n_clusters:
    st.error("Jumlah data lebih kecil dari jumlah cluster. Kurangi jumlah cluster atau perbanyak data.")
else:
    clustered_data, kmeans_model = perform_clustering(data[columns_to_use], n_clusters)

    # Tampilkan data yang telah dikelompokkan
    st.header("Clustered Data")
    st.dataframe(clustered_data)

    # **Cluster 1: Grafik dan Elbow**
    st.subheader("Cluster 1")
    fig1 = px.scatter(
        clustered_data[clustered_data['Cluster'] == 0],
        x='Kuantitas',
        y='Total',
        color='Cluster',
        title='Cluster 1 Visualization',
        labels={'Cluster': 'Cluster'},
        hover_data=clustered_data.columns
    )
    st.plotly_chart(fig1)

    # Elbow Method untuk Cluster 1
    st.subheader("Elbow Method - Cluster 1")
    k_rng = range(1, 11)
    sse = []
    try:
        for k in k_rng:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(clustered_data[clustered_data['Cluster'] == 0][columns_to_use])
            sse.append(km.inertia_)

        elbow_fig1 = go.Figure()
        elbow_fig1.add_trace(
            go.Scatter(
                x=list(k_rng),
                y=sse,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(color='blue'),
                name='SSE'
            )
        )
        elbow_fig1.update_layout(
            title='Elbow Method for Cluster 1',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Sum of Squared Errors (SSE)',
            template='plotly_white'
        )
        st.plotly_chart(elbow_fig1)
    except ValueError as e:
        st.error(f"Error saat menjalankan Elbow Method untuk Cluster 1: {e}. Coba kurangi jumlah cluster.")

    # **Cluster 2: Grafik dan Elbow**
    st.subheader("Cluster 2")
    fig2 = px.scatter(
        clustered_data[clustered_data['Cluster'] == 1],
        x='Kuantitas',
        y='Total',
        color='Cluster',
        title='Cluster 2 Visualization',
        labels={'Cluster': 'Cluster'},
        hover_data=clustered_data.columns
    )
    st.plotly_chart(fig2)

    # Elbow Method untuk Cluster 2
    st.subheader("Elbow Method - Cluster 2")
    sse = []
    try:
        for k in k_rng:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(clustered_data[clustered_data['Cluster'] == 1][columns_to_use])
            sse.append(km.inertia_)

        elbow_fig2 = go.Figure()
        elbow_fig2.add_trace(
            go.Scatter(
                x=list(k_rng),
                y=sse,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(color='green'),
                name='SSE'
            )
        )
        elbow_fig2.update_layout(
            title='Elbow Method for Cluster 2',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Sum of Squared Errors (SSE)',
            template='plotly_white'
        )
        st.plotly_chart(elbow_fig2)
    except ValueError as e:
        st.error(f"Error saat menjalankan Elbow Method untuk Cluster 2: {e}. Coba kurangi jumlah cluster.")

    # **Cluster 3: Grafik dan Elbow**
    st.subheader("Cluster 3")
    fig3 = px.scatter(
        clustered_data[clustered_data['Cluster'] == 2],
        x='Kuantitas',
        y='Total',
        color='Cluster',
        title='Cluster 3 Visualization',
        labels={'Cluster': 'Cluster'},
        hover_data=clustered_data.columns
    )
    st.plotly_chart(fig3)

    # Elbow Method untuk Cluster 3
    st.subheader("Elbow Method - Cluster 3")
    sse = []
    try:
        for k in k_rng:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(clustered_data[clustered_data['Cluster'] == 2][columns_to_use])
            sse.append(km.inertia_)

        elbow_fig3 = go.Figure()
        elbow_fig3.add_trace(
            go.Scatter(
                x=list(k_rng),
                y=sse,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(color='red'),
                name='SSE'
            )
        )
        elbow_fig3.update_layout(
            title='Elbow Method for Cluster 3',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Sum of Squared Errors (SSE)',
            template='plotly_white'
        )
        st.plotly_chart(elbow_fig3)
    except ValueError as e:
        st.error(f"Error saat menjalankan Elbow Method untuk Cluster 3: {e}. Coba kurangi jumlah cluster.")

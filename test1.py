import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target
    return X, y

# Preprocess data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Perform KMeans clustering
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# Perform DBSCAN clustering
def dbscan_clustering(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    return dbscan.labels_

# Visualize clusters
def visualize_clusters(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar()
    st.pyplot()

# Streamlit app
def main():
    st.title("MNIST Clustering with KMeans and DBSCAN")
    
    # Load data
    X, y = load_data()
    X_scaled = preprocess_data(X)
    
    # Sidebar for parameters
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of clusters (KMeans)", 2, 20, 10)
    eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 1, 20, 5)
    
    # Perform KMeans clustering
    if st.sidebar.button("Run KMeans"):
        with mlflow.start_run():
            labels = kmeans_clustering(X_scaled, n_clusters)
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_metric("silhouette_score", silhouette_score(X_scaled, labels))
            st.write("KMeans Clustering Results")
            visualize_clusters(X_scaled, labels)
    
    # Perform DBSCAN clustering
    if st.sidebar.button("Run DBSCAN"):
        with mlflow.start_run():
            labels = dbscan_clustering(X_scaled, eps, min_samples)
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)
            mlflow.log_metric("silhouette_score", silhouette_score(X_scaled, labels))
            st.write("DBSCAN Clustering Results")
            visualize_clusters(X_scaled, labels)

if __name__ == "__main__":
    main()
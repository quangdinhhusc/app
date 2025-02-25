import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
import mlflow
import mlflow.sklearn

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

X_normalized = X / 255.0

X_subset = X_normalized.sample(n=1000, random_state=42)

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_subset)
labels_kmeans = kmeans.labels_

mlflow.start_run()
mlflow.sklearn.log_model(kmeans, "kmeans_model")
mlflow.log_param("n_clusters", 10)
mlflow.end_run()

plt.figure(figsize=(10, 10))
plt.scatter(X_subset.iloc[:, 0], X_subset.iloc[:, 1], c=labels_kmeans, cmap='viridis', s=15)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_subset)

mlflow.start_run()
mlflow.sklearn.log_model(dbscan, "dbscan_model")
mlflow.log_param("eps", 0.5)
mlflow.log_param("min_samples", 5)
mlflow.end_run()

plt.figure(figsize=(10, 10))
plt.scatter(X_subset.iloc[:, 0], X_subset.iloc[:, 1], c=labels_dbscan, cmap='viridis', s=15)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

st.title("MNIST Clustering with K-means and DBSCAN")

st.subheader("K-means Clustering")
st.pyplot(plt)

st.subheader("DBSCAN Clustering")
st.pyplot(plt)

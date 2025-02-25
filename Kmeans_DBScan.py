import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import struct
import seaborn as sns
import mlflow
import mlflow.sklearn
import kagglehub
import openml
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# st.set_page_config(page_title="Phân loại ảnh", layout="wide")

# Streamlit app
st.title("MNIST Classification with Streamlit & MLFlow")

@st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
def get_sampled_pixels(images, sample_size=100_000):
    return np.random.choice(images.flatten(), sample_size, replace=False)

@st.cache_data  # Cache danh sách ảnh ngẫu nhiên
def get_random_indices(num_images, total_images):
    return np.random.randint(0, total_images, size=num_images)


# Định nghĩa đường dẫn đến các file MNIST
# Download latest version
# dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
# train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
# train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
# test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
# test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

# # Tải dữ liệu
# train_images = load_mnist_images(train_images_path)
# train_labels = load_mnist_labels(train_labels_path)
# test_images = load_mnist_images(test_images_path)
# test_labels = load_mnist_labels(test_labels_path)

# st.write(f"Số lượng ảnh trong tập train: {len(train_images)}")
# st.write(f"Số lượng ảnh trong tập train: {len(test_images)}")
# st.subheader("Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị")
# num_images = 10
# random_indices = random.sample(range(len(train_images)), num_images)
# fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

# for ax, idx in zip(axes, random_indices):
#         ax.imshow(train_images[idx], cmap='gray')
#         ax.axis("off")
#         ax.set_title(f"Label: {train_labels[idx]}")

# st.pyplot(fig)

# # Flatten the images
# X_train = train_images.reshape(-1, 28 * 28)
# X_test = test_images.reshape(-1, 28 * 28)
# y_train = train_labels
# y_test = test_labels

def get_X_and_y(dataset):
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="array"
    )
    return X, y

# Tải dữ liệu MNIST từ OpenML
dataset = openml.datasets.get_dataset(554)

# Tải dữ liệu
X, y = get_X_and_y(dataset)

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiển thị dữ liệu trên Streamlit
st.write("Dữ liệu train:")
st.write(X_train)
st.write("Dữ liệu test:")
st.write(X_test)

# Hiển thị dữ liệu trên matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.show()
# Hiển thị 10 ảnh đầu tiên của tập dữ liệu
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    axes[i // 5, i % 5].imshow(X_train[i].reshape(28, 28), cmap='gray')
    axes[i // 5, i % 5].set_title(f"Label: {y_train[i]}")
    axes[i // 5, i % 5].axis('off')

st.pyplot(fig)

# Flatten the images
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

mlflow.start_run()
mlflow.sklearn.log_model(KMeans, "kmeans_model")
mlflow.log_param("n_clusters", 10)
mlflow.end_run()

# Biểu đồ phân phối nhãn dữ liệu
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=list(Counter(y_train).keys()), y=list(Counter(y_train).values()), palette="Blues", ax=ax)
ax.set_title("Phân phối nhãn trong tập huấn luyện")
ax.set_xlabel("Nhãn")
ax.set_ylabel("Số lượng")
st.pyplot(fig)

# Normalize the data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# # Tạo bộ dữ liệu
# train_data = (train_images, train_labels)
# test_data = (test_images, test_labels)

st.subheader("Tùy chọn chia dữ liệu train")
train_ratio = st.slider("Tỷ lệ dữ liệu train (%)", min_value=10, max_value=90, value=80, step=1)
test_ratio = 100 - train_ratio
a = 100 - train_ratio

# Chia tách dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_val_test, y_train, y_val_test = train_test_split(X_train, y_train, test_size=test_ratio/100, random_state=42)

# Tạo phần tùy chọn chia dữ liệu test thành validation và test
st.subheader("Tùy chọn chia dữ liệu test thành validation và test")
val_ratio = st.slider("Tỷ lệ dữ liệu validation (%)", min_value=0, max_value=a, value=a, step=1)

x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=(100-val_ratio)/100, random_state=42)

# In ra số lượng của các tập train, test và val
st.subheader("Số lượng của các tập dữ liệu")
st.write("Số lượng dữ liệu train: ", len(x_train))
st.write("Số lượng dữ liệu validation: ", len(x_val))
st.write("Số lượng dữ liệu test: ", len(x_test))



# # Huấn luyện model K-means
# kmeans = KMeans(n_clusters=10, random_state=42)
# kmeans.fit(X_train)

# # # Huấn luyện model DBSCAN
# # dbscan = DBSCAN(eps=0.5, min_samples=5)
# # dbscan.fit(X_train)

# # Đánh giá model K-means
# silhouette_kmeans = silhouette_score(X_train, kmeans.labels_)
# calinski_harabasz_kmeans = calinski_harabasz_score(X_train, kmeans.labels_)
# davies_bouldin_kmeans = davies_bouldin_score(X_train, kmeans.labels_)

# # Đánh giá model DBSCAN
# silhouette_dbscan = silhouette_score(X_train, dbscan.labels_)
# calinski_harabasz_dbscan = calinski_harabasz_score(X_train, dbscan.labels_)
# davies_bouldin_dbscan = davies_bouldin_score(X_train, dbscan.labels_)

# # # Logging với MLFlow
# # mlflow.start_run()
# # mlflow.sklearn.log_model(kmeans, "kmeans_model")
# # mlflow.log_param("n_clusters", 10)
# # mlflow.log_metric("silhouette_score", silhouette_kmeans)
# # mlflow.log_metric("calinski_harabasz_score", calinski_harabasz_kmeans)
# # mlflow.log_metric("davies_bouldin_score", davies_bouldin_kmeans)
# # mlflow.end_run()

# # mlflow.start_run()
# # mlflow.sklearn.log_model(dbscan, "dbscan_model")
# # mlflow.log_param("eps", 0.5)
# # mlflow.log_param("min_samples", 5)
# # mlflow.log_metric("silhouette_score", silhouette_dbscan)
# # mlflow.log_metric("calinski_harabasz_score", calinski_harabasz_dbscan)
# # mlflow.log_metric("davies_bouldin_score", davies_bouldin_dbscan)
# # mlflow.end_run()

# # Hiển thị kết quả lên Streamlit
# st.title("Kết quả phân cụm")

# st.subheader("K-means")
# st.write(f"Silhouette Score: {silhouette_kmeans}")
# st.write(f"Calinski-Harabasz Index: {calinski_harabasz_kmeans}")
# st.write(f"Davies-Bouldin Index: {davies_bouldin_kmeans}")

# st.subheader("DBSCAN")
# st.write(f"Silhouette Score: {silhouette_dbscan}")
# st.write(f"Calinski-Harabasz Index: {calinski_harabasz_dbscan}")
# st.write(f"Davies-Bouldin Index: {davies_bouldin_dbscan}")

# # Hiển thị dữ liệu đã phân cụm lên Streamlit
# st.subheader("Dữ liệu đã phân cụm")
# col1, col2 = st.columns(2)

# with col1:
#     plt.figure(figsize=(10, 10))
#     plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.labels_, cmap='viridis', s=15)
#     plt.title('K-means Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     st.pyplot(plt)

# with col2:
#     plt.figure(figsize=(10, 10))
#     plt.scatter(X_train[:, 0], X_train[:, 1], c=dbscan.labels_, cmap='viridis', s=15)
#     plt.title('DBSCAN Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     st.pyplot(plt)
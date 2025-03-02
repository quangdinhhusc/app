from email.mime import image
from sklearn.cluster import KMeans, DBSCAN
import joblib
import streamlit as st
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
import altair
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from collections import Counter
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import kagglehub
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE




# Định nghĩa hàm tải dữ liệu
def load_data():
    dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    return train_images, train_labels, test_images, test_labels

# Định nghĩa hàm tải dữ liệu MNIST
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Định nghĩa hàm xử lí dữ liệu
def process_data(train_images, train_labels, test_images, test_labels):
    X_train = train_images.reshape(-1, 28 * 28)
    X_test = test_images.reshape(-1, 28 * 28)
    y_train = train_labels
    y_test = test_labels

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return X_train, X_test, y_train, y_test

# Định nghĩa hàm chia tập dữ liệu
def split_data(X_train, X_test, y_train, y_test, train_ratio):
    test_ratio = 100 - train_ratio
    X_train, val_x, y_train, val_y = train_test_split(X_train, y_train, test_size=test_ratio/100, random_state=42)

    return X_train, val_x, y_train, val_y

# Định nghĩa hàm giảm chiều dữ liệu bằng PCA
def reduce_dimension_pca(X_train, X_test, n_components):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Định nghĩa hàm giảm chiều dữ liệu bằng t-SNE
def reduce_dimension_tsne(X_train, X_test, n_components):
    tsne = TSNE(n_components=n_components, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train)
    X_test_tsne = tsne.fit_transform(X_test)
    return X_train_tsne, X_test_tsne

# Định nghĩa hàm huấn luyện mô hình
def train_model(X_train, model_choice, n_clusters=None, eps=None, min_samples=None):
    if model_choice == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif model_choice == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X_train)
    return model

# Định nghĩa hàm demo dự đoán
def demo_prediction(model, uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape(1, 784)
    prediction = model.predict(image)
    return prediction

# Định nghĩa hàm chính
def main():
    st.set_page_config(page_title="Phân loại ảnh", layout="wide")
    st.title("MNIST Assignment - Clustering Algorithms with Streamlit & MLFlow")
    st.markdown("""
                **Tập dữ liệu MNIST (Modified National Institute of Standards and Technology database)** là một bộ dữ liệu kinh điển trong lĩnh vực học máy, đặc biệt là trong thị giác máy tính. 
                - Bộ dữ liệu này được "modified" từ bộ dữ liệu gốc của NIST, nhằm mục đích làm cho nó phù hợp hơn cho các thí nghiệm học máy.
                - Bộ dữ liệu gồm 60.000 ảnh dùng để hun luyện và 10.000 ảnh dùng để kiểm tra. 
                - Mỗi ảnh là một chữ số viết tay từ 0 đến 9 với kích thước 28x28 pixel.
                - MNIST được tạo ra để huấn luyện và đánh giá các thuật toán nhận dạng chữ số viết tay.
                """)

    # Tải dữ liệu
    train_images, train_labels, test_images, test_labels = load_data()

    # Xử lí dữ liệu
    X_train, X_test, y_train, y_test = process_data(train_images, train_labels, test_images, test_labels)

    # Chia tập dữ liệu
    train_ratio = st.slider("Tỷ lệ dữ liệu train (%)", min_value=10, max_value=100, value=90, step=1)
    X_train, val_x, y_train, val_y = split_data(X_train, X_test, y_train, y_test, train_ratio)

    # Chọn phương pháp giảm chiều dữ liệu
    method = st.radio("Chọn phương pháp giảm chiều dữ liệu:", ["PCA", "t-SNE"])

    # Chọn số chiều giảm xuống
    n_components = st.slider("Chọn số chiều giảm xuống", 1, 784, 500)

    # Giảm chiều dữ liệu
    if method == "PCA":
        X_train, X_test = reduce_dimension_pca(X_train, X_test, n_components)
    elif method == "t-SNE":
        X_train, X_test = reduce_dimension_tsne(X_train, X_test, n_components)

    # Huấn luyện mô hình
    model_choice = st.radio("Chọn mô hình:", ["K-Means", "DBSCAN"])
    if model_choice == "K-Means":
        n_clusters = st.slider("n_clusters", 2, 20, 10)
        model = train_model(X_train, model_choice, n_clusters=n_clusters)
    elif model_choice == "DBSCAN":
        eps = st.slider("eps", 0.1, 10.0, 0.5)
        min_samples = st.slider("min_samples", 2, 20, 5)
        model = train_model(X_train, model_choice, eps=eps, min_samples=min_samples)

    # Demo dự đoán
    uploaded_file = st.sidebar.file_uploader("Chọn hình ảnh", type=["png", "jpg", "jpeg"])
    if st.sidebar.button("Kiểm tra"):
        prediction = demo_prediction(model, uploaded_file)
        st.sidebar.image(image.reshape(28, 28), caption="Hình ảnh chữ viết tay")
        st.sidebar.write("Kết quả dự đoán:")
        st.sidebar.write("Chữ viết tay:", prediction[0])

if __name__ == "__main__":
    main()
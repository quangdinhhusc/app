import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # type: ignore

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Streamlit app
st.title("MNIST Classification with Streamlit & MLFlow")

st.sidebar.header("Model Selection")
model_name = st.sidebar.radio("", ["Decision Tree", "SVM"])

# st.title("Chọn tỉ lệ của các tập dữ liệu")
# train_ratio = st.slider("Tập huấn luyện", 0, 90, 70)
# a = 100 - train_ratio
# val_ratio = st.slider("Tập xác thực", 0, a, 5)

# # Tính toán tỉ lệ của tập kiểm tra
# test_ratio = 100 - train_ratio - val_ratio
# total_data = len(mnist["data"])
# train_size = int(total_data * train_ratio / 100)
# val_size = int(total_data * val_ratio / 100)
# test_size = total_data - train_size - val_size

# st.write("Số lượng của các tập dữ liệu:")
# st.write("Tập huấn luyện:", train_size)
# st.write("Tập xác thực:", val_size)
# st.write("Tập kiểm tra:", test_size)

st.subheader("Tùy chọn chia dữ liệu train")
train_ratio = st.slider("Tỷ lệ dữ liệu train", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
test_ratio = 1 - train_ratio

# Chia tách dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=1-train_ratio, random_state=42)

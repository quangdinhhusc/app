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
train_ratio = st.slider("Tỷ lệ dữ liệu train (%)", min_value=10, max_value=90, value=80, step=10)
test_ratio = 100 - train_ratio

# Chia tách dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=42)

# Tạo phần tùy chọn chia dữ liệu test thành validation và test
st.subheader("Tùy chọn chia dữ liệu test thành validation và test")
val_ratio = st.slider("Tỷ lệ dữ liệu validation (%)", min_value=10, max_value=90, value=50, step=10)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(100-val_ratio)/100, random_state=42)

# In ra số lượng của các tập train, test và val
st.subheader("Số lượng của các tập dữ liệu")
st.write("Số lượng dữ liệu train: ", len(x_train))
st.write("Số lượng dữ liệu validation: ", len(x_val))
st.write("Số lượng dữ liệu test: ", len(x_test))

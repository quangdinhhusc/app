import joblib
import streamlit as st
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



st.set_page_config(page_title="Phân loại ảnh", layout="wide")

# Streamlit app
st.title("MNIST Classification with Streamlit & MLFlow")

@st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
def get_sampled_pixels(images, sample_size=100_000):
    return np.random.choice(images.flatten(), sample_size, replace=False)

@st.cache_data  # Cache danh sách ảnh ngẫu nhiên
def get_random_indices(num_images, total_images):
    return np.random.randint(0, total_images, size=num_images)

# # Cấu hình Streamlit
# def config_page():
#     st.set_page_config(page_title="Phân loại ảnh", layout="wide", initial_sidebar_state="expanded")

# config_page()

# st.set_page_config(page_title="Phân loại ảnh", layout="wide", initial_sidebar_state="expanded")
# Định nghĩa hàm để đọc file .idx
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

# Định nghĩa đường dẫn đến các file MNIST
# Download latest version
dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
# dataset_path = os.path.dirname(os.path.abspath(__file__))
train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

# Tải dữ liệu
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Flatten the images
X_train = train_images.reshape(-1, 28 * 28)
X_test = test_images.reshape(-1, 28 * 28)
y_train = train_labels
y_test = test_labels

# Normalize the data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Tạo bộ dữ liệu
train_data = (train_images, train_labels)
test_data = (test_images, test_labels)

# # Kiểm tra giá trị null hoặc NaN trong dữ liệu
# na_counts = X.isna().sum()
# st.write("Số lượng giá trị null hoặc NaN trong dữ liệu:", na_counts)

# # Kiểm tra giá trị vô hạn trong dữ liệu
# inf_counts = X.isinf().sum()
# st.write("Số lượng giá trị vô hạn trong dữ liệu:", inf_counts)

# # Đếm số lượng của mỗi giá trị trong dữ liệu
# value_counts = X.apply(lambda x: x.value_counts())
# st.write("Số lượng của mỗi giá trị trong dữ liệu:", value_counts)

# # Tạo phần tùy chọn chia dữ liệu train
# st.subheader("Tùy chọn chia dữ liệu train")
# train_ratio = st.slider("Tỷ lệ dữ liệu train (%)", min_value=10, max_value=90, value=70, step=1)
# test_ratio = 100 - train_ratio
# a = 100 - train_ratio

# # Chia tách dữ liệu thành tập huấn luyện và kiểm tra
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=42)

# # Tạo phần tùy chọn chia dữ liệu test thành validation và test
# st.subheader("Tùy chọn chia dữ liệu test thành validation và test")
# val_ratio = st.slider("Tỷ lệ dữ liệu validation (%)", min_value=0, max_value=a, value=10, step=1)

# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(100-val_ratio)/100, random_state=42)

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

# Chọn model
st.sidebar.header("Model Selection")
model_name = st.sidebar.radio("", ["Decision Tree", "SVM"])

# Train and evaluate model
if st.sidebar.button("Train Model"):
    if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
    elif model_name == "SVM":
            param_grid = {
                'C': [0.1],
                'kernel': ['linear'],
                'gamma': [0.1]
            }
            grid_search = GridSearchCV(SVC(), param_grid, cv=5)
            grid_search.fit(x_train, y_train)
            model = SVC(kernel="linear", random_state=42)
    with mlflow.start_run():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log parameters and metrics to MLFlow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # Display MLFlow logs in Streamlit
        st.subheader("MLFlow Logs")
        st.write("Run ID:", mlflow.active_run().info.run_id)
        st.write("Experiment ID:", mlflow.active_run().info.experiment_id)
        st.write(f"Model: {model_name}")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.write("Confusion Matrix:")
        # st.write(cm)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        st.pyplot(fig)

        # Display MLFlow metrics in Streamlit
        # st.subheader("MLFlow Metrics")
        # st.write("Accuracy:", mlflow.active_run().data.metrics["accuracy"])
        # st.write("Accuracy:", mlflow.active_run().data.metrics["accuracy:" + mlflow.active_run().info.run_id])

        # Display MLFlow parameters in Streamlit
        # st.subheader("MLFlow Parameters")
        # st.write("Model:", mlflow.get_param("model"))

        # # Save model to MLFlow
        # mlflow.sklearn.log_model(model, "model")

        # # Display MLFlow model in Streamlit
        # st.subheader("MLFlow Model")
        # st.write("Model:", mlflow.get_model("model"))

        # # Display classification report
        # st.write("Classification Report:")
        # st.write(classification_report(y_test, y_pred))

        # # Display precision, recall, f1-score
        # st.write("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
        # st.write("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
        # st.write("F1-score: {:.2f}".format(f1_score(y_test, y_pred)))

        # Save model to MLFlow
        mlflow.sklearn.log_model(model, "model", input_example=x_train[:1])
    # Huấn luyện mô hình
    model.fit(x_train, y_train)
    joblib.dump(model, "model.joblib")
    st.sidebar.subheader("Demo dự đoán chữ viết tay")
    st.sidebar.write("Vui lòng nhập hình ảnh chữ viết tay để dự đoán:")

    # Tạo phần nhập hình ảnh
    uploaded_file = st.sidebar.file_uploader("Chọn hình ảnh", type=["png", "jpg", "jpeg"])

    # Xử lý hình ảnh
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((28, 28))
        image = image.convert('L')
        image = np.array(image)
        image = image.reshape(1, 784)

        # Dự đoán chữ viết tay
        prediction = model.predict(image)


        # Hiển thị kết quả
        st.sidebar.write("Kết quả dự đoán:")
        st.sidebar.write("Chữ viết tay:", prediction[0])




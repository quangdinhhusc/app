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




st.set_page_config(page_title="Phân loại ảnh", layout="wide")

# Streamlit app
st.title("MNIST Assignment - Clustering Algorithms with Streamlit & MLFlow")
st.markdown("""
            **Tập dữ liệu MNIST (Modified National Institute of Standards and Technology database)** là một bộ dữ liệu kinh điển trong lĩnh vực học máy, đặc biệt là trong thị giác máy tính. 
            - Bộ dữ liệu này được "modified" từ bộ dữ liệu gốc của NIST, nhằm mục đích làm cho nó phù hợp hơn cho các thí nghiệm học máy.
            - Bộ dữ liệu gồm 60.000 ảnh dùng để hun luyện và 10.000 ảnh dùng để kiểm tra. 
            - Mỗi ảnh là một chữ số viết tay từ 0 đến 9 với kích thước 28x28 pixel.
            - MNIST được tạo ra để huấn luyện và đánh giá các thuật toán nhận dạng chữ số viết tay.
            """)

@st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
def get_sampled_pixels(images, sample_size=100_000):
    return np.random.choice(images.flatten(), sample_size, replace=False)

@st.cache_data  # Cache danh sách ảnh ngẫu nhiên
def get_random_indices(num_images, total_images):
    return np.random.randint(0, total_images, size=num_images)

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

st.write(f"Số lượng ảnh trong tập train: {len(train_images)}")
st.write(f"Số lượng ảnh trong tập test: {len(test_images)}")

def display_mnist_grid():
    # Tải tập dữ liệu MNIST
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    
    # Số hàng và cột
    num_rows, num_cols = 10, 10
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 6))
    fig.suptitle("Một số hình ảnh từ MNIST Dataset", fontsize=14, fontweight='bold')
    
    for i in range(num_rows):
        for j in range(num_cols):
            index = np.where(y_train == i)[0][j]  # Lấy ảnh thứ j của số i
            axes[i, j].imshow(x_train[index], cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(str(i), fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig

# Demo dự đoán chữ viết tay
st.sidebar.subheader("Demo dự đoán chữ viết tay")
st.sidebar.write("Vui lòng chuyển hình ảnh chữ viết tay cần dự đoán:")

# Tạo phần nhập hình ảnh
uploaded_file = st.sidebar.file_uploader("Chọn hình ảnh", type=["png", "jpg", "jpeg"])

# Hiển thị hình ảnh
st.subheader("Hiển thị MNIST Dataset trên Streamlit")
st.pyplot(display_mnist_grid())

# Flatten the images
X_train = train_images.reshape(-1, 28 * 28)
X_test = test_images.reshape(-1, 28 * 28)
y_train = train_labels
y_test = test_labels

# Biểu đồ phân phối nhãn dữ liệu
st.subheader("Biểu đồ phân phối nhãn dữ liệu")     
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=list(Counter(train_labels).keys()), y=list(Counter(train_labels).values()), palette="Blues", ax=ax)
ax.set_title("Phân phối nhãn trong tập huấn luyện")
ax.set_xlabel("Label")
ax.set_ylabel("Số lượng")
st.pyplot(fig)

# Normalize the data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Tạo bộ dữ liệu
train_data = (train_images, train_labels)
test_data = (test_images, test_labels)

st.subheader("Tùy chọn chia dữ liệu train")
train_ratio = st.slider("Tỷ lệ dữ liệu train (%)", min_value=10, max_value=100, value=90, step=1)
test_ratio = 100 - train_ratio
x_test = X_test
y_test = y_test

if train_ratio == 100:
    x_train = X_train
    y_train = y_train
else:
    # Chia tách dữ liệu thành tập huấn luyện và kiểm tra
    x_train, val_x, y_train, val_y = train_test_split(X_train, y_train, test_size=test_ratio/100, random_state=42)

    # Tạo phần lựa chọn dữ liệu tập val
    st.subheader("Tùy chọn dữ liệu tập val")
    val_ratio = st.slider("Tỷ lệ dữ liệu tập val (%)", min_value=0, max_value=test_ratio, value=test_ratio, step=1)

    # Chia tách dữ liệu tập val thành tập val và tập test
    if val_ratio == test_ratio:
        x_test_add = 0
        y_test_add = 0
    elif val_ratio < test_ratio:
        x_val, x_test_add, y_val, y_test_add = train_test_split(val_x, val_y, test_size=(test_ratio-val_ratio)/test_ratio, random_state=42)
        # Cộng thêm dữ liệu tập test
        x_test = np.concatenate((X_test, x_test_add))
        y_test = np.concatenate((y_test, y_test_add))
    else:
        x_val = np.array([])
        y_val = np.array([])
        x_test_add = val_x
        y_test_add = val_y
        # Cộng thêm dữ liệu tập test
        x_test = np.concatenate((X_test, x_test_add))
        y_test = np.concatenate((y_test, y_test_add))



# In ra số lượng của các tập train, test và val
st.subheader("Số lượng của các tập dữ liệu")
st.write("Số lượng dữ liệu train: ", len(x_train))
st.write("Số lượng dữ liệu validation: ", len(val_x))
st.write("Số lượng dữ liệu test: ", len(x_test))

# Lựa chọn phương pháp xử lý dữ liệu
method = st.radio("Chọn phương pháp xử lý dữ liệu:", ["PCA", "t-SNE"])

if method == "PCA":
    st.markdown("""
    - **PCA (Principal Component Analysis)** là một phương pháp chuyển đổi một tập hợp các biến có tương quan thành một tập hợp các biến không tương quan tuyến tính, được gọi là các thành phần chính, đồng thời giữ lại càng nhiều thông tin quan trọng của dữ liệu gốc càng tốt.
    - **Tham số cần chọn:**
        - Số chiều giảm xuống.  
    """)
    # Lựa chọn số chiều giảm xuống
    n_components = st.slider("Chọn số chiều giảm xuống", 1, 784, 500)

    # Chuẩn hóa dữ liệu X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test)
    st.write("Số chiều dữ liệu sau khi giảm:", X_train_pca.shape[1])
elif method == "t-SNE":
    st.markdown("""
    - **t-SNE (t-distributed Stochastic Neighbor Embedding)** là một phương pháp giảm chiều dữ liệu không tuyến tính, giữ lại cấu trúc của dữ liệu gốc. Đặc biệt hiệu quả trong việc trực quan hóa dữ liệu có số chiều cao.
    - **Tham số cần chọn:**
        - Số chiều giảm xuống.  
    """)
    # Lựa chọn số chiều giảm xuống
    n_components = st.slider("Chọn số chiều giảm xuống", 1, 3, 2)
    tsne = TSNE(n_components=n_components, random_state=42)
    # Giảm chiều dữ liệu train và test
    X_train_tsne = tsne.fit_transform(X_train)
    X_test_tsne = tsne.fit_transform(X_test)
    st.write("Số chiều dữ liệu sau khi giảm:", X_train_tsne.shape[1])

st.header("Chọn mô hình & Huấn luyện")

# Chọn mô hình
model_choice = st.radio("Chọn mô hình:", ["K-Means", "DBSCAN"])

if model_choice == "K-Means":
    st.markdown("""
    - **K-means clustering** là thuật toán phân cụm phổ biến, chia dữ liệu thành K cụm dựa trên khoảng cách.
    - **Tham số cần chọn:**
        - Số lượng cụm (k).  
    """)
        
    n_clusters = st.slider("n_clusters", 2, 20, 10)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    
elif model_choice == "DBSCAN":
    st.markdown("""
    - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là thuật toán phân cụm dựa trên mật độ.
    - **Tham số cần chọn:**  
        - Bán kính lân cận.  
        - Số lượng điểm tối thiểu để tạo cụm.  
    """)
    eps = st.slider("eps", 0.1, 10.0, 0.5)
    min_samples = st.slider("min_samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)



if st.button("Huấn luyện mô hình"):
    model.fit(X_train)
    labels = model.labels_
    st.success("✅ Huấn luyện thành công!")

    # Lưu mô hình vào session_state dưới dạng danh sách nếu chưa có
    if "models" not in st.session_state:
        st.session_state["models"] = []

    model_name = model_choice.lower().replace(" ", "_")

    existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
        
    if existing_model:
        count = 1
        new_model_name = f"{model_name}_{count}"
        while any(item["name"] == new_model_name for item in st.session_state["models"]):
            count += 1
            new_model_name = f"{model_name}_{count}"
        model_name = new_model_name


    # Tạo nút kiểm tra
    if st.sidebar.button("Kiểm tra"):
        # Xử lý hình ảnh
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((28, 28))
            image = image.convert('L')
            image = np.array(image)
            image = image.reshape(1, 784)

            # Dự đoán chữ viết tay
            prediction = model.predict(image)

            # Hiển thị hình ảnh
            st.sidebar.image(image.reshape(28, 28), caption="Hình ảnh chữ viết tay")

            # Hiển thị kết quả
            st.sidebar.write("Kết quả dự đoán:")
            st.sidebar.write("Chữ viết tay:", prediction[0])
        else:
            st.sidebar.write("Vui lòng nhập hình ảnh chữ viết tay để dự đoán:")



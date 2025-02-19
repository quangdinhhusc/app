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
import numpy as np
from PIL import Image # type: ignore
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Streamlit app
st.title("MNIST Classification with Streamlit & MLFlow")

st.write("Bảng dữ liệu gốc:")
st.write(X.head())
st.write("Số lượng dữ liệu:", len(X))
st.write("Số lượng thuộc tính:", len(X.columns))

# st.write("Bảng dữ liệu số lượng dữ liệu lỗi hoặc NULL của các cột:")
# st.write(X.isnull().sum())
# st.write("Tỷ lệ dữ liệu lỗi hoặc NULL của các cột:")
# st.write(X.isnull().mean())


st.sidebar.header("Model Selection")
model_name = st.sidebar.radio("", ["Decision Tree", "SVM"])

st.subheader("Tùy chọn chia dữ liệu train")
train_ratio = st.slider("Tỷ lệ dữ liệu train (%)", min_value=10, max_value=90, value=70, step=1)
test_ratio = 100 - train_ratio

a = 100 - train_ratio

# Chia tách dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=42)

# Tạo phần tùy chọn chia dữ liệu test thành validation và test
st.subheader("Tùy chọn chia dữ liệu test thành validation và test")
val_ratio = st.slider("Tỷ lệ dữ liệu validation (%)", min_value=0, max_value=a, value=10, step=1)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(100-val_ratio)/100, random_state=42)

# In ra số lượng của các tập train, test và val
st.subheader("Số lượng của các tập dữ liệu")
st.write("Số lượng dữ liệu train: ", len(x_train))
st.write("Số lượng dữ liệu validation: ", len(x_val))
st.write("Số lượng dữ liệu test: ", len(x_test))

# Train and evaluate model
if st.sidebar.button("Train Model"):
    with mlflow.start_run():
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "SVM":
            model = SVC()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log parameters and metrics to MLFlow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # Display results in Streamlit
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

        # # Display classification report
        # st.write("Classification Report:")
        # st.write(classification_report(y_test, y_pred))

        # # Display precision, recall, f1-score
        # st.write("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
        # st.write("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
        # st.write("F1-score: {:.2f}".format(f1_score(y_test, y_pred)))

        # Save model to MLFlow
        mlflow.sklearn.log_model(model, "model")

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
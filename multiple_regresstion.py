import streamlit as st
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Tải dữ liệu
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv" # Link dữ liệu Titanic
df = data_url
st.subheader("Dữ liệu Titanic")
st.write(df)

# Tiền xử lý dữ liệu

# Chia tập dữ liệu
# đưa dữ liệu vào X và y
X = df.drop('Survived', axis=1)
y = df['Survived']

# chia dữ liệu thành train 70, test 15, valid 15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Huấn luyện mô hình
def train_model(train_df, val_df, params):
    with mlflow.start_run():
        # Ghi lại các tham số
        mlflow.log_params(params)

        # Huấn luyện mô hình
        model = LinearRegression(**params)
        model.fit(train_df.drop("Survived", axis=1), train_df["Survived"])

        # Đánh giá mô hình trên tập validation
        y_pred = model.predict(val_df.drop("Survived", axis=1))
        mse = mean_squared_error(val_df["Survived"], y_pred)
        r2 = r2_score(val_df["Survived"], y_pred)

        # Ghi lại các metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Lưu mô hình
        mlflow.sklearn.log_model(model, "model")

        # Cross-validation
        cv_scores = cross_val_score(model, train_df.drop("Survived", axis=1), train_df["Survived"], cv=5, scoring="neg_mean_squared_error")
        mlflow.log_metric("cv_mse", -cv_scores.mean())

        return model

# Khởi tạo MLflow
mlflow.set_tracking_uri("runs:/mlruns") # Lưu trữ logs tại thư mục mlruns

# Định nghĩa các tham số
params = {
    "fit_intercept": True
}

# Huấn luyện mô hình
model = train_model(X_train, y_train, params)

st.title("Titanic Survival Prediction")

# Chọn model từ MLflow
model_uri = "runs:/mlruns/0/model" # Thay đổi ID của run nếu cần
model = mlflow.sklearn.load_model(model_uri)

# Hiển thị các metrics
run = mlflow.get_run(model_uri.split("/")[2])
metrics = run.data.metrics
st.write("Metrics:")
st.write(metrics)

# Demo dự đoán
st.subheader("Prediction")
# Tạo form nhập liệu
# ...

# Dự đoán kết quả
# ...
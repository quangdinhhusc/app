import streamlit as st
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


# Tiêu đề ứng dụng
st.title("Ứng dụng Titanic với Streamlit")
st.write("""
## Phân tích dữ liệu và huấn luyện mô hình Multiple Rgresstion
""")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Hiển thị dữ liệu gốc
st.subheader("Dữ liệu Titanic gốc")
st.write(data)

# Tiền xử lý dữ liệu
st.subheader("Tiền xử lý dữ liệu")

# Xóa các dòng có ít nhất 2 cột chứa giá trị null
thresh_value = data.shape[1] - 1
df_cleaned = data.dropna(thresh=thresh_value)
st.write("- Xóa các dòng có ít nhất 2 cột chứa giá trị null.")
st.write(f"Số dòng sau khi xóa: {df_cleaned.shape[0]}")

st.write("- Điền dữ liệu tuổi null thành giá trị trung bình của tuổi.")
st.write("- Điền dữ liệu Embarked null thành giá trị mode của Embarked.")
# xử lý dữ liệu
data_cleaned = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data_cleaned['Age'] = data_cleaned['Age'].fillna(data_cleaned['Age'].median())
data_cleaned['Embarked'] = data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].mode()[0])
data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex', 'Embarked'], drop_first=True)

# Hiển thị dữ liệu sau khi tiền xử lý
st.write("Dữ liệu sau khi tiền xử lý:")
st.write(data_cleaned)

# Chia tập dữ liệu
# đưa dữ liệu vào X và y
X = data_cleaned.drop('Survived', axis=1)
y = data_cleaned['Survived']
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
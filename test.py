
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

st.write("- Xóa một số cột giá trị có thể gây ảnh hưởng (như chứa nhiều dữ liệu bị nhiễu, dữ liệu không nhất quá,...) đến quá trình huấn luyện model")
data_cleaned = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
st.write(f"""1. PassengerId:
- Đây là một định danh duy nhất cho mỗi hành khách và không mang thông tin có giá trị dự đoán về khả năng sống sót.
- Việc đưa PassengerId vào mô hình có thể gây nhầm lẫn hoặc làm giảm hiệu suất của mô hình.

2. Name:
- Tên hành khách thường là dữ liệu dạng text và rất đa dạng.
- Mặc dù có thể trích xuất một số thông tin (ví dụ: tước hiệu), nhưng việc xử lý tên phức tạp và không chắc chắn mang lại lợi ích đáng kể cho mô hình.
- Trong trường hợp này, chúng ta đơn giản hóa bằng cách loại bỏ cột Name.

3. Ticket:
- Số vé cũng là một định danh và không có mối quan hệ rõ ràng với khả năng sống sót.

4. Cabin:
- Cột Cabin chứa nhiều giá trị bị thiếu (NaN).
- Việc xử lý các giá trị thiếu này có thể phức tạp.
- Hơn nữa, thông tin về cabin có thể không phải là yếu tố quyết định đến khả năng sống sót""")

st.write("- Điền dữ liệu tuổi null thành giá trị trung bình của tuổi.")
data_cleaned['Age'] = data_cleaned['Age'].fillna(data_cleaned['Age'].median())

st.write("- Điền dữ liệu Embarked null thành giá trị mode của Embarked.")
data_cleaned['Embarked'] = data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].mode()[0])

st.write("- Chuẩn hóa các cột về các giá trị để giúp cho quá trình huấn luyện.")
# data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex', 'Embarked'], drop_first=True)
data_cleaned['Embarked'] = data_cleaned['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})
data_cleaned['Fare'] = data_cleaned['Fare'].round().astype(int)


# Hiển thị dữ liệu sau khi tiền xử lý
st.write("Dữ liệu sau khi tiền xử lý:")
st.write(data_cleaned)

# Chia tập dữ liệu
def split_data(df, train_ratio, val_ratio, test_ratio, random_state):
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio/(train_ratio+val_ratio), random_state=random_state)
    return train_df, val_df, test_df

# Chia tập dữ liệu
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_state = 42
train_df, val_df, test_df = split_data(data_cleaned, train_ratio, val_ratio, test_ratio, random_state)

# Định nghĩa các tham số
params = {
    "fit_intercept": True
}

# Huấn luyện mô hình
def train_model(train_df, val_df, params):
    # with mlflow.start_run():
    #     # Ghi lại các tham số
    #     mlflow.log_params(params)

        # Huấn luyện mô hình
        model = LinearRegression(**params)
        model.fit(train_df.drop("Survived", axis=1), train_df["Survived"])

        # # Đánh giá mô hình trên tập validation
        # y_pred = model.predict(val_df.drop("Survived", axis=1))
        # mse = mean_squared_error(val_df["Survived"], y_pred)
        # r2 = r2_score(val_df["Survived"], y_pred)

        # # Ghi lại các metrics
        # mlflow.log_metric("mse", mse)
        # mlflow.log_metric("r2", r2)

        # # Lưu mô hình
        # mlflow.sklearn.log_model(model, "model")

        # # Cross-validation
        # cv_scores = cross_val_score(model, train_df.drop("Survived", axis=1), train_df["Survived"], cv=5, scoring="neg_mean_squared_error")
        # mlflow.log_metric("cv_mse", -cv_scores.mean())

        return model

# Khởi tạo MLflow
# mlflow.set_tracking_uri("runs:/mlruns") # Lưu trữ logs tại thư mục mlruns

# Huấn luyện mô hình
try:
    model = train_model(train_df, val_df, params)
except Exception as e:
    print(f"Error training model: {e}")
    print(f"train_df: {train_df.info()}")
    print(f"val_df: {val_df.info()}")
    print(f"params: {params}")


# Cross-validation
cv_scores = cross_val_score(train_model(train_df, val_df, params), 
                            train_df.drop("Survived", axis=1), 
                            train_df["Survived"], cv=5, 
                            scoring="neg_mean_squared_error")
st.write(f"Độ chính xác trung bình sau Cross-Validation: {cv_scores.mean():.2f}")

# Đánh giá mô hình trên tập validation
y_pred = model.predict(val_df.drop("Survived", axis=1))
valid_accuracy = r2_score(val_df["Survived"], y_pred)
st.write(f"Độ chính xác trên tập Validation: {valid_accuracy:.2f}")


# Đánh giá mô hình trên tập test
y_test_pred = model.predict(test_df.drop("Survived", axis=1))
test_accuracy = r2_score(test_df["Survived"], y_test_pred)
st.write(f"Độ chính xác trên tập Test: {test_accuracy:.2f}")

# Hiển thị biểu đồ phân phối độ tuổi
st.subheader("Phân phối độ tuổi của hành khách")
st.write("- Trong dữ liệu cho thấy phần lớn khách hàng nằm trong độ tuổi từ 20 đến 40.")
fig, ax = plt.subplots()
sns.histplot(data['Age'].dropna(), kde=True, ax=ax)
st.pyplot(fig)

# Hiển thị biểu đồ tương quan giữa các đặc trưng
st.subheader("Tương quan giữa các đặc trưng")
st.write("- Mô hình có vẻ tập trung vào các yếu tố quan trọng như hạng vé, giới tính và giá vé, những yếu tố có tương quan mạnh với khả năng sống sót.")
st.write("- Các yếu tố như tuổi, số anh chị em, số bố mẹ con cái và cảng lên tàu có tương quan ít với khả năng sống sót.")
st.write("- Mô hình Multiple Regression có thể chưa phải là mô hình tối ưu cho bài toán này. Có thể có những yếu tố khác chưa được xem xét hoặc có những mô hình phức tạp hơn có thể cho kết quả tốt hơn.")

fig, ax = plt.subplots()
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Chọn model từ MLflow
# model_uri = "runs:/mlruns/0/model" # Thay đổi ID của run nếu cần
# model = mlflow.sklearn.load_model(model_uri)

# Hiển thị các metrics
# run = mlflow.get_run(model_uri.split("/")[2])
# metrics = run.data.metrics
# st.write("Metrics:")
# st.write(metrics)

# # Demo dự đoán
# st.subheader("Prediction")

# # ...existing code...

# # Chuyển đổi dữ liệu (ví dụ: one-hot encoding cho biến categorical)
# input_df = pd.get_dummies(data_cleaned, columns=['Sex', 'Embarked'], drop_first=True)

# # Đảm bảo các cột của input_df khớp với các cột của train_df
# missing_cols = set(train_df.drop("Survived", axis=1).columns) - set(input_df.columns)
# for col in missing_cols:
#     input_df[col] = 0
# input_df = input_df[train_df.drop("Survived", axis=1).columns]

# # Dự đoán kết quả
# if st.button("Predict"):
#     prediction = model.predict(input_df)
#     st.write(f"Prediction: {prediction[0]}")

# Lấy tên đặc trưng huấn luyện (bạn cần đoạn code này khi train model)
train_features = train_df.drop("Survived", axis=1).columns.tolist()  # Giả sử "Survived" là cột mục tiêu

# # ...existing code...
st.sidebar.title("Titanic Survival Prediction")

# Tạo form nhập liệu trong sidebar

with st.sidebar.form("input_form"):
    pclass = st.selectbox("Hạng Vé", [1, 2, 3])
    sex = st.selectbox("Giới Tính", ["male", "female"])
    sex = st.selectbox("Giới Tính", ["0 (male)", "1 (female)"])
    # sex = st.selectbox("Giới Tính", ["male", "female"])
    age = st.number_input("Tuổi", min_value=0, max_value=100, value=25)
    sibsp = st.number_input("Anh Chị Em", min_value=0, value=0)
    parch = st.number_input("Bố Mẹ Con Cái", min_value=0, value=0)
    fare = st.number_input("Giá Vé", min_value=0, value=0)  # Đã sửa lỗi ở đây
    embarked = st.selectbox("Cảng", ["Southampton", "Cherbourg", "Queenstown"])
    embarked = st.selectbox("Cảng", ["1 (Southampton)", "2 (Cherbourg)", "3 (Queenstown)"])
    submit_button = st.form_submit_button("Dự Đoán")

if submit_button:
    # Tạo DataFrame từ dữ liệu nhập vào
    data = {
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
    }
    input_df = pd.DataFrame(data)

    # Chuyển đổi dữ liệu (ví dụ: one-hot encoding cho biến categorical)
    # ... (bạn cần thực hiện các bước tiền xử lý tương tự như khi huấn luyện mô hình)
    # ... (trong Streamlit app)
    # Xử lý giá trị mới (nếu có)
    for col in train_features:
        if col not in input_df.columns:
            input_df[col] = 0


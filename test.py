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
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Hàm tải dữ liệu
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    return data

# Hàm tiền xử lý dữ liệu
def preprocess_data(data):
    # Xóa các dòng có ít nhất 2 cột chứa giá trị null
    thresh_value = data.shape[1] - 1
    df_cleaned = data.dropna(thresh=thresh_value)
    
    # Xóa một số cột giá trị có thể gây ảnh hưởng đến quá trình huấn luyện model
    data_cleaned = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Điền dữ liệu tuổi null thành giá trị trung bình của tuổi
    data_cleaned['Age'] = data_cleaned['Age'].fillna(data_cleaned['Age'].median())
    
    # Điền dữ liệu Embarked null thành giá trị mode của Embarked
    data_cleaned['Embarked'] = data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].mode()[0])
    
    # Chuẩn hóa các cột về các giá trị để giúp cho quá trình huấn luyện
    data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex', 'Embarked'], drop_first=True)
    
    return data_cleaned

# Hàm kiểm tra lỗi dữ liệu
def check_error(data):
    # Kiểm tra giá trị thiếu
    missing_values = data.isnull().sum()
    
    # Kiểm tra dữ liệu trùng lặp
    duplicate_count = data.duplicated().sum()
    
    # Kiểm tra giá trị quá lớn (outlier) bằng Z-score
    outlier_count = {
        col: (abs(zscore(data[col], nan_policy='omit')) > 3).sum()
        for col in data.select_dtypes(include=['number']).columns
    }
    
    # Tạo báo cáo lỗi
    error_report = pd.DataFrame({
        'Cột': data.columns,
        'Giá trị thiếu': missing_values,
        'Outlier': [outlier_count.get(col, 0) for col in data.columns]
    })
    
    return error_report, duplicate_count

# Hàm chia tập dữ liệu
def split_data(data):
    train_ratio = st.slider("Tập huấn luyện", 0, 90, 70)
    a = 100 - train_ratio
    val_ratio = st.slider("Tập xác thực", 0, a, 5)
    test_ratio = 100 - train_ratio - val_ratio
    
    train_df, val_test_df = train_test_split(data, test_size=(100 - train_ratio) / 100, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=test_ratio / (100 - train_ratio), random_state=42)
    
    return train_df, val_df, test_df

# Hàm chọn mô hình
def choose_model():
    model_choice = st.radio("Chọn mô hình:", ["Multiple_Regression", "Polynomial_Regression"])
    
    if model_choice == "Multiple_Regression":
        fit_intercept = st.selectbox("Fit Intercept", [True, False])
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_choice == "Polynomial_Regression":
        degree = st.slider("Degree", 1, 10, 2)
        interaction_only = st.selectbox("Interaction Only", [True, False])
        model = make_pipeline(PolynomialFeatures(degree=degree, interaction_only=interaction_only), LinearRegression())
    
    return model

# Hàm huấn luyện mô hình
def train_model(model, train_df, val_df, test_df):
    model.fit(train_df.drop("Survived", axis=1), train_df["Survived"])
    st.success("✅ Huấn luyện thành công!")
    y_pred_train = model.predict(train_df.drop("Survived", axis=1))
    y_pred_val = model.predict(val_df.drop("Survived", axis=1))
    y_pred_test = model.predict(test_df.drop("Survived", axis=1))
    return y_pred_train, y_pred_val, y_pred_test

# Hàm dự đoán kết quả
def predict_result(model, input_df):
    prediction = model.predict(input_df)[0]
    
    if prediction > 0.5:
        prodiction = 1
        message = "Sống sót 😇"
    else:
        prodiction = 0
        message = "Không sống sót ☠️"
    
    return message

# Hàm tạo form nhập liệu
def create_input_form():
    with st.sidebar.form("input_form"):
        pclass = st.selectbox("Hạng Vé", [1, 2, 3])
        sex = st.selectbox("Giới Tính", ["male", "female"])
        age = st.number_input("Tuổi", min_value=0, max_value=100, value=25)
        sibsp = st.number_input("Anh Chị Em", min_value=0, value=0)
        parch = st.number_input("Bố Mẹ Con Cái", min_value=0, value=0)
        fare = st.number_input("Giá Vé", min_value=0, value=0)  # Đã sửa lỗi ở đây
        embarked = st.selectbox("Cảng", ["Southampton", "Cherbourg", "Queenstown"])

    
    return pclass, sex, age, sibsp, parch, fare, embarked

st.title("Ứng dụng Titanic với Streamlit")
st.write("""
## Phân tích dữ liệu và huấn luyện mô hình Multiple Rgresstion
""")

# Tải dữ liệu
data = load_data()

# Kiểm tra lỗi dữ liệu
error_report, duplicate_count = check_error(data)
st.subheader("Kiểm tra lỗi dữ liệu")
st.table(error_report)
st.write(f"**Số lượng dòng bị trùng lặp:** {duplicate_count}")

# Tiền xử lý dữ liệu
data_cleaned = preprocess_data(data)
st.subheader("Tiền xử lý dữ liệu")
st.write(data_cleaned)

# Chia tập dữ liệu
train_df, val_df, test_df = split_data(data_cleaned)
st.title("Chọn tỉ lệ của các tập dữ liệu")
st.write("Số lượng của các tập dữ liệu:")
st.write("Tập huấn luyện:", len(train_df))
st.write("Tập xác thực:", len(val_df))
st.write("Tập kiểm tra:", len(test_df))

# Chọn mô hình
model = choose_model()

# Huấn luyện mô hình
y_pred_train, y_pred_val, y_pred_test = train_model(model, train_df, val_df, test_df)

# Tạo form nhập liệu
pclass, sex, age, sibsp, parch, fare, embarked = create_input_form()

# Dự đoán kết quả
if st.form_submit_button.button("Dự Đoán"):
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
    })
    
    input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"], drop_first=True)
    input_df = input_df[train_df.drop("Survived", axis=1).columns]
    
    message = predict_result(model, input_df)
    st.sidebar.write(f"Kết quả: {message}")
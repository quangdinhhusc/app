
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




# Tiêu đề ứng dụng
st.title("Ứng dụng Titanic với Streamlit")

st.write("""
## Phân tích dữ liệu và huấn luyện mô hình Multiple Rgresstion
""")


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

st.subheader("Thay đổi dữ liệu")

# Tạo một phần upload dữ liệu
uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "xlsx", "xls"])

# Nếu người dùng chọn upload dữ liệu
if uploaded_file is not None:
    # Đọc dữ liệu từ file
    if uploaded_file.type == "text/csv":
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.type == "application/vnd.ms-excel":
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Loại file không được hỗ trợ")
        st.stop()

    # Hiển thị dữ liệu
    st.write("Dữ liệu đã được upload thành công!")

# Hiển thị dữ liệu gốc
st.subheader("Dữ liệu Titanic gốc")
st.write(data)

# Hiển thị bảng chứa số lượng dữ liệu bị thiếu hoặc null của các cột
st.subheader("Kiểm tra lỗi dữ liệu")

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
                # Hiển thị báo cáo lỗi
st.table(error_report)

                # Hiển thị số lượng dữ liệu trùng lặp
st.write(f"**Số lượng dòng bị trùng lặp:** {duplicate_count}")      
st.write(len(data))  


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
data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex', 'Embarked'], drop_first=True)

# Hiển thị dữ liệu sau khi tiền xử lý
st.write("Dữ liệu sau khi tiền xử lý:")
st.write(data_cleaned)

# Chia tập dữ liệu
# Tự chọn tỉ lệ của các tập dữ liệu
st.title("Chọn tỉ lệ của các tập dữ liệu")

train_ratio = st.slider("Tập huấn luyện", 0, 90, 70)
a = 100 - train_ratio
val_ratio = st.slider("Tập xác thực", 0, a, 5)

# Tính toán tỉ lệ của tập kiểm tra
test_ratio = 100 - train_ratio - val_ratio

# Chia dữ liệu
train_df, val_test_df = train_test_split(data, test_size=(100 - train_ratio) / 100, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=test_ratio / (100 - train_ratio), random_state=42)

total_data = len(data_cleaned)
train_size = int(total_data * train_ratio / 100)
val_size = int(total_data * val_ratio / 100)
test_size = total_data - train_size - val_size

st.write("Số lượng của các tập dữ liệu:")
st.write("Tập huấn luyện:", train_size)
st.write("Tập xác thực:", val_size)
st.write("Tập kiểm tra:", test_size)

# Chọn mô hình
model_choice = st.radio("Chọn mô hình:", ["Multiple_Regression", "Polynomial_Regression"])

if model_choice == "Multiple_Regression":
    st.markdown("""
    - **Multiple Regression** là thuật toán hồi quy tuyến tính đa biến, dự đoán giá trị của biến phụ thuộc dựa trên các biến độc lập.
    - **Tham số cần chọn:**
        - Fit Intercept (có hoặc không).  
    """)
        
    fit_intercept = st.selectbox("Fit Intercept", [True, False])
    model = LinearRegression(fit_intercept=fit_intercept)
    
elif model_choice == "Polynomial_Regression":
    st.markdown("""
    - **Polynomial Regression** là thuật toán hồi quy tuyến tính đa biến, dự đoán giá trị của biến phụ thuộc dựa trên các biến độc lập và các bậc của chúng.
    - **Tham số cần chọn:**  
        - Bậc của đa thức (degree).  
        - Interaction Only (có hoặc không).  
    """)
    degree = st.slider("Degree", 1, 10, 2)
    interaction_only = st.selectbox("Interaction Only", [True, False])
    model = make_pipeline(PolynomialFeatures(degree=degree, interaction_only=interaction_only), LinearRegression())



if st.button("Huấn luyện mô hình"):
    if model_choice == "Multiple_Regression":
        model.fit(train_df.drop("Survived", axis=1), train_df["Survived"])
        st.success("✅ Huấn luyện thành công!")
    elif model_choice == "Polynomial_Regression":
        model.fit(train_df.drop("Survived", axis=1), train_df["Survived"])
        st.success("✅ Huấn luyện thành công!")

# Đánh giá mô hình trên tập kiểm tra
y_pred_test = model.predict(test_df.drop("Survived", axis=1).values)
mse_test = mean_squared_error(test_df["Survived"].astype(int), y_pred_test)
r2_test = r2_score(test_df["Survived"].astype(int), y_pred_test)

# Đánh giá mô hình trên tập xác thực
y_pred = model.predict(val_df.drop("Survived", axis=1))
mse = mean_squared_error(val_df["Survived"], y_pred)
r2 = r2_score(val_df["Survived"], y_pred)

# Cross-validation
cv_scores = cross_val_score(model, train_df.drop("Survived", axis=1), train_df["Survived"], cv=5, scoring="neg_mean_squared_error")

# Hiển thị kết quả
st.write("Kết quả:")
st.write("MSE trên tập validation:", mse)
st.write("R2 trên tập validation:", r2)
st.write("MSE trên tập kiểm tra:", mse_test)
st.write("R2 trên tập kiểm tra:", r2_test)
st.write("Độ chính xác trung bình sau Cross-Validation:", -cv_scores.mean())

# Lấy tên đặc trưng huấn luyện (bạn cần đoạn code này khi train model)
train_features = train_df.drop("Survived", axis=1).columns.tolist()

# # ...existing code...
st.sidebar.title("Titanic Survival Prediction")

# Tạo form nhập liệu trong sidebar

with st.sidebar.form("input_form"):
    pclass = st.selectbox("Hạng Vé", [1, 2, 3])
    sex = st.selectbox("Giới Tính", ["male", "female"])
    age = st.number_input("Tuổi", min_value=0, max_value=100, value=25)
    sibsp = st.number_input("Anh Chị Em", min_value=0, value=0)
    parch = st.number_input("Bố Mẹ Con Cái", min_value=0, value=0)
    fare = st.number_input("Giá Vé", min_value=0, value=0)  # Đã sửa lỗi ở đây
    embarked = st.selectbox("Cảng", ["Southampton", "Cherbourg", "Queenstown"])
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

    input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"], drop_first=True) #one-hot encoding

    # Đảm bảo thứ tự cột giống như khi train
    input_df = input_df[train_features] # Sắp xếp theo thứ tự khi train

    # Dự đoán kết quả
    prediction = model.predict(input_df)[0]

    if prediction > 0.5:
        prodiction = 1
        message = "Sống sót 😇"
    else:
        prodiction = 0
        message = "Không sống sót ☠️"

    st.sidebar.write(f"Kết quả: {message}")
    # st.sidebar.write(f"Xác suất sống sót: {prediction}")
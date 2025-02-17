import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Tiêu đề ứng dụng
st.title("Ứng dụng Titanic với Streamlit")
st.write("""
## Phân tích dữ liệu và huấn luyện mô hình Random Forest
""")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Hiển thị dữ liệu gốc
st.subheader("Dữ liệu Titanic gốc")
st.write(data)

# Tiền xử lý dữ liệu
st.subheader("Tiền xử lý dữ liệu")

# Xóa các dòng có ít nhất 2 cột chứa giá trị null
thresh_value = data.shape[1] - 2
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

# Huấn luyện mô hình Random Forest
st.subheader("Huấn luyện mô hình Random Forest")
n_estimators = st.slider("Số lượng cây trong Random Forest (n_estimators)", 10, 200, 100)
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
st.write(f"Độ chính xác trung bình sau Cross-Validation: {cv_scores.mean():.2f}")

# Huấn luyện mô hình trên tập train
model.fit(X_train, y_train)

# Đánh giá mô hình trên tập validation
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
st.write(f"Độ chính xác trên tập Validation: {valid_accuracy:.2f}")

# Đánh giá mô hình trên tập test
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
st.write(f"Độ chính xác trên tập Test: {test_accuracy:.2f}")

# Hiển thị biểu đồ phân phối độ tuổi
st.subheader("Phân phối độ tuổi của hành khách")
fig, ax = plt.subplots()
sns.histplot(data['Age'].dropna(), kde=True, ax=ax)
st.pyplot(fig)

# Hiển thị biểu đồ tương quan giữa các đặc trưng
st.subheader("Ma trận tương quan giữa các đặc trưng")
corr = data_cleaned.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
# Start an MLflow run

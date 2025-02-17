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

# Tải dữ liệu Titanic
@st.cache_data  # Cache dữ liệu để tăng hiệu suất
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# Hiển thị dữ liệu gốc
st.subheader("Dữ liệu Titanic gốc")
st.write(data)

# Tiền xử lý dữ liệu
st.subheader("Tiền xử lý dữ liệu")
st.write("- Điền dữ liệu tuổi null thành giá trị trung bình của tuổi.")
import mlflow
import mlflow.sklearn
# Load the data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# xử lý dữ liệu
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# data['Age'].fillna(data['Age'].median(), inplace=True)
# data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Hiển thị dữ liệu sau khi tiền xử lý
st.write("Dữ liệu sau khi tiền xử lý:")
st.write(data)

# Chia tập dữ liệu
# đưa dữ liệu vào X và y
X = data.drop('Survived', axis=1)
y = data['Survived']
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
corr = data.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
# Start an MLflow run
esp = mlflow.set_experiment("Data Titanic")
with mlflow.start_run():
    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Log model parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Perform cross-validation on the training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
    mlflow.log_metric("cv_accuracy_std", cv_scores.std())
    
    # Train the model on the full training set
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_valid_pred = model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    mlflow.log_metric("valid_accuracy", valid_accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    print(f"Validation Accuracy: {valid_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
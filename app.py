import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
# Load the data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Tiền xử lý dữ liệu (ví dụ)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Chọn đặc trưng và target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Chia dữ l
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

with mlflow.start_run():
    # Khai báo các tham số
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("max_depth", 2, 20, 10)

    # Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    # Logging vào MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Lưu model
    mlflow.sklearn.log_model(model, "random-forest-model")

# Load model từ MLflow
model = mlflow.sklearn.load_model("random-forest-model")

# Dự đoán trên tập test
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Độ chính xác trên tập test: {accuracy}")

st.title("Kết quả huấn luyện mô hình Random Forest")

# Hiển thị các thông số và metric
st.write(f"n_estimators: {n_estimators}")
st.write(f"max_depth: {max_depth}")
st.write(f"Độ chính xác trên tập validation: {accuracy}")

# Vẽ biểu đồ (ví dụ)
st.bar_chart(pd.Series(y_pred).value_counts())

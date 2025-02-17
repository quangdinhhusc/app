import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
# Load the data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# xử lý dữ liệu
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# đưa dữ liệu vào X và y
X = data.drop('Survived', axis=1)
y = data['Survived']
# chia dữ liệu thành train 70, test 15, valid 15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
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

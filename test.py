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
import numpy as np # type: ignore

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Streamlit app
st.title("MNIST Classification with Streamlit & MLFlow")

# Sidebar for user input
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose Model", ["Decision Tree", "SVM"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
if st.sidebar.button("Train Model"):
    with mlflow.start_run():
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "SVM":
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log parameters and metrics to MLFlow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # Display results in Streamlit
        st.write(f"Model: {model_name}")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.write("Confusion Matrix:")
        st.write(cm)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        st.pyplot(fig)

        # Save model to MLFlow
        mlflow.sklearn.log_model(model, "model")


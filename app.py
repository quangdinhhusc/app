import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

st.title("Titanic Data Processing and Logging with MLflow")

# Load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Data cleaning (handling missing Age values)
new_data = data.dropna(subset=['Age'])

# Split data
x = new_data.drop('Survived', axis=1)
y = new_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

new_data_split = pd.DataFrame(list(new_data.items()), columns=['Data Set', 'Size'])

st.title("Titanic Data Processing and Logging with MLflow")
st.table(new_data_split)

fig = px.bar(new_data_split, x='Data Set', y='Size', title='Data Split', color='Data Set') # type: ignore
st.plotly_chart(fig)

st.write("Data Split")

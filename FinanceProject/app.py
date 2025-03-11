import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Financial Transactions: Fraud Detection App")

uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    feature_columns = ['amount', 'transaction_type', 'account_balance']  

    if all(col in df.columns for col in feature_columns):
        X = df[feature_columns]

        X = pd.get_dummies(X, drop_first=True)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, df["fraudulent"]) 

        predictions = model.predict(X)
        df["Fraud Prediction"] = predictions

        st.write("Predictions:", df[["amount", "transaction_type", "Fraud Prediction"]])
    else:
        st.error(f"The required columns {feature_columns} are missing from the uploaded file. Please check your dataset.")



import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Finance Fraud Detection App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a transaction dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    # Define expected feature columns (modify based on your dataset)
    feature_columns = ['amount', 'transaction_type', 'account_balance']  

    # Check if required columns exist
    if all(col in df.columns for col in feature_columns):
        X = df[feature_columns]

        # Convert categorical data to numbers (if necessary)
        X = pd.get_dummies(X, drop_first=True)

        # Train a simple RandomForest model (Modify if using a pre-trained model)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, df["fraudulent"])  # Assuming "fraudulent" is the target column

        # Make Predictions
        predictions = model.predict(X)
        df["Fraud Prediction"] = predictions

        # Show Results
        st.write("Predictions:", df[["amount", "transaction_type", "Fraud Prediction"]])
    else:
        st.error(f"The required columns {feature_columns} are missing from the uploaded file. Please check your dataset.")



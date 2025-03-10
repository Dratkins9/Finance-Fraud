<<<<<<< HEAD
import streamlit as st
import pandas as pd
import joblib  
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Finance Fraud Detection App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a transaction dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    # Updated feature column names to match your dataset
    feature_columns = ['Amount', 'Transaction_Type', 'User_ID']  # Adjust as needed

    # Check if required columns exist
    if all(col in df.columns for col in feature_columns):
        X = df[feature_columns]

        # Convert categorical data to numbers (for model training)
        X = pd.get_dummies(X, drop_first=True)

        # Load or Train Model (Modify as needed)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, df["Fraudulent"])  # Assuming "Fraudulent" is the target column

        # Make Predictions
        predictions = model.predict(X)
        df["Fraud Prediction"] = predictions

        # Show Results
        st.write("Predictions:", df[["Amount", "Transaction_Type", "Fraud Prediction"]])
    else:
        st.error(f"The required columns {feature_columns} are missing from the uploaded file. Please check your dataset.")


=======
import streamlit as st
import pandas as pd
import joblib  # If you're using a saved model
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Finance Fraud Detection App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a transaction dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    # Assume we need these features
    feature_columns = ['amount', 'transaction_type', 'account_balance']  # Modify as per your dataset

    # Check if required columns exist
    if all(col in df.columns for col in feature_columns):
        X = df[feature_columns]

        # Load or Train Model (Modify this part as needed)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, df["fraudulent"])  # Assuming "fraudulent" is the target column

        # Make Predictions
        predictions = model.predict(X)
        df["Fraud Prediction"] = predictions

        # Show Results
        st.write("Predictions:", df[["amount", "transaction_type", "Fraud Prediction"]])
    else:
        st.error("The required columns are missing from the uploaded file. Please check your dataset.")



>>>>>>> c0d0b27 (Saved changes before pulling updates)

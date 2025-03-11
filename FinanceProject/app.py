import streamlit as st
import streamlit_authenticator as stauth
import yaml
import pandas as pd
import numpy as np
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Authentication Configuration
with open("users.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name="auth_cookie",
    key="random_secret_key",
    cookie_expiry_days=30
)

# ✅ Login Page
def login():
    st.title("🔐 Login to Finance Fraud Detector")
    
    name, auth_status, username = authenticator.login()

    if auth_status:
        st.session_state["page"] = "main"
        st.session_state["username"] = username
        st.experimental_rerun()
    elif auth_status is False:
        st.error("Invalid username/password")
    elif auth_status is None:
        st.warning("Please enter your credentials")

# ✅ Main Fraud Detection Page
def main():
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"👋 Welcome, **{st.session_state['username']}**!")
    
    st.title("📊 Financial Fraud Detection System")
    
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Preview of Uploaded Data:")
        st.dataframe(df.head())

        if "fraudulent" in df.columns:
            st.write("### 🚀 Training Fraud Detection Model...")

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["year"] = df["timestamp"].dt.year
                df["month"] = df["timestamp"].dt.month
                df["day"] = df["timestamp"].dt.day
                df["hour"] = df["timestamp"].dt.hour
                df["minute"] = df["timestamp"].dt.minute
                df["second"] = df["timestamp"].dt.second
                df["day_of_week"] = df["timestamp"].dt.weekday
                df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
                df.drop(columns=["timestamp"], inplace=True)

            feature_columns = ["amount", "account_balance", "year", "month", "day", "hour", "minute", "second", "day_of_week", "is_weekend"]
            if "transaction_type" in df.columns:
                feature_columns.append("transaction_type")

            if all(col in df.columns for col in feature_columns):
                X = df[feature_columns].copy()
                if "transaction_type" in X.columns:
                    X = pd.get_dummies(X, columns=["transaction_type"], drop_first=True)

                y = df["fraudulent"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                classification_rep = classification_report(y_test, y_pred)

                st.write(f"### ✅ Model Accuracy: **{accuracy:.2f}**")
                st.text("### 📊 Classification Report:")
                st.text(classification_rep)
            else:
                missing_columns = [col for col in feature_columns if col not in df.columns]
                st.error(f"⚠ Missing columns: {missing_columns}. Please upload a valid dataset.")
    else:
        st.warning("📌 Please upload a CSV file.")

# ✅ Page Routing Logic
if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "main":
    main()

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---- Hash Passwords Correctly ----
hashed_passwords = stauth.Hasher(["password123", "userpass"]).generate()

# ---- Authentication Config ----
config = {
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'admin@example.com',
                'name': 'Admin',
                'password': hashed_passwords[0]
            },
            'user': {
                'email': 'user@example.com',
                'name': 'User',
                'password': hashed_passwords[1]
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'random_secret_key',
        'name': 'auth_cookie'
    }
}

# ---- Initialize Authenticator ----
authenticator = stauth.Authenticate(
    config['credentials'], 
    config['cookie']['name'], 
    config['cookie']['key'], 
    config['cookie']['expiry_days']
)

# ---- Login UI ----
name, authentication_status, username = authenticator.login(location="main")

if authentication_status:
    st.write(f"Welcome *{name}*!")
    authenticator.logout("Logout", "sidebar")

    # ---- File Upload Prompt (After Login) ----
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of first 5 entries:", df.head())

        # ---- Data Preprocessing ----
        df.columns = df.columns.str.strip().str.lower()
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

        # ---- Feature Selection ----
        feature_columns = ["amount", "transaction_type", "account_balance", "year", "month", "day", "hour", "minute", "second", "day_of_week", "is_weekend"]
        
        if all(col in df.columns for col in feature_columns):
            X = df[feature_columns].copy()
            X = pd.get_dummies(X, columns=["transaction_type"], drop_first=True)

            y = df["fraudulent"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # ---- Train Random Forest Model ----
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            st.write(f"Model training completed! Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_rep)

        else:
            missing_columns = [col for col in feature_columns if col not in df.columns]
            st.error(f"Missing columns: {missing_columns}. Please update your dataset.")

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
else:
    st.error("Authentication system error")



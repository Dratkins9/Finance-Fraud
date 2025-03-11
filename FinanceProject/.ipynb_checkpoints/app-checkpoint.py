import streamlit as st
import streamlit_authenticator as stauth
import yaml
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

hashed_passwords = stauth.Hasher(["password123", "userpass"]).generate()

config = {
    'credentials': {
        'usernames': {
            'admin': {'email': 'admin@example.com', 'name': 'Admin', 'password': hashed_passwords[0]},
            'user': {'email': 'user@example.com', 'name': 'User', 'password': hashed_passwords[1]}
        }
    },
    'cookie': {'expiry_days': 30, 'key': 'random_secret_key', 'name': 'auth_cookie'}
}

authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.sidebar.write(f"Welcome, *{name}*!")
    authenticator.logout("Logout", "sidebar")

    st.subheader("Upload your transaction CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df.columns = df.columns.str.strip().str.lower()

        st.write("Preview of first 5 rows:", df.head())

        feature_columns = ["amount", "transaction_type", "account_balance", "year", "month", "day", "hour", "minute", "second", "day_of_week", "is_weekend"]

        if all(col in df.columns for col in feature_columns):
            X = df[feature_columns].copy()
            X = pd.get_dummies(X, columns=["transaction_type"], drop_first=True) 

            y = df["fraudulent"] 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            st.success(f"Model training completed! Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_rep)

        else:
            missing_columns = [col for col in feature_columns if col not in df.columns]
            st.error(f"Missing columns in the dataset: {missing_columns}")

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")

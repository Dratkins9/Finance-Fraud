import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
import random  
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🔐 Hash passwords correctly
hashed_passwords = stauth.Hasher().hash(["password123", "userpass"])

# 🛠️ Config for authentication
config = {
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'admin@example.com',
                'name': 'Admin',
                'password': hashed_passwords[0]  # ✅ Use hashed password
            },
            'user': {
                'email': 'user@example.com',
                'name': 'User',
                'password': hashed_passwords[1]  # ✅ Use hashed password
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'random_secret_key',
        'name': 'auth_cookie'
    }
}

# 🔑 Set up authentication
authenticator = stauth.Authenticate(
    config['credentials'], 
    config['cookie']['name'], 
    config['cookie']['key'], 
    config['cookie']['expiry_days']
)

# 🔓 Login
login_result = authenticator.login(location="main")

if login_result is not None:
    name, authentication_status, username = login_result  

    if authentication_status:
        st.success(f"Welcome *{name}*!")
        authenticator.logout("Logout", "sidebar")

        # ✅ Now prompt user to upload a CSV file
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.write("✅ CSV successfully uploaded! Here's a preview:")
            st.write(df.head())

            # ✅ Ensure the correct feature columns exist
            feature_columns = ["amount", "transaction_type", "account_balance", "year", "month", "day", "hour", "minute", "second", "day_of_week", "is_weekend"]

            if all(col in df.columns for col in feature_columns):
                # 🔢 Process dataset
                X = df[feature_columns].copy()
                X = pd.get_dummies(X, columns=["transaction_type"], drop_first=True)

                y = df["fraudulent"]

                # 🏋️ Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # 🌲 Train Random Forest model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # 🧐 Predictions & accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                classification_rep = classification_report(y_test, y_pred)

                st.write(f"✅ Model training completed! Accuracy: {accuracy:.2f}")
                st.text("📊 Classification Report:")
                st.text(classification_rep)

            else:
                missing_columns = [col for col in feature_columns if col not 


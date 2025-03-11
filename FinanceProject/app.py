import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Initialize Faker for Synthetic Data
fake = Faker()

# ✅ Generate Fake Transaction Data
num_transactions = 100
data = []
for _ in range(num_transactions):
    transaction = {
        "transaction_id": fake.uuid4(),
        "user_id": fake.random_int(min=1000, max=2500),
        "amount": round(random.uniform(10, 5000), 2),
        "transaction_type": random.choice(["Online Transaction", "ATM Transaction", "Bank Transaction", "Bill Payment"]),
        "timestamp": fake.date_time(),
        "location": fake.city(),
        "account_balance": round(random.uniform(100, 20000), 2),
        "fraudulent": random.choice([0, 1])
    }
    data.append(transaction)

# ✅ Convert to DataFrame
df_synthetic = pd.DataFrame(data)

# ✅ Feature Engineering for Synthetic Data
df_synthetic["timestamp"] = pd.to_datetime(df_synthetic["timestamp"])
df_synthetic["year"] = df_synthetic["timestamp"].dt.year
df_synthetic["month"] = df_synthetic["timestamp"].dt.month
df_synthetic["day"] = df_synthetic["timestamp"].dt.day
df_synthetic["hour"] = df_synthetic["timestamp"].dt.hour
df_synthetic["minute"] = df_synthetic["timestamp"].dt.minute
df_synthetic["second"] = df_synthetic["timestamp"].dt.second
df_synthetic["day_of_week"] = df_synthetic["timestamp"].dt.weekday
df_synthetic["is_weekend"] = df_synthetic["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

df_synthetic.drop(columns=["timestamp"], inplace=True)

# ✅ Hasher Initialization
hasher = stauth.Hasher()
hashed_passwords = [hasher.hash("password123"), hasher.hash("userpass")]

# ✅ Authentication Config
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

# ✅ Initialize Authenticator
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

# ✅ Function to Handle Login
def login():
    """Login Screen"""
    st.write("### Welcome to the Login Screen")
    st.write("Please enter your username and password below.")

    authentication_status = authenticator.login()

    if authentication_status:
        st.session_state["page"] = "main"
        st.session_state["username"] = authenticator.username  
        st.experimental_rerun()
    elif authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")

# ✅ Function to Handle Main Application
def main():
    """Main Screen after Login"""
    st.sidebar.write(f"Welcome, *{st.session_state['username']}*!")  
    authenticator.logout("Logout", "sidebar")

    st.write("## Upload Your CSV File for Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # ✅ Load CSV or Use Synthetic Data
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
    else:
        df = df_synthetic
        st.write("### Using Synthetic Data (No File Uploaded)")

    st.write(df.head())

    # ✅ Feature Engineering (For CSV Data)
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

    # ✅ Fraud detection processing
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

        st.write(f"### Model Training Completed! Accuracy: {accuracy:.2f}")
        st.text("### Classification Report:")
        st.text(classification_rep)
    else:
        missing_columns = [col for col in feature_columns if col not in df.columns]
        st.error(f"Missing columns: {missing_columns}. Please upload a valid dataset.")

# ✅ Handle Page Navigation
if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "main":
    main()

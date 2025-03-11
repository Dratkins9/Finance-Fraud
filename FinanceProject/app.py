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

# ✅ Initialize Faker for Fake Data
fake = Faker()

# ✅ Create Fake Transaction Data
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

df = pd.DataFrame(data)
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
df.to_csv("fake_transactions.csv", index=False)

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

# ✅ Function to Force Page Refresh on Login
def rerun(data=None):
    """Forces Streamlit to refresh after login."""
    st.rerun()

# ✅ Function to Handle Login
def login():
    """Login Screen"""
    st.title("🔐 Login to Your Account")
    st.write("Please enter your username and password.")

    authentication_status = authenticator.login(callback=rerun)

    if authentication_status:
        st.session_state["page"] = "main"
        st.session_state["username"] = authenticator.username  
        st.experimental_rerun()  # ✅ Force UI refresh
    elif authentication_status is False:
        st.error("❌ Incorrect username or password!")
    elif authentication_status is None:
        st.warning("⚠️ Please enter your credentials.")

# ✅ Function to Handle Main Application
def main():
    """Main Screen after Login"""
    st.sidebar.title("📌 Menu")
    st.sidebar.write(f"👋 Welcome, **{st.session_state['username']}**!")
    authenticator.logout("🚪 Logout", "sidebar")

    st.title("📊 Upload Your CSV File for Analysis")
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Preview of Uploaded Data:")
        st.write(df.head())

        # ✅ Fraud detection processing
        if "fraudulent" in df.columns:
            st.write("### 🔎 Processing Fraud Detection...")

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

                st.write(f"### ✅ Model Training Completed! Accuracy: **{accuracy:.2f}**")
                st.text("### 📄 Classification Report:")
                st.text(classification_rep)
            else:
                missing_columns = [col for col in feature_columns if col not in df.columns]
                st.error(f"⚠️ Missing columns: {missing_columns}. Please upload a valid dataset.")
    else:
        st.warning("⚠️ Please upload a CSV file to proceed.")

# ✅ Handle Page Navigation
if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "main":
    main()

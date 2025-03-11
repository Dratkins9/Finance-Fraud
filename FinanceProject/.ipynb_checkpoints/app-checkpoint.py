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

# Initialize Faker for Fake Data
fake = Faker()

# Create Fake Transaction Data
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

# Hasher Initialization
hasher = stauth.Hasher(['password123', 'userpass'])  # Hash passwords in bulk
hashed_passwords = hasher.hashes

# Authentication Config (Updated)
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
    },
    'preauthorized': {
        'emails': []  # Empty list instead of None
    }
}

# Debug: Print the config to verify
st.write("Debug: Config =", config)

# Initialize Authenticator
try:
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    st.write("Debug: Authenticator initialized successfully")
except Exception as e:
    st.error(f"❌ Error initializing authenticator: {e}")

# Login Function (Fixed)
def login():
    st.title("🔐 Login to Your Account")
    st.write("Please enter your username and password.")
    
    # Render login form (for streamlit-authenticator >= 0.3.0)
    login_result = authenticator.login(
        fields={
            'Form name': 'Login',
            'Username': 'Username',
            'Password': 'Password',
            'Login': 'Login'
        }
    )
    
    # Debug: Check the return value of authenticator.login()
    st.write("Debug: login_result =", login_result)
    
    if login_result is None:
        st.error("❌ Login form failed to render. Please check your streamlit-authenticator version and configuration.")
        return
    
    # Unpack the result
    name, authentication_status, username = login_result
    
    if authentication_status:
        # Successful login
        st.session_state["authentication_status"] = True
        st.session_state["username"] = username
        st.session_state["page"] = "main"
        st.rerun()  # Rerun to switch to main page
    elif authentication_status is False:
        st.error("❌ Incorrect username or password!")
    elif authentication_status is None:
        st.warning("⚠️ Please enter your credentials.")

# Main Application Function
def main():
    st.sidebar.title("📌 Menu")
    st.sidebar.write(f"👋 Welcome, **{st.session_state['username']}**!")
    authenticator.logout("🚪 Logout", "sidebar")

    st.title("📊 Upload Your CSV File for Analysis")
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Preview of Uploaded Data:")
        st.write(df.head())

        # Fraud detection processing
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

# Page Navigation Logic
if "page" not in st.session_state:
    st.session_state["page"] = "login"
    st.session_state["authentication_status"] = False

if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "main" and st.session_state["authentication_status"]:
    main()
else:
    st.error("Session expired or invalid. Please log in again.")
    st.session_state["page"] = "login"
    st.rerun()
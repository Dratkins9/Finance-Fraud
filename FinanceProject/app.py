import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Authentication Config
with open("users.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# ✅ Initialize Authenticator
authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"]
)

# ✅ Ensure session state variables exist
if "auth_status" not in st.session_state:
    st.session_state["auth_status"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# ✅ Function to Transition to Main Page After Login
def rerun():
    st.session_state["page"] = "main"
    st.experimental_rerun()

# ✅ Login Function
def login():
    """Login Screen"""
    st.title("🔒 Login")
    name, auth_status, username = authenticator.login()

    if auth_status:
        st.session_state["auth_status"] = True
        st.session_state["username"] = username
        rerun()  # ✅ Redirect to Main Page
    elif auth_status is False:
        st.error("Username or password is incorrect")
    elif auth_status is None:
        st.warning("Please enter your username and password")

# ✅ Main Application
def main():
    """Main App - Fraud Detection"""
    st.sidebar.title(f"Welcome, {st.session_state['username']}! 🎉")
    authenticator.logout("Logout", "sidebar")

    st.title("📊 Fraud Detection - Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data:")
        st.write(df.head())

        # ✅ Fraud Detection Processing
        if "fraudulent" in df.columns:
            st.write("### Running Fraud Detection...")
            feature_columns = ["amount", "account_balance"]

            if all(col in df.columns for col in feature_columns):
                X = df[feature_columns].copy()
                y = df["fraudulent"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                classification_rep = classification_report(y_test, y_pred)

                st.write(f"### Model Accuracy: {accuracy:.2f}")
                st.text("### Classification Report:")
                st.text(classification_rep)
            else:
                missing_columns = [col for col in feature_columns if col not in df.columns]
                st.error(f"Missing columns: {missing_columns}. Please upload a valid dataset.")
    else:
        st.warning("Please upload a CSV file to proceed.")

# ✅ Page Routing - Ensure Redirect Works
if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "main" and st.session_state["auth_status"]:
    main()
else:
    st.session_state["page"] = "login"
    st.experimental_rerun()

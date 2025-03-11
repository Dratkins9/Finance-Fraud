import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

fake = Faker()
num_transactions = 100
data = []

for _ in range(num_transactions):
    data.append({
        "transaction_id": fake.uuid4(),
        "user_id": fake.random_int(min=1000, max=2500),
        "amount": round(random.uniform(10, 5000), 2),
        "transaction_type": random.choice(["Online Transaction", "ATM Transaction", "Bank Transaction", "Bill Payment"]),
        "timestamp": fake.date_time(),
        "location": fake.city(),
        "account_balance": round(random.uniform(100, 20000), 2),
        "fraudulent": random.choice([0, 1])
    })

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

df = pd.read_csv("fake_transactions.csv")
df.columns = df.columns.str.strip().str.lower()

st.title("Finance Fraud Detection")
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(df.head())

    if "fraudulent" in df.columns:
        st.write("Processing Fraud Detection...")

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

            st.write(f"Model Training Completed! Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            X_test["Predicted Fraud"] = y_pred
            X_test["Actual Fraud"] = y_test.values
            X_test.reset_index(drop=True, inplace=True)

            st.write("Fraud Prediction Results Per Transaction:")
            st.dataframe(X_test)

            fig, ax = plt.subplots()
            sns.countplot(x=y_pred, palette="coolwarm", ax=ax)
            ax.set_xticklabels(["Not Fraud", "Fraud"])
            ax.set_title("Fraud vs. Non-Fraud Transactions")
            st.pyplot(fig)
        else:
            missing_columns = [col for col in feature_columns if col not in df.columns]
            st.error(f"Missing columns: {missing_columns}. Please upload a valid dataset.")
else:
    st.warning("Please upload a CSV file to proceed.")

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

hashed_passwords = [stauth.Hasher().hash(pwd) for pwd in ["password123", "userpass"]]

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

authenticator = stauth.Authenticate(
    config['credentials'], 
    config['cookie']['name'], 
    config['cookie']['key'], 
    config['cookie']['expiry_days']
)

login_result = authenticator.login(location="main")

if login_result is not None:
    name, authentication_status, username = login_result  

    if authentication_status:
        st.write(f"Welcome *{name}*!")
        authenticator.logout("Logout", "sidebar")
    elif authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")
else:
    st.error("Authentication system error")

st.write(st.session_state)

import numpy as numpy
import streamlit as streamlit 
import pandas as pandas
import random  
import matplotlib.pyplot as plt
import seaborn as seaborn
from faker import Faker  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

fake = Faker()

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

df = pandas.DataFrame(data)

df["timestamp"] = pandas.to_datetime(df["timestamp"])


df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour  
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second
df["day"] = df["timestamp"].dt.weekday 
df["weekend"] = df["day"].apply(lambda x: 1 if x >= 5 else 0) 

df.drop(columns=["timestamp"], inplace=True)

df.to_csv("fake_transactions.csv", index=False)

df = pandas.read_csv("fake_transactions.csv")

df.columns = df.columns.str.strip().str.lower()

streamlit.write("Preview of first 5 entries:", df.head())

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

    st.write(f"Model training completed! Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_rep)

else:
    missing_columns = [col for col in feature_columns if col not in df.columns]
    st.error(f"There are columns {missing_columns} missing! The dataset needs updated.")



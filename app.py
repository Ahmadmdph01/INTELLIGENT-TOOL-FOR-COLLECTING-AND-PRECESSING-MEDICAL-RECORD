import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sidebar navigation
rad = st.sidebar.radio("Navigation Menu", ["Home", "Diabetes Section", "Heart Disease Section", "Plots", "User Information"])

# Home Page
if rad == "Home":
    st.title("Intelligent Tool for Medical Data Collection and Processing")
    st.text("Project By Ahmad Ibrahim:")
    st.text("Supervised By Mallam shehu isa hussaini and mallam Ahmad jajere:")
    st.text("The Following Diseases are Predicted based on Medical information provided->")
    st.text("1. Diabetes Predictions")
    st.text("2. Heart Disease Predictions")

# Diabetes Prediction
df2 = pd.read_csv("Diabetes Predictions.csv")
x2 = df2.iloc[:, [1, 4, 5, 7]].values
y2 = df2.iloc[:, [-1]].values
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=0)
model2 = RandomForestClassifier()
model2.fit(x2_train, y2_train)

# Diabetes Page
if rad == "Diabetes Section":
    st.header("Measure risk of Diabetes for Patient")
    st.write("All The Values Should Be In Range Mentioned")
    name = st.text_input("Enter Your Name")
    address = st.text_input("Enter Your Address")
    glucose = st.number_input("Enter Your Glucose Level (0-200)", min_value=0, max_value=200, step=1)
    insulin = st.number_input("Enter Your Insulin Level In Body (0-850)", min_value=0, max_value=850, step=1)
    bmi = st.number_input("Enter Your Body Mass Index/BMI Value (0-70)", min_value=0, max_value=70, step=1)
    age = st.number_input("Enter Your Age (20-80)", min_value=20, max_value=80, step=1)
    prediction2 = model2.predict([[glucose, insulin, bmi, age]])[0]

    if st.button("Predict"):
        if prediction2 == 1:
            st.warning("You Might Be Affected By Diabetes")
        elif prediction2 == 0:
            st.success("You Are Safe")

# Heart Disease Prediction
df3 = pd.read_csv("Heart Disease Predictions.csv")
x3 = df3.iloc[:, [2, 3, 4, 7]].values
y3 = df3.iloc[:, [-1]].values
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=0)
model3 = RandomForestClassifier()
model3.fit(x3_train, y3_train)

# Heart Disease Page
if rad == "Heart Disease Section":
    name = st.text_input("Enter Your Name")
    address = st.text_input("Enter Your Address")
    st.header("Measure risk of Heart Disease for Patient")
    st.write("All The Values Should Be In Range Mentioned")
    chestpain = st.number_input("Rate Your Chest Pain (1-4)", min_value=1, max_value=4, step=1)
    bp = st.number_input("Enter Your Blood Pressure Rate (95-200)", min_value=95, max_value=200, step=1)
    cholesterol = st.number_input("Enter Your Cholesterol Level Value (125-565)", min_value=125, max_value=565, step=1)
    maxhr = st.number_input("Enter Your Maximum Heart Rate (70-200)", min_value=70, max_value=200, step=1)
    prediction3 = model3.predict([[chestpain, bp, cholesterol, maxhr]])[0]

    if st.button("Predict"):
        if prediction3 == 1:
            st.warning("You Might Be Affected By Heart Disease")
        elif prediction3 == 0:
            st.success("You Are Safe")

# Plots Page
if rad == "Plots":
    type = st.selectbox("Which Plot Do You Want To See?", ["Covid-19", "Diabetes", "Heart Disease"])
    if type == "Covid-19":
        fig = px.scatter(df1, x="Difficulty in breathing", y="Infected with Covid19")
        st.plotly_chart(fig)
    elif type == "Diabetes":
        fig = px.scatter(df2, x="Glucose", y="Outcome")
        st.plotly_chart(fig)
    elif type == "Heart Disease":
        fig = px.scatter(df3, x="BP", y="Heart Disease")
        st.plotly_chart(fig)
# User Information Page
if rad == "User Information":
    st.header("User Information Input")
    user_name = st.text_input("Enter Your Name")
    user_address = st.text_input("Enter Your Address")
    
    if st.button("Submit"):
        if user_name and user_address:
            st.success("Information Submitted Successfully!")
            user_data = {'Name': [user_name], 'Address': [user_address]}
            user_df = pd.DataFrame(user_data)
            st.table(user_df)
        else:
            st.error("Please enter both name and address.")


# Add login page and display user input in a table
if rad == "Login":
    username = st.text_input("Admin")
    password = st.text_input("Password", type='12345')
    if st.button("Login"):
        if username and password:
            st.success("Logged in successfully!")
            # Display user input in a table
            user_data = {'Username': [username], 'Password': [password]}
            user_df = pd.DataFrame(user_data)
            st.table(user_df)
        else:
            st.error("Please enter both username and password.")

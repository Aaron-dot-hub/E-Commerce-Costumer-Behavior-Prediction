import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model_path = "best_xgb_model.pkl"
model_path_1 = "pca_model.pkl"
# Load trained XGBoost model and PCA model
xgb_classifier = joblib.load(model_path)
pca_optimal = joblib.load(model_path_1)  # Load trained PCA model

# Streamlit app title
st.title("E-Commerce Customer Behavior Prediction")

# Sidebar input fields
st.sidebar.header("Enter Customer Details")

# User Inputs
tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=61, value=0)
PreferredLoginDevice = st.sidebar.selectbox("PreferredLoginDevice", ['Mobile Phone', 'Phone', 'Computer'])
CityTier = st.sidebar.number_input("CityTier", min_value=1, max_value=3, value=1)
WarehouseToHome = st.sidebar.number_input("WarehouseToHome", min_value=5, max_value=127, value=5)
PreferredPaymentMode = st.sidebar.selectbox("PreferredPaymentMode", 
                                            ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card'])
Gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
HourSpendOnApp = st.sidebar.number_input("HourSpendOnApp", min_value=0, max_value=5, value=0)
NumberOfDeviceRegistered = st.sidebar.number_input("NumberOfDeviceRegistered", min_value=1, max_value=6, value=1)
PreferedOrderCat = st.sidebar.selectbox('PreferedOrderCat', ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
SatisfactionScore = st.sidebar.number_input("SatisfactionScore", min_value=1, max_value=5, value=1)
MaritalStatus = st.sidebar.selectbox('MaritalStatus', ['Single', 'Divorced', 'Married'])
NumberOfAddress = st.sidebar.number_input("NumberOfAddress", min_value=1, max_value=22, value=1)
Complain = st.sidebar.number_input("Complain", min_value=0, max_value=1, value=0)
OrderAmountHikeFromlastYear = st.sidebar.number_input("OrderAmountHikeFromlastYear", min_value=11, max_value=26, value=11)
CouponUsed = st.sidebar.number_input("CouponUsed", min_value=0, max_value=16, value=0)
OrderCount = st.sidebar.number_input("OrderCount", min_value=1, max_value=16, value=1)
DaySinceLastOrder = st.sidebar.number_input("DaySinceLastOrder", min_value=0, max_value=46, value=0)
CashbackAmount = st.sidebar.number_input("CashbackAmount", min_value=0, max_value=325, value=0)

# Define categorical encoding
categorical_columns = {
    'PreferredLoginDevice': ['Mobile Phone', 'Phone', 'Computer'],
    'PreferredPaymentMode': ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card'],
    'Gender': ['Female', 'Male'],
    'PreferedOrderCat': ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'],
    'MaritalStatus': ['Single', 'Divorced', 'Married']
}

# Encode categorical inputs
label_encoders = {col: LabelEncoder() for col in categorical_columns}
encoded_inputs = {}

for col, categories in categorical_columns.items():
    label_encoders[col].fit(categories)
    encoded_inputs[col] = label_encoders[col].transform([locals()[col]])[0]  # Encode user selection

# Create DataFrame for prediction (Ensure column order matches training data)
input_data = pd.DataFrame([[
    tenure, encoded_inputs['PreferredLoginDevice'], CityTier, WarehouseToHome, encoded_inputs['PreferredPaymentMode'], 
    encoded_inputs['Gender'], HourSpendOnApp, NumberOfDeviceRegistered, encoded_inputs['PreferedOrderCat'], 
    SatisfactionScore, encoded_inputs['MaritalStatus'], NumberOfAddress, Complain, 
    OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
]], columns=[
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 
    'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 
    'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 
    'DaySinceLastOrder', 'CashbackAmount'
])

# Transform input data using trained PCA model
input_data_pca = pca_optimal.transform(input_data)

# Predict Button
if st.sidebar.button("Predict Purchase"):
    prediction = xgb_classifier.predict(input_data_pca)[0]
    result = "This Customer Will Not Churn" if prediction == 1 else "This Customer Will Churn"
    st.write(f"### Prediction: {result}")

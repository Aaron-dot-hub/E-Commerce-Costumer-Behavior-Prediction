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

# Title
st.title("E-Commerce Customer Behavior Prediction")

st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select how to provide input:", ("Manual Entry", "Upload Test Dataset"))

# Define categorical encoding
categorical_columns = {
    'PreferredLoginDevice': ['Mobile Phone', 'Phone', 'Computer'],
    'PreferredPaymentMode': ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card'],
    'Gender': ['Female', 'Male'],
    'PreferedOrderCat': ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'],
    'MaritalStatus': ['Single', 'Divorced', 'Married']
}
label_encoders = {col: LabelEncoder().fit(categories) for col, categories in categorical_columns.items()}

# Prediction function
def make_prediction(df_row):
    df_row_pca = pca_optimal.transform(df_row)
    pred = xgb_classifier.predict(df_row_pca)[0]
    prob = xgb_classifier.predict_proba(df_row_pca)[0][1]  # probability for class 1
    result = "Likely to Purchase" if pred == 1 else "Not Likely to Purchase"
    return result, prob

# Manual Entry
if input_method == "Manual Entry":
    st.sidebar.subheader("Enter Customer Details")

    # Inputs
    tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=61, value=0)
    PreferredLoginDevice = st.sidebar.selectbox("PreferredLoginDevice", categorical_columns['PreferredLoginDevice'])
    CityTier = st.sidebar.number_input("CityTier", min_value=1, max_value=3, value=1)
    WarehouseToHome = st.sidebar.number_input("WarehouseToHome", min_value=5, max_value=127, value=5)
    PreferredPaymentMode = st.sidebar.selectbox("PreferredPaymentMode", categorical_columns['PreferredPaymentMode'])
    Gender = st.sidebar.selectbox("Gender", categorical_columns['Gender'])
    HourSpendOnApp = st.sidebar.number_input("HourSpendOnApp", min_value=0, max_value=5, value=0)
    NumberOfDeviceRegistered = st.sidebar.number_input("NumberOfDeviceRegistered", min_value=1, max_value=6, value=1)
    PreferedOrderCat = st.sidebar.selectbox('PreferedOrderCat', categorical_columns['PreferedOrderCat'])
    SatisfactionScore = st.sidebar.number_input("SatisfactionScore", min_value=1, max_value=5, value=1)
    MaritalStatus = st.sidebar.selectbox('MaritalStatus', categorical_columns['MaritalStatus'])
    NumberOfAddress = st.sidebar.number_input("NumberOfAddress", min_value=1, max_value=22, value=1)
    Complain = st.sidebar.number_input("Complain", min_value=0, max_value=1, value=0)
    OrderAmountHikeFromlastYear = st.sidebar.number_input("OrderAmountHikeFromlastYear", min_value=11, max_value=26, value=11)
    CouponUsed = st.sidebar.number_input("CouponUsed", min_value=0, max_value=16, value=0)
    OrderCount = st.sidebar.number_input("OrderCount", min_value=1, max_value=16, value=1)
    DaySinceLastOrder = st.sidebar.number_input("DaySinceLastOrder", min_value=0, max_value=46, value=0)
    CashbackAmount = st.sidebar.number_input("CashbackAmount", min_value=0, max_value=325, value=0)

    # Encode
    encoded_inputs = {
        'PreferredLoginDevice': label_encoders['PreferredLoginDevice'].transform([PreferredLoginDevice])[0],
        'PreferredPaymentMode': label_encoders['PreferredPaymentMode'].transform([PreferredPaymentMode])[0],
        'Gender': label_encoders['Gender'].transform([Gender])[0],
        'PreferedOrderCat': label_encoders['PreferedOrderCat'].transform([PreferedOrderCat])[0],
        'MaritalStatus': label_encoders['MaritalStatus'].transform([MaritalStatus])[0]
    }

    input_data = pd.DataFrame([[
        tenure, encoded_inputs['PreferredLoginDevice'], CityTier, WarehouseToHome, encoded_inputs['PreferredPaymentMode'], 
        encoded_inputs['Gender'], HourSpendOnApp, NumberOfDeviceRegistered, encoded_inputs['PreferedOrderCat'], 
        SatisfactionScore, encoded_inputs['MaritalStatus'], NumberOfAddress, Complain, 
        OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
    ]], columns=[
        'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 
        'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 
        'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 
        'DaySinceLastOrder', 'CashbackAmount']
    )

    if st.sidebar.button("Predict Purchase"):
        result, prob = make_prediction(input_data)
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {prob:.2f}")

# Upload CSV option
else:
    uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(test_df)

        index = st.sidebar.number_input("Select Row Index to Predict", min_value=0, max_value=len(test_df)-1, value=0)

        input_row = test_df.iloc[[index]].copy()

        # Encode categorical columns if present
        for col in categorical_columns:
            if col in input_row.columns:
                input_row[col] = label_encoders[col].transform(input_row[col])

        if st.sidebar.button("Predict from Uploaded Data"):
            result, prob = make_prediction(input_row)
            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {prob:.2f}")

    else:
        st.warning("Please upload a CSV file to proceed.")

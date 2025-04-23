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
    result = "Churned " if pred == 1 else "Not Churned"
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
        

# Upload CSV option
else:
    uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])

    if uploaded_file is not None:
        original_df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.markdown(f"**Original Data Shape:** {original_df.shape[0]} rows × {original_df.shape[1]} columns")
        st.dataframe(original_df)


        # Make a copy for encoding
        test_df = original_df.copy()

        # Encode categorical columns
        for col in categorical_columns:
            if col in test_df.columns:
                test_df[col] = label_encoders[col].transform(test_df[col])

        # Predict all customers
        if st.sidebar.button("Predict All Customers"):
            test_df_pca = pca_optimal.transform(test_df)
            predictions = xgb_classifier.predict(test_df_pca)
            probabilities = xgb_classifier.predict_proba(test_df_pca)[:, 1]

            result_df = original_df.copy()
            result_df['Prediction'] = np.where(predictions == 1, 'Churned', 'Not Churned')
            result_df['Confidence'] = probabilities

            churned = result_df[result_df['Prediction'] == 'Churned']
            not_churned = result_df[result_df['Prediction'] == 'Not Churned']

           
            
            # Display shapes and original columns only
            st.subheader("Churned Customers")
            st.markdown(f"**Shape:** {churned.shape[0]} rows × {original_df.shape[1]} columns")
            st.dataframe(churned[original_df.columns])

            
            
            st.subheader("Non-Churned Customers")
            st.markdown(f"**Shape:** {not_churned.shape[0]} rows × {original_df.shape[1]} columns")
            st.dataframe(not_churned[original_df.columns])

            # Charts
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.subheader("Churn Distribution")

            fig1, ax1 = plt.subplots()
            counts = [len(churned), len(not_churned)]
            labels = ['Churned', 'Not Churned']
            colors = ['red', 'green']
            ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.axis('equal')
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.barplot(x=labels, y=counts, palette=colors, ax=ax2)
            ax2.set_ylabel("Number of Customers")
            ax2.set_title("Churn vs Non-Churned Customers")
            st.pyplot(fig2)

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

        # Predict a selected row
        st.subheader("Predict for a Specific Row")

        selected_index = st.number_input("Select Row Index", min_value=0, max_value=len(test_df)-1, step=1)
        selected_row_encoded = test_df.iloc[[selected_index]].copy()
        selected_row_original = original_df.iloc[[selected_index]].copy()

        if st.button("Predict Selected Row"):
            single_row_pca = pca_optimal.transform(selected_row_encoded)
            single_pred = xgb_classifier.predict(single_row_pca)[0]
            single_prob = xgb_classifier.predict_proba(single_row_pca)[0][1]

            result_label = "Churned" if single_pred == 1 else "Not Churned"
            st.write("#### Selected Row Details")
            st.dataframe(selected_row_original)
            st.success(f"Prediction: {result_label}")
            #st.info(f"Confidence: {single_prob:.2f}")

    else:
        st.warning("Please upload a CSV file to proceed.")


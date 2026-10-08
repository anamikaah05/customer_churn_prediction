import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("Customer Churn Predictor")
st.write("Enter the key customer details below for an instant churn risk evaluation.")

# 2. Load Model, Scaler, and Columns
@st.cache_resource
def load_artifacts():
    model = joblib.load('churnprediction_model.pkl')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Make sure 'churn_model1.pkl', 'scaler.pkl', and 'model_columns.pkl' are in the folder.")
    st.stop()

# 3. Streamlined User Input Form (Only essential fields)
with st.form("simple_churn_form"):
    st.subheader("Key Customer Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    submit_button = st.form_submit_button(label="Predict Churn Risk")

# 4. Processing and Prediction Logic
if submit_button:
    # Build complete input dataframe using safe baseline defaults for the fields we hid
    input_data = {
        'gender': ["Male"],
        'SeniorCitizen': [0],
        'Partner': ["Yes"],
        'Dependents': ["No"],
        'tenure': [tenure],
        'PhoneService': ["Yes"],
        'MultipleLines': ["No"],
        'InternetService': [internet_service],
        'OnlineSecurity': ["No"],
        'OnlineBackup': ["No"],
        'DeviceProtection': ["No"],
        'TechSupport': ["No"],
        'StreamingTV': ["No"],
        'StreamingMovies': ["No"],
        'Contract': [contract],
        'PaperlessBilling': ["Yes"],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # One-hot encode matching training format
    df_input_encoded = pd.get_dummies(df_input)
    
    # Reindex columns to match exact model structure
    df_input_encoded = df_input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Scale continuous features
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_input_encoded[continuous_cols] = scaler.transform(df_input_encoded[continuous_cols])
    
    # Prediction
    prediction_proba = model.predict_proba(df_input_encoded)
    churn_probability = prediction_proba[0][1] * 100
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if churn_probability >= 50.0:
        st.error(f"⚠️ **High Churn Risk** (Probability: {churn_probability:.2f}%)")
    else:
        st.success(f"✅ **Low Churn Risk** (Probability: {churn_probability:.2f}%)")
        
    st.progress(int(churn_probability))
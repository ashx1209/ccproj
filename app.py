import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide"
)

# --- Session State Initialization ---
# This keeps track of how many months to show, defaulting to 3.
if 'month_count' not in st.session_state:
    st.session_state.month_count = 3

def add_month():
    if st.session_state.month_count < 6:
        st.session_state.month_count += 1

def remove_month():
    if st.session_state.month_count > 1:
        st.session_state.month_count -= 1

# Cache resource loading
@st.cache_resource
def load_artifacts():
    expected_cols = joblib.load('models/expected_columns.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    model = joblib.load('models/final_tuned_svm_model.pkl')
    return expected_cols, scaler, model

expected_columns, scaler, model = load_artifacts()

# Preprocessing & Inference Function
def preprocess_and_predict(raw_data, model, scaler, expected_columns):
    df = pd.DataFrame([raw_data])
    
    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    if 'MARRIAGE' in df.columns:
        df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
        
    encoded_df = pd.get_dummies(df)
    aligned_df = encoded_df.reindex(columns=expected_columns, fill_value=0)
    scaled_array = scaler.transform(aligned_df)
    
    prediction = model.predict(scaled_array)
    return prediction[0]

# --- UI Header ---
st.title("💳 Credit Card Default Risk Evaluator")
st.markdown("Enter customer details below to evaluate the probability of payment default.")

st.divider()

# --- Dynamic History Buttons ---
st.markdown("### 🗓️ Financial History Tracker")
st.write(f"Currently entering data for **{st.session_state.month_count} month(s)**. (Maximum: 6)")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    st.button("➕ Add a Month", on_click=add_month, disabled=(st.session_state.month_count >= 6), use_container_width=True)
with col_btn2:
    st.button("➖ Remove a Month", on_click=remove_month, disabled=(st.session_state.month_count <= 1), use_container_width=True)

st.divider()

# --- User Inputs Form ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographic Information")
    limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, max_value=1000000, value=50000, step=5000)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education Level", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {
        1: "Graduate School", 2: "University", 3: "High School", 4: "Others", 5: "Unknown", 6: "Unknown"
    }[x])
    marriage = st.selectbox("Marital Status", options=[1, 2, 3, 0], format_func=lambda x: {
        1: "Married", 2: "Single", 3: "Others", 0: "Others"
    }[x])
    age = st.slider("Age", min_value=18, max_value=100, value=30)

with col2:
    st.subheader("⏱ Recent Repayment Status")
    st.markdown("**0** = Paid minimum required bill on time  \n**-1** = Paid full bill on time  \n**-2** = Paid full bill 10 days before deadline  \n**1, 2, 3...** = Months delayed")
    
    pay_inputs = {}
    for i in range(1, st.session_state.month_count + 1):
        label = "Most Recent" if i == 1 else f"Month {i}"
        pay_inputs[f'PAY_{i}'] = st.number_input(f"PAY_{i} ({label})", min_value=-2, max_value=8, value=0)

with col3:
    st.subheader("💵 Financial History")
    
    bill_inputs = {}
    pay_amt_inputs = {}
    
    for i in range(1, st.session_state.month_count + 1):
        label = "Most Recent" if i == 1 else f"Month {i}"
        
        with st.expander(f"Financials: {label}", expanded=(i <= 2)):
            bill_inputs[f'BILL_AMT{i}'] = st.number_input(f"Bill Amount (BILL_AMT{i})", value=15000, key=f"bill_{i}")
            pay_amt_inputs[f'PAY_AMT{i}'] = st.number_input(f"Paid Amount (PAY_AMT{i})", value=2000, key=f"payamt_{i}")

st.divider()

# --- Prediction Action ---
if st.button("🔍 Assess Default Risk", type="primary", use_container_width=True):
    
    input_data = {
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        **pay_inputs,
        **bill_inputs,
        **pay_amt_inputs
    }
    
    with st.spinner("Processing customer profile through SVM pipeline..."):
        result = preprocess_and_predict(input_data, model, scaler, expected_columns)
        
    st.subheader("Assessment Result")
    if result == 1:
        st.error("🚨 **HIGH RISK**: The model predicts a high likelihood of default next month.")
        st.warning("Recommended Action: Flag account for manual review or limit extension freeze.")
    else:
        st.success("✅ **LOW RISK**: The customer profile indicates safe repayment behavior.")
        st.info("Recommended Action: Standard account processing approved.")
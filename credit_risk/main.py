import streamlit as st
import numpy as np
import joblib

# 🔄 Load model and preprocessors
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")
    return model, scaler, imputer

model, scaler, imputer = load_artifacts()

# 📌 Title
st.title("🧠 Credit Risk Prediction")
st.write("Enter applicant details to assess loan default risk.")

# ➗ Two column layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    revolving_util = st.slider("Revolving Utilization", 0.0, 1.0, 0.3)
    debt_ratio = st.slider("Debt Ratio", 0.0, 3.0, 0.8)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
    num_open_credit = st.slider("Open Credit Lines", 0, 50, 5)

with col2:
    num_dependents = st.slider("Dependents", 0, 20, 1)
    late_30_59 = st.slider("30-59 Days Late", 0, 100, 1)
    late_60_89 = st.slider("60-89 Days Late", 0, 100, 0)
    late_90 = st.slider("90+ Days Late", 0, 100, 0)
    real_estate_loans = st.slider("Real Estate Loans", 0, 10, 1)

# ✨ Predict button
if st.button("Predict Risk"):
    input_data = np.array([[revolving_util, age, late_30_59, debt_ratio, monthly_income,
                            num_open_credit, late_90, real_estate_loans, late_60_89, num_dependents]])

    # Preprocess input
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # Probability of default

    # Result
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Default (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Low Risk of Default (Confidence: {1 - prob:.2%})")

    st.progress(int(prob * 100))
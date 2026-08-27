import streamlit as st
import requests


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("💰 Loan Approval Predictor")

st.write(
    "Enter the applicant's information below "
    "to predict whether the loan is likely to be approved."
)


# ---------------------------------------------------
# Input fields
# ---------------------------------------------------

income = st.number_input(
    "Annual Income (₹)",
    min_value=20000.0,
    max_value=200000.0,
    value=75000.0,
    step=5000.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300.0,
    max_value=850.0,
    value=650.0,
    step=10.0
)

employment_years = st.number_input(
    "Employment Experience (Years)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=1.0
)

loan_amount = st.number_input(
    "Requested Loan Amount (₹)",
    min_value=50000.0,
    max_value=1000000.0,
    value=300000.0,
    step=10000.0
)

age = st.number_input(
    "Age",
    min_value=21.0,
    max_value=60.0,
    value=30.0,
    step=1.0
)


# ---------------------------------------------------
# Prediction button
# ---------------------------------------------------

if st.button("🔍 Predict Loan Approval"):

    # Prepare applicant data
    data = {
        "Income": income,
        "Credit_Score": credit_score,
        "Employment_Years": employment_years,
        "Loan_Amount": loan_amount,
        "Age": age
    }

    try:

        # Send request to FastAPI
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        # Check response
        if response.status_code == 200:

            result = response.json()

            prediction = result["result"]
            probability = result["approval_probability"]

            # Display result
            st.subheader("Prediction Result")

            if prediction == "Approved":

                st.success(
                    f"✅ Loan Approved\n\n"
                    f"Approval Probability: {probability}%"
                )

            else:

                st.error(
                    f"❌ Loan Rejected\n\n"
                    f"Approval Probability: {probability}%"
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI. "
            "Make sure the backend server is running."
        )
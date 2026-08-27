from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


# ---------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------

app = FastAPI(
    title="Loan Approval Predictor API",
    description="Machine Learning API for loan approval prediction",
    version="1.0"
)


# ---------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------

with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ---------------------------------------------------
# Define input structure
# ---------------------------------------------------

class Applicant(BaseModel):

    Income: float
    Credit_Score: float
    Employment_Years: float
    Loan_Amount: float
    Age: float


# ---------------------------------------------------
# Home endpoint
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Loan Approval Predictor API is running!"
    }


# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict(applicant: Applicant):

    # Convert input into DataFrame
    data = pd.DataFrame([{
        "Income": applicant.Income,
        "Credit_Score": applicant.Credit_Score,
        "Employment_Years": applicant.Employment_Years,
        "Loan_Amount": applicant.Loan_Amount,
        "Age": applicant.Age
    }])

    # Scale input
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)[0]

    # Get prediction probability
    probability = model.predict_proba(data_scaled)[0][1]

    # Convert prediction to readable result
    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"

    return {
        "prediction": int(prediction),
        "result": result,
        "approval_probability": round(float(probability) * 100, 2)
    }
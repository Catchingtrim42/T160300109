import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of applicants
n = 1000

# Generate applicant information
income = np.random.randint(20000, 200001, n)

credit_score = np.random.randint(300, 851, n)

employment_years = np.random.randint(0, 21, n)

loan_amount = np.random.randint(50000, 1000001, n)

age = np.random.randint(21, 61, n)

# Calculate an approval score
score = (
    (income / 200000) * 30
    + (credit_score / 850) * 40
    + (employment_years / 20) * 20
    + ((1000000 - loan_amount) / 1000000) * 10
)

# Add a little randomness
score += np.random.normal(0, 5, n)

# Approval decision
loan_approved = (score >= 55).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Income": income,
    "Credit_Score": credit_score,
    "Employment_Years": employment_years,
    "Loan_Amount": loan_amount,
    "Age": age,
    "Loan_Approved": loan_approved
})

# Save dataset
df.to_csv("loans.csv", index=False)

print("Dataset generated successfully!")
print("File created: loans.csv")
print()
print(df.head())
print()
print("Dataset shape:", df.shape)
print()
print("Approval distribution:")
print(df["Loan_Approved"].value_counts())
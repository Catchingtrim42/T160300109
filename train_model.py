import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------

df = pd.read_csv("loans.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ---------------------------------------------------
# 2. Separate features and target
# ---------------------------------------------------

X = df[
    [
        "Income",
        "Credit_Score",
        "Employment_Years",
        "Loan_Amount",
        "Age"
    ]
]

y = df["Loan_Approved"]


# ---------------------------------------------------
# 3. Split dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# 4. Feature scaling
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------
# 5. Train Logistic Regression model
# ---------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# ---------------------------------------------------
# 6. Test model
# ---------------------------------------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print()
print("Model Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# 7. Save model
# ---------------------------------------------------

with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)


# ---------------------------------------------------
# 8. Save scaler
# ---------------------------------------------------

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


print()
print("Model saved as: loan_model.pkl")
print("Scaler saved as: scaler.pkl")
print()
print("Training pipeline completed successfully!")
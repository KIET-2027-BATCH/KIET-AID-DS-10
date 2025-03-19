import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("loan_input_dataset.csv")  # Ensure the dataset is in the same folder as this file

# Fill missing values
data.ffill(inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education'], drop_first=True)

# Feature selection
selected_features = ['Gender_Male', 'Married_Yes', 'Education_Not Graduate', 
                     'ApplicantIncome', 'CoapplicantIncome']
X = data[selected_features]
y = data['LoanAmount']

# Scale the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
print(f"\nR² Score: {model.score(X_test, y_test):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# User Input for Prediction
print("\n--- Loan Amount Prediction ---\n")
try:
    gender = int(input("Enter Gender (0 = Female, 1 = Male): "))
    married = int(input("Enter Married Status (0 = No, 1 = Yes): "))
    education = int(input("Enter Education (0 = Graduate, 1 = Not Graduate): "))
    applicant_income = float(input("Enter Applicant Income: "))
    coapplicant_income = float(input("Enter Coapplicant Income: "))

    # Prepare user data for prediction
    user_data = pd.DataFrame({
        'Gender_Male': [gender],
        'Married_Yes': [married],
        'Education_Not Graduate': [education],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income]
    })

    # Scale user data for prediction
    user_data_scaled = scaler.transform(user_data)

    # Predict the loan amount
    predicted_loan_amount = model.predict(user_data_scaled)
    print(f"\nPredicted Loan Amount: ₹{predicted_loan_amount[0]:.2f}")
except ValueError:
    print("\nInvalid input. Please enter valid numeric values.")
import joblib

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model saved successfully as 'model.pkl'")

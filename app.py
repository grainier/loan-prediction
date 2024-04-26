import numpy as np
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from joblib import load

app = FastAPI()

# Load the trained Random Forest model
model = load("models/random_forest_model.pkl")


# Function to preprocess user inputs and make predictions
def predict_loan_approval(gender, married, dependents, education, self_employed, applicant_monthly_income, co_applicant_monthly_income,
                          total_loan_amount, loan_amount_term_in_months, credit_history, property_area):
    # Convert categorical inputs to numerical format
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents >= 3 else dependents
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area_semiurban = 1 if property_area == "Semiurban" else 0
    property_area_urban = 1 if property_area == "Urban" else 0
    loan_amount = int(total_loan_amount / loan_amount_term_in_months)
    monthly_balance_income = (applicant_monthly_income + co_applicant_monthly_income) - loan_amount

    # Create input array for prediction
    input_data = np.array([[
        gender, married, dependents, education, self_employed,
        loan_amount, loan_amount_term_in_months, credit_history,
        property_area_semiurban, property_area_urban,
        monthly_balance_income
    ]])

    # Make prediction
    prediction = model.predict(input_data)

    # Map prediction to human-readable format
    result = "Loan Approved" if prediction[0] == 1 else "Loan Not Approved"

    return result


# Define the route for the single-page application
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as file:
        html_template = file.read()
    return HTMLResponse(content=html_template, status_code=200)

# Define the route for receiving form submissions and making predictions
@app.post("/predict")
async def predict_loan_approval_api(gender: str = Form(...), married: str = Form(...),
                                    dependents: int = Form(...), education: str = Form(...),
                                    self_employed: str = Form(...), applicant_monthly_income: int = Form(...),
                                    co_applicant_monthly_income: int = Form(...), total_loan_amount: int = Form(...),
                                    loan_amount_term_in_months: int = Form(...), credit_history: float = Form(...),
                                    property_area: str = Form(...)):
    # Predict loan approval
    prediction_result = predict_loan_approval(gender, married, dependents, education, self_employed,
                                              applicant_monthly_income, co_applicant_monthly_income, total_loan_amount,
                                              loan_amount_term_in_months, credit_history, property_area)
    return prediction_result

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)

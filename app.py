import numpy as np
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from joblib import load

app = FastAPI()

# Load the trained Random Forest model
model = load("models/random_forest_model.pkl")


# Function to preprocess user inputs and make predictions
def predict_loan_approval(gender, married, dependents, education, self_employed, applicant_income, coapplicant_income,
                          loan_amount, loan_amount_term, credit_history, property_area):
    # Convert categorical inputs to numerical format
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area_rural = 1 if property_area == "Rural" else 0
    property_area_semiurban = 1 if property_area == "Semiurban" else 0
    property_area_urban = 1 if property_area == "Urban" else 0

    # Create input array for prediction
    input_data = np.array([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income,
                            loan_amount, loan_amount_term, credit_history, property_area_rural, property_area_semiurban,
                            property_area_urban]])

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
                                    self_employed: str = Form(...), applicant_income: int = Form(...),
                                    coapplicant_income: int = Form(...), loan_amount: int = Form(...),
                                    loan_amount_term: int = Form(...), credit_history: float = Form(...),
                                    property_area: str = Form(...)):
    # Predict loan approval
    prediction_result = predict_loan_approval(gender, married, dependents, education, self_employed,
                                              applicant_income, coapplicant_income, loan_amount,
                                              loan_amount_term, credit_history, property_area)
    return prediction_result

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)

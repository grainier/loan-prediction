### Project Overview

This repository contains code for a loan approval prediction project using machine learning techniques. The project aims
to develop a predictive model that assists banks in automating the loan approval process based on applicant features.

### Project Components

**Jupyter Notebook:** The main analysis and modeling work are conducted in a Jupyter
Notebook [notebook.ipynb](notebooks%2Fnotebook.ipynb). This notebook covers data preprocessing, exploratory data
analysis (EDA), model training, evaluation, and model selection.

**FastAPI Web Application:** Additionally, a FastAPI web application ([app.py](app.py)) is provided, allowing users to
input applicant details and receive predictions on loan approval status in real-time.

### Setup Instructions

#### Prerequisites

- Python 3.9 or higher installed
- Poetry package manager installed (`pip install poetry`)

#### Install Dependencies:

- `poetry install`

#### Run Jupyter Notebook:

- `poetry run jupyter notebook notebooks/notebook.ipynb`

#### Run Web Application:

- `poetry run python app.py`

#### Web Application Instructions:

- Once the web application is running, Access the FastAPI web application at http://localhost:8000 in your web browser.
- Fill in the applicant details in the provided form fields.
- Click the "Predict" button to receive a prediction on loan approval status.
- The prediction result will be displayed on the web page.

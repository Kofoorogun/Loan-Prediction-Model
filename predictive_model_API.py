from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
import pickle
import json
import requests

class input_data(BaseModel):
    Loan_Term : str
    Verification_by_Loan_Company : str
    Loan_Purpose : str
    Application_Type : str
    Total_Years_of_Employment : float
    Annual_Income : float
    Income_Debt_Ratio : float
    No_of_Open_Credit_Lines : int
    Credit_Utilization_Rate : float
    No_of_Mortgage_Account :int

# load the saved model
with open("model.pkl", 'rb') as pkl:
    model = pickle.load(pkl)

# Declaring our FastAPI instance
app = FastAPI()

@app.post('/request body')

def predict(input_parameters : input_data):
    input_dict = input_parameters.dict()
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)
    
    if (prediction[0] == 0):
        return 'Applicant will not default'
    else:
        return 'Applicant will default'
    
if __name__ == '_main_':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

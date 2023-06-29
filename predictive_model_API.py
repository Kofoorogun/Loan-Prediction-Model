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

@app.post('/request_body')

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

# # load the saved model
# with open("model.pkl", 'rb') as pkl:
#     model = pickle.load(pkl)

# # Creating an Endpoint to receive the data
# # to make prediction on.
# app = FastAPI()

# @app.post('/request_body')
# def predict(input_parameters : input_data):
    
#     input_dict = input_parameters.dict()
#     # input_data = input_parameters.json()
#     # input_dictionary = json.loads(input_data)
#     # convert input_dictionary to a dataframe
#     input_df= pd.DataFrame([input_dict])
#     # get the column values as a numpy array
#     input_array = input_df.values
#     # get the column values
    
#     # Loan_Term =  input_dictionary["Loan_Term"]
#     # Verification_by_Loan_Company = input_dictionary["Verification_by_Loan_Company"]
#     # Loan_Purpose = input_dictionary["Loan_Purpose"]
#     # Application_Type = input_dictionary["Application_Type"]
#     # Total_Years_of_Employment = input_dictionary["Total_Years_of_Employment"]
#     # Annual_Income = input_dictionary["Annual_Income"]
#     # Income_Debt_Ratio = input_dictionary["Income_Debt_Ratio"]
#     # No_of_Open_Credit_Lines = input_dictionary["No_of_Open_Credit_Lines"]
#     # Credit_Utilization_Rate = input_dictionary["Credit_Utilization_Rate"]
#     # No_of_Mortgage_Account = input_dictionary["No_of_Mortgage_Account"]

#     input_array = [[input_dict["Loan_Term"], 
#                    input_dict["Verification_by_Loan_Company"], 
#                    input_dict["Loan_Purpose"],
#                    input_dict["Application_Type"], 
#                    input_dict["Annual_Income"], 
#                    input_dict["Income_Debt_Ratio"], 
#                    input_dict["No_of_Open_Credit_Lines"],
#                    input_dict["Credit_Utilization_Rate"], 
#                    input_dict["No_of_Mortgage_Account"]]]
    
#     prediction = model.predict([input_array])
    
#     if (prediction[0] == 0):
#         return 'Applicant will not default'
#     else:
#         return 'Applicant will default'
    
# if __name__ == '_main_':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8003)




# # my_dict = {'Loan_Term' : '36 months', 
# #             'Verification_by_Loan_Company' : 'not verified',
# #             'Loan_Purpose' : 'vacation',
# #             'Application_Type' : 'individual',
# #             'Total_Years_of_Employment' : 4,
# #             'Annual_Income' : 65000,
# #             'Income_Debt_Ratio' : 26.24,
# #             'No_of_Open_Credit_Lines' : 13,
# #             'Credit_Utilization_Rate' : 21.5,
# #             'No_of_Mortgage_Account' : 3}

# # json_object = json.dumps(my_dict, indent = 4)

# # predict(json_object)

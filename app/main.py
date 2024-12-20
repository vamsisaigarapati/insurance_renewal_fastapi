from typing import Union

from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd
import numpy as np

import numpy as np
app = FastAPI()

import dill
with open('/Users/vamsisaigarapati/Documents/github/insurance_renewal_app/app/random_forest_pipeline.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


class Payload(BaseModel):
    Upper_Age: int  
    Lower_Age: int  
    Reco_Policy_Premium: float  
    City_Code: str  
    Accomodation_Type: str  
    Reco_Insurance_Type: str  
    Is_Spouse: str  
    Health_Indicator: str  
    Holding_Policy_Duration: float 
    Holding_Policy_Type: str

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Name": "Vamsi Sai Garapati",
        "Project": "Insurance Renewal",
        "Model": "Random Forest classifier"
    }


@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    prediction = int(y_hat[0])
    output= "The customer possibly can renew the Insurance plan" if predict==1 else "The customer possibly cannot renew the Insurance plan"
    return {"prediction": output}
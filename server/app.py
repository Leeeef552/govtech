from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import Any, Dict
import pandas as pd
from api.analyst import Analyst, QueryResult
from utils.utils import preprocess


# Load your trained ML model
model = joblib.load("model/xgb_tuned.joblib")

app = FastAPI()

########################################
##              pydantic              ##
########################################

class PredictionRequest(BaseModel):
    month: str
    town: str
    flat_type: str
    flat_model: str
    storey_range: str
    floor_area_sqm: int
    lease_commence_date: int


class PredictionResponse(BaseModel):
    predicted_price: float


class AnalystRequest(BaseModel):
    query: str


class AnalystResponse(BaseModel):
    sql: str
    results: list
    columns: list
    explanation: str


########################################
##              endpoints             ##
########################################

## predict
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # Convert request into dict
    input_dict = data.dict()
    # Preprocess into feature DataFrame
    X = preprocess(input_dict)
    # Predict
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}


## analyze
analyst = Analyst("data/hdb_prices.db")

@app.post("/analyze", response_model=AnalystResponse)
def analyze(request: AnalystRequest):
    result: QueryResult = analyst.query(request.query, display=False)
    return str(AnalystResponse(
        sql=result.sql,
        results=result.results,
        columns=result.columns,
        explanation=result.explanation
    ))

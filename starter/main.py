from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

# Load the model, encoder, and label binarizer
model = joblib.load("model/model.joblib")
encoder = joblib.load("model/encoder.joblib")
lb = joblib.load("model/lb.joblib")


class CensusItem(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
async def predict(item: CensusItem):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([item.dict(by_alias=True)])

    # Process input data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Make prediction
    prediction = inference(model, X)

    # Convert prediction to label
    predicted_label = lb.inverse_transform(prediction)[0].strip()

    return {"prediction": predicted_label}

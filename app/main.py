from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_price

app = FastAPI()


class PredictionInput(BaseModel):
    town: str
    flat_type: str
    floor_area_sqm: float
    flat_model: str
    lease_commence_date: int
    sale_year: int
    sale_month: int
    storey_mid: float
    remaining_lease_years: float


@app.get("/")
def home():
    return {"message": "HDB Price Prediction API is running!"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    prediction = predict_price(
        input_data.town,
        input_data.flat_type,
        input_data.floor_area_sqm,
        input_data.flat_model,
        input_data.lease_commence_date,
        input_data.sale_year,
        input_data.sale_month,
        input_data.storey_mid,
        input_data.remaining_lease_years
    )

    return {"predicted_price": prediction}
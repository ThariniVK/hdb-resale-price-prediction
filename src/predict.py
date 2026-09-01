import joblib
import pandas as pd
from pathlib import Path


# Find the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model
model_path = BASE_DIR / "models" / "hdb_price_model.pkl"
model = joblib.load(model_path)


def predict_price(
    town,
    flat_type,
    floor_area_sqm,
    flat_model,
    lease_commence_date,
    sale_year,
    sale_month,
    storey_mid,
    remaining_lease_years
):
    
    flat = pd.DataFrame({
        "town": [town],
        "flat_type": [flat_type],
        "floor_area_sqm": [floor_area_sqm],
        "flat_model": [flat_model],
        "lease_commence_date": [lease_commence_date],
        "sale_year": [sale_year],
        "sale_month": [sale_month],
        "storey_mid": [storey_mid],
        "remaining_lease_years": [remaining_lease_years]
    })
    
    prediction = model.predict(flat)[0]
    
    return float(prediction)
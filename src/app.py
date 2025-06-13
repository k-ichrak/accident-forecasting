from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

app = FastAPI()

class DateInput(BaseModel):
    year: int
    month: int

# Load model
model: SARIMAXResultsWrapper = joblib.load("model.pkl")


base_year, base_month = 2020, 12

@app.post("/")
async def predict(data: DateInput):
    year = data.year
    month = data.month

    steps = (year - base_year) * 12 + (month - base_month)
    if steps < 1:
        return {"error": "Date must be after December 2020"}

    forecast = model.get_forecast(steps=steps)
    pred_value = forecast.predicted_mean.iloc[-1]

    return {"prediction": round(float(pred_value), 2)}


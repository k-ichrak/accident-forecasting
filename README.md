# Accident Predictor (Alkoholunfälle Forecasting)

This project forecasts monthly **alcohol-related traffic accidents** in Germany using a SARIMA time series model. The model is deployed as an API using FastAPI.

## Project Overview

- **Data Source:** [German Traffic Accident Dataset](https://www-genesis.destatis.de/)
- **Forecasting Model:** SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- **Target Variable:** Monthly total of `Alkoholunfälle (insgesamt)`
- **API:** FastAPI with a deployed POST endpoint that returns predictions

##  How it Works

1. The data is preprocessed and filtered to include only the total alcohol-related accidents.
2. A SARIMA model is trained on data until December 2020.
3. The trained model is saved as `model.pkl`.
4. The FastAPI app serves predictions for any month after December 2020.

## Example API Usage

POST request to `/`:

```json
{
  "year": 2021,
  "month": 1
}

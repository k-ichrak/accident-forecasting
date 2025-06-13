import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Load the data
df = pd.read_csv(r"C:\Users\khald\Downloads\monatszahlen2505_verkehrsunfaelle_06_06_25.csv", sep=",")

# Filter for Alkoholunfälle and 'insgesamt'
df = df[(df["MONATSZAHL"] == "Alkoholunfälle") & (df["AUSPRAEGUNG"] == "insgesamt")]

# Filter and convert date
df = df[df["MONAT"].astype(str).str.match(r"^\d{6}$", na=False)].copy()
df["date"] = pd.to_datetime(df["MONAT"], format="%Y%m")
df = df.sort_values("date").set_index("date")

# Train-test split
train = df[:'2020-12-01']
test = df['2021-01-01':]

# Fit SARIMA model
model = SARIMAX(train["WERT"], order=(2,2,1), seasonal_order=(0,0,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# Forecast January 2021
forecast = results.get_forecast(steps=1)
pred_value = forecast.predicted_mean.iloc[0]

# Actual value
actual_value = df.loc['2021-01-01', 'WERT']

# Evaluate
mae = mean_absolute_error([actual_value], [pred_value])
mse = mean_squared_error([actual_value], [pred_value])
mape = np.mean(np.abs((actual_value - pred_value) / actual_value)) * 100

# Save model
joblib.dump(results, "model.pkl")

# Print results
print(f" Predicted WERT for January 2021: {pred_value:.2f}")
print(f" Actual WERT: {actual_value:.2f}")
print(f" MAE: {mae:.2f}")
print(f" MSE: {mse:.2f}")
print(f" MAPE: {mape:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train["WERT"], label='Training Data')
plt.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', label='Forecast Point')
plt.scatter(pd.to_datetime('2021-01-01'), actual_value, color='green', label='Actual Jan 2021')
plt.scatter(pd.to_datetime('2021-01-01'), pred_value, color='red', label='Predicted Jan 2021')
plt.title("Forecasting Alkoholunfälle (insgesamt)")
plt.xlabel("Date")
plt.ylabel("WERT")
plt.legend()
plt.tight_layout()
plt.show()
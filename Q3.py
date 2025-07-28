import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

df = pd.read_csv(
    r"C:\Users\Ali's HP\Desktop\task3\household_power_consumption.txt", 
    sep=';', 
    parse_dates={'Datetime': ['Date', 'Time']}, 
    infer_datetime_format=True, 
    na_values='?', 
    low_memory=False
)


df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

df = df[['Global_active_power']].dropna()
df = df.astype(float)

df_hourly = df.resample('H').mean()
print(df_hourly.head())

df_hourly['hour'] = df_hourly.index.hour
df_hourly['dayofweek'] = df_hourly.index.dayofweek
df_hourly['month'] = df_hourly.index.month
df_hourly['is_weekend'] = (df_hourly['dayofweek'] >= 5).astype(int)

train = df_hourly[:-24*7]
test = df_hourly[-24*7:]

model_arima = ARIMA(train['Global_active_power'], order=(2,1,2))
arima_result = model_arima.fit()

arima_forecast = arima_result.forecast(steps=len(test))

prophet_df = train.reset_index().rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
model_prophet = Prophet()
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=len(test), freq='H')
forecast_prophet = model_prophet.predict(future)

prophet_forecast = forecast_prophet.iloc[-len(test):]['yhat'].values

features = ['hour', 'dayofweek', 'month', 'is_weekend']
X_train = train[features]
y_train = train['Global_active_power']
X_test = test[features]

model_xgb = XGBRegressor(n_estimators=100)
model_xgb.fit(X_train, y_train)
xgb_forecast = model_xgb.predict(X_test)

#evaluate all models 
def evaluate(true, pred, name):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

evaluate(test['Global_active_power'], arima_forecast, "ARIMA")
evaluate(test['Global_active_power'], prophet_forecast, "Prophet")
evaluate(test['Global_active_power'], xgb_forecast, "XGBoost")

plt.figure(figsize=(15, 5))
plt.plot(test.index, test['Global_active_power'], label='Actual', linewidth=2)
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, prophet_forecast, label='Prophet Forecast')
plt.plot(test.index, xgb_forecast, label='XGBoost Forecast')
plt.legend()
plt.title("Actual vs Forecasted Energy Usage")
plt.xlabel("Datetime")
plt.ylabel("Global Active Power (kW)")
plt.show()
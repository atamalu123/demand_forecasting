import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

def calculate_wape(y_true, y_pred):
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return wape

def forecast_wapes(df: pd.DataFrame, warehouse: str, product_code: str) -> List:

    ### Subset based on warehouse and product code
    df_subset = df[(df["Warehouse"] == warehouse) & (df["Product_Code"] == product_code)]

    ### Create time series
    series = (
        df_subset
        .set_index("Date")["Order_Demand"]
        .asfreq("MS")
    )
    series = series.fillna(0)

    ### Train/test split
    h = 12 # 12 periods
    train, test = series.iloc[:-h], series.iloc[-h:]

    # Last month
    naive_forecast = pd.Series(
        [train.iloc[-1]] * h, 
        index=test.index
    )

    # Average of last 12 months
    last_12_avg = train.iloc[-12:].mean()
    avg12_forecast = pd.Series(
        [last_12_avg] * h,
        index=test.index
    )

    # Same month of last year
    seasonal_naive_forecast = series.shift(12).iloc[-h:]

    # ETS
    ets_model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    ets_forecast = ets_model.forecast(h)

    # ARIMA No seasonality
    arima_s = auto_arima(
        train,
        seasonal=True,
        m=12,
        stepwise=True
    )
    sarima_forecast_no_seasonality = arima_s.predict(n_periods=h)

    # SARIMA w/ seasonality
    arima_s2 = auto_arima(
        train,
        stepwise=True
    )
    sarima_forecast_seasonality = arima_s2.predict(n_periods=h)

    ### Calculate WAPEs
    wape_values = []

    wape_values.append(calculate_wape(test, naive_forecast))
    wape_values.append(calculate_wape(test, avg12_forecast))
    wape_values.append(calculate_wape(test, seasonal_naive_forecast))
    wape_values.append(calculate_wape(test, ets_forecast))
    wape_values.append(calculate_wape(test, sarima_forecast_no_seasonality))
    wape_values.append(calculate_wape(test, sarima_forecast_seasonality))

    return wape_values

MODEL_NAMES = [
    "Last_Month",      
    "Last_12_Avg",     
    "Seasonal_Naive",   
    "ETS",        
    "ARIMA",      
    "SARIMA"    
]

def best_future_forecast(df: pd.DataFrame, warehouse: str, product_code: str, h: int = 12):
    """
    Uses forecast_wapes() to pick the best model (lowest WAPE),
    then fits that same model on the FULL series (df) and produces an h-step forecast.
    Returns: (best_model_name, best_wape, forecast_series)
    """
    # 1) Get WAPEs from your unchanged function
    wapes = forecast_wapes(df, warehouse, product_code)
    best_idx = int(np.nanargmin(wapes))  # pick lowest WAPE (ignores NaNs)

    # 2) Rebuild the FULL series from df (same steps as in your function)
    df_subset = df[(df["Warehouse"] == warehouse) & (df["Product_Code"] == product_code)]
    series = (
        df_subset
        .set_index("Date")["Order_Demand"]
        .asfreq("MS")
    ).fillna(0)

    # 3) Future index
    last_date = series.index.max()
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS")

    # 4) Fit the selected model on the FULL series & forecast
    if best_idx == 0:
        # Last month (repeat last observed)
        y_last = series.iloc[-1]
        forecast = pd.Series([y_last] * h, index=future_index)

    elif best_idx == 1:
        # Average of last 12 months (repeat)
        y_avg12 = series.iloc[-12:].mean()
        forecast = pd.Series([y_avg12] * h, index=future_index)

    elif best_idx == 2:
        # Seasonal naive = repeat last 12 months pattern
        vals = series.iloc[-12:].values
        # if h > 12, tile; else take first h
        reps = int(np.ceil(h / 12))
        forecast_vals = np.tile(vals, reps)[:h]
        forecast = pd.Series(forecast_vals, index=future_index)

    elif best_idx == 3:
        # ETS on full series
        ets_model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()
        forecast = pd.Series(ets_model.forecast(h), index=future_index)

    elif best_idx == 4:
        # auto_arima with seasonal=True, m=12
        arima_s = auto_arima(series, seasonal=True, m=12, stepwise=True)
        forecast = pd.Series(arima_s.predict(n_periods=h), index=future_index)

    else:
        # auto_arima without seasonal args
        arima_s2 = auto_arima(series, stepwise=True)
        forecast = pd.Series(arima_s2.predict(n_periods=h), index=future_index)

    return MODEL_NAMES[best_idx], float(wapes[best_idx]), forecast

def plot_product_demand_forecast(df: pd.DataFrame, warehouse: str, product: str):

    product_1 = df[(df["Warehouse"] == warehouse) & (df["Product_Code"] == product)]
    model_name, wape, forecast = best_future_forecast(df, warehouse, product)

    if not isinstance(forecast, pd.Series):
        # assume forecast is an array → build an index starting after last date
        last_date = product_1['Date'].max()
        forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                                    periods=len(forecast), freq="MS")
        forecast = pd.Series(forecast, index=forecast_index)

    plt.plot(
        product_1['Date'],
        product_1['Order_Demand']
    )
    plt.plot(forecast.index, forecast.values, label=f"Forecast ({model_name})", linestyle="--")

    plt.title(f"{product} - Best model: {model_name}, WAPE={wape:.2f}%")
    plt.xlabel("Date")
    plt.ylabel("Order Demand")

    plt.show()
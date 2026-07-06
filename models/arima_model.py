import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def run_arima(df):

    data = df["Close"].astype(float).dropna()

    model = ARIMA(data, order=(5,1,0))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)

    return forecast.to_numpy()
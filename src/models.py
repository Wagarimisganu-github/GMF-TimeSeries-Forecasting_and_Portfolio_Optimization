import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def build_arima_model(train_series):
    """
    Builds and fits an ARIMA model using auto_arima to find optimal parameters.
    """
    auto_model = auto_arima(train_series, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
    return auto_model

def build_lstm_model(train_data, n_steps=60, epochs=50, batch_size=32):
    """
    Builds, compiles, and fits an LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(n_steps, len(scaled_data)):
        X_train.append(scaled_data[i-n_steps:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    return model, scaler

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculates and prints evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return mae, rmse, mape

def plot_forecast(historical_data, test_data, forecast, model_name, confidence_intervals=None):
    """
    Plots the historical, test, and forecasted data.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(historical_data.index, historical_data.values, label='Historical Data (Train)')
    plt.plot(test_data.index, test_data.values, label='Historical Data (Test)')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')

    if confidence_intervals is not None:
        plt.fill_between(forecast.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='red', alpha=0.2, label='Confidence Interval')

    plt.title(f"{model_name} Forecast vs. Actual")
    plt.xlabel("Date")
    plt.ylabel("TSLA Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{model_name}_forecast.png')
    plt.show()
    
import streamlit as st
import yfinance as yf
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta

from preprocess import create_sequences
from model import LSTMModel
from strategy import generate_signals
from backtest import backtest

st.title("AI Trading Assistant")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Run Analysis"):
    data = yf.download(ticker, period="5y")

    if data.empty:
        st.error("Invalid ticker")
    else:
        df = pd.DataFrame()
        df["Close"] = data["Close"]

        close = df["Close"].squeeze()

        df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
        df["MA"] = close.rolling(20).mean()
        df["MACD"] = ta.trend.MACD(close).macd()

        df.dropna(inplace=True)

        # Preprocess
        X, y, scaler = create_sequences(df.values, seq_length=20)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Load model
        model = LSTMModel(input_size=4)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        with torch.no_grad():
            predictions = model(X_tensor).numpy()

        # Inverse scaling
        num_features = df.shape[1]

        dummy_pred = np.zeros((predictions.shape[0], num_features))
        dummy_pred[:, 0] = predictions[:, 0]

        dummy_actual = np.zeros((y.shape[0], num_features))
        dummy_actual[:, 0] = y

        predictions = scaler.inverse_transform(dummy_pred)[:, 0]
        y_actual = scaler.inverse_transform(dummy_actual)[:, 0]

        # RSI aligned
        rsi_values = df["RSI"].values[-len(predictions):]

        # Signals
        signals = generate_signals(predictions, y_actual, rsi_values)

        # Backtest
        final_value, profit = backtest(signals, y_actual)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(y_actual, label="Actual")
        ax.plot(predictions, label="Predicted")
        ax.legend()

        st.pyplot(fig)

        st.subheader("Results")
        st.write(f"Final Value: ₹{final_value:.2f}")
        st.write(f"Profit: ₹{profit:.2f}")

        st.subheader("Latest Signals")
        st.write(signals[-10:])
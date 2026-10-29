import yfinance as yf
import pandas as pd
import ta

def fetch_stock_data(ticker="AAPL", start="2015-01-01"):
    df = yf.download(ticker, start=start)

    # Ensure proper 1D series
    close = df["Close"].squeeze()

    df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    df["MA"] = close.rolling(window=20).mean()
    df["MACD"] = ta.trend.MACD(close).macd()

    df = df[["Close", "RSI", "MA", "MACD"]]
    df.dropna(inplace=True)

    return df
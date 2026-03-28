import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="RELIANCE.NS", start="2015-01-01"):
    df = yf.download(ticker, start=start)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    data = fetch_stock_data()
    print(data.head())
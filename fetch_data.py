import requests
import numpy as np
from scipy.stats import norm
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.optimize import brentq

API_KEY = "YOUR API KEY HERE"
valid_tickers = []

strikes = range(4000, 7000, 50)
expiries = ["260918", "261218", "260320"]

for expiry in expiries:
    for K in strikes:
        for cp in ["C", "P"]:
            ticker = f"O:SPX{expiry}{cp}0{K}000"
            url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/2026-01-01/2026-06-01"
            r = requests.get(url, params={"apiKey": API_KEY})
            data = r.json()
            if data.get('resultsCount', 0) > 0:
                valid_tickers.append((ticker, data['resultsCount']))
                print(ticker, data['resultsCount'])
            time.sleep(12)

liquid = [(t, n) for t, n in valid_tickers if n >= 50]

all_data = {}

for ticker, _ in liquid:
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/2026-01-01/2026-06-03"
    r = requests.get(url, params={"apiKey": API_KEY})
    data = r.json()
    if data.get('resultsCount', 0) > 0:
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('date')[['c', 'v']]  # close price et volume
        df.columns = ['close', 'volume']
        all_data[ticker] = df
    time.sleep(12)

spx = yf.Ticker("^SPX").history(start="2026-01-01", end="2026-06-03")['Close']
spx.index = spx.index.tz_localize(None)

prices = pd.DataFrame({ticker: df['close'] for ticker, df in all_data.items()})
prices.index = prices.index.tz_localize(None).normalize()

def parse_ticker(ticker):
    parts = ticker.replace('O:SPX', '')
    expiry = parts[:6]
    cp = parts[6]
    strike = int(parts[7:]) / 1000
    return expiry, cp, strike

def parse_expiry(expiry_str):
    return datetime.strptime(expiry_str, '%y%m%d')

ticker_info = {}
for ticker in prices.columns:
    expiry, cp, strike = parse_ticker(ticker)
    ticker_info[ticker] = {'expiry': expiry, 'cp': cp, 'strike': strike}

ticker_df = pd.DataFrame(ticker_info).T
ticker_df['strike'] = ticker_df['strike'].astype(float)
ticker_df['expiry_date'] = ticker_df['expiry'].apply(parse_expiry)

tbill = yf.Ticker("^IRX")
r = tbill.history(period="1d")["Close"].iloc[-1] / 100

def BS_call(S0, K, r, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BS_put(S0, K, r, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

def get_iv(row, T_i, S):
    K = row["strike"] / S
    mid = row["price"] / S
    if row['cp'] == 'C':
        try:
            iv = brentq(lambda sigma: BS_call(1, K, r, T_i, sigma) - mid, 1e-6, 5.0)
            return iv
        except:
            return np.nan
    else:
        try:
            iv = brentq(lambda sigma: BS_put(1, K, r, T_i, sigma) - mid, 1e-6, 5.0)
            return iv
        except:
            return np.nan

    
df_iv = prices.stack().reset_index()
df_iv.columns = ['date', 'ticker', 'price']
df_iv = df_iv.join(ticker_df, on='ticker')
df_iv['S'] = df_iv['date'].map(spx)
df_iv['expiry_date'] = df_iv['expiry'].apply(parse_expiry)
df_iv['T'] = (df_iv['expiry_date'] - df_iv['date']).dt.days / 365
    
df_iv['iv_market'] = df_iv.apply(lambda row: get_iv(row, row['T'], row['S']), axis=1)

df_iv.to_csv("df_iv.csv", index=False)
spx.to_csv("spx.csv")
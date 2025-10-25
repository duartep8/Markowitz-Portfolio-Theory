import yfinance as yf
import pandas as pd
import os

from datetime import datetime, timedelta

# ---- TICKER LIST ----

tickers = [
    # --- Technology (Software / Semiconductors) ---
    'HO.PA',
    'SYP.DE',
    'ASML.AS',
    'SAP.DE',
    'IFX.DE',

    # --- Consumer & Luxury Goods ---
    'MC.PA',
    'RACE.MI',
    'ITX.MC',
    'AD.AS',

    # --- Industrials & Aerospace ---
    'AIR.PA',
    'SIE.DE',
    'SU.PA',

    # --- Healthcare ---
    'SAN.PA',
    'SHL.DE',

    # --- Financials ---
    'ALV.DE',
    'BNP.PA',
    'INGA.AS',

    # --- Energy & Utilities ---
    'TTE.PA',
    'IBE.MC',
    'ENEL.MI',
    'EDP.LS',

    # --- Materials & Chemicals ---
    'BAS.DE',
    'LIN.DE',

    # --- Telecommunications ---
    'DTE.DE',
    'TEF.MC'
]

# define start and end dates
end_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
start_date = datetime.strptime("2020-01-31", "%Y-%m-%d")

data = yf.download(tickers, start= start_date, end=end_date, auto_adjust=True, interval="1mo")

adj_close_df = data['Close']  # Extract adjusted close prices

file_path = 'data/stock_data.csv'
adj_close_df.to_csv(file_path)

print(f"Saved adjusted close prices for {len(tickers)} tickers to: {file_path}")

import pandas as pd
import os

def load_stock_data(file_path="data/stock_data.csv"):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

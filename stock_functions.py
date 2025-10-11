import pandas as pd
import numpy as np

def simple_returns(adj_close_df):
    s_simple_returns = adj_close_df.pct_change()  # (P_t - P_{t-1}) / P_{t-1}
    s_simple_returns = s_simple_returns.dropna()  # Drop NaN from first row
    return s_simple_returns

def covariance_matrix(log_returns):
    return log_returns.cov() * 252  # Annualize covariance matrix assuming 252 trading days

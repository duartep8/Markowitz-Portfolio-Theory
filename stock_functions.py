import pandas as pd
import numpy as np

def simple_returns(adj_close_df):
    s_simple_returns = adj_close_df.pct_change()  # (P_t - P_{t-1}) / P_{t-1}
    s_simple_returns = s_simple_returns.dropna()  # Drop NaN from first row
    return s_simple_returns

def covariance_matrix(simple_returns):
    return simple_returns.cov() * 12  # Annualize covariance matrix assuming 12 months

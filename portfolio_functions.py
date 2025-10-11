import numpy as np

#portfolio standard deviation
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights # transpose weights so an array that is 1 x n -> n x 1 then multiply with cov matrix and weights
    return np.sqrt(variance) # sd

# portfolio return
def expected_returns(weights, simple_returns):
    return np.sum(simple_returns.mean() * weights) * 252 # annualized returns

# Sharpe ratio
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate = 0.02):
    return (expected_returns(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# Maximize Sharpe ratio (scipy minimize function minimizes a function so we need to negate the sharpe ratio, it has no maximize function)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate = 0.02):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# variance
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

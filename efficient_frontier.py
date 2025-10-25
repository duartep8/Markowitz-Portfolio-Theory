import numpy as np

from scipy.optimize import minimize
from portfolio_functions import expected_returns, standard_deviation, portfolio_variance
from download_data import tickers

def generate_efficient_frontier(returns, cov_matrix, mvp_weights, num_points=200):
    """
    Generate points on the efficient frontier by varying target returns.
    Returns arrays of volatilities and returns for plotting.
    """
    # Get range of possible returns - start from MVP return to max individual stock return
    mvp_return = expected_returns(mvp_weights, returns)
    
    # Find maximum return among all individual stocks
    max_return = max([expected_returns(np.eye(len(tickers))[i], returns) for i in range(len(tickers))])
    
    # Extend the range to ensure frontier goes beyond tangent portfolio
    extended_max = max_return * 1.1
    
    # Generate target returns starting from MVP return
    target_returns = np.linspace(mvp_return, extended_max, num_points)
    
    efficient_vols = []
    efficient_returns = []
    
    bounds = tuple((-0.08, 0.1) for _ in range(len(tickers)))
    
    # Constraints: sum of weights = 1, and target return
    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: expected_returns(w, returns) - target_return}
        )
        
        initial_weights = np.array([1/len(tickers)] * len(tickers))
        
        try:
            result = minimize(portfolio_variance, initial_weights,
                            args=(cov_matrix,),
                            method='SLSQP',
                            constraints=constraints,
                            bounds=bounds)  
            
            if result.success:
                vol = standard_deviation(result.x, cov_matrix)
                efficient_vols.append(vol)
                efficient_returns.append(target_return)
        except:
            continue
    
    return np.array(efficient_returns), np.array(efficient_vols)
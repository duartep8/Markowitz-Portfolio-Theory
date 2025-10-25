import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_stock_data
from stock_functions import simple_returns, covariance_matrix
from portfolio_functions import neg_sharpe_ratio
from download_data import tickers
from portfolio_functions import standard_deviation, expected_returns, sharpe_ratio, portfolio_variance
from efficient_frontier import generate_efficient_frontier
from plot_functions import plot_portfolio_weights, prepare_portfolio_data, plot_industry_weights, plot_sector_weights

def main():
    adj_close_df = load_stock_data("data/stock_data.csv")
    returns = simple_returns(adj_close_df)
    cov_matrix = covariance_matrix(returns)
    
    # Constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    #Bondaries for weights: between -8% and 10%
    bounds = tuple((-0.08, 0.1) for _ in range(len(tickers)))
    
    # Set initial weights
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    
    # Optimize weights to maximize Sharpe ratio
    optimized_results = minimize(neg_sharpe_ratio, initial_weights,
                                args=(returns, cov_matrix),
                                method='SLSQP',
                                constraints=constraints,
                                bounds=bounds)
    optimal_weights = optimized_results.x
    
    # Find MVP
    mvp_result = minimize(portfolio_variance,
                         initial_weights,
                         args=(cov_matrix,),
                         method='SLSQP',
                         constraints=constraints,
                         bounds=bounds)
    mvp_weights = mvp_result.x
    
    # Calculate MVP metrics
    mvp_return = expected_returns(mvp_weights, returns)
    mvp_volatility = standard_deviation(mvp_weights, cov_matrix)
    mvp_sharpe = sharpe_ratio(mvp_weights, returns, cov_matrix)
    
    # Print optimal portfolio values
    print("Optimal Weights (Max Sharpe Ratio):")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")
    print()
    
    optimal_portfolio_return = expected_returns(optimal_weights, returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, returns, cov_matrix)
    
    print(f"Portfolio Expected Return: {optimal_portfolio_return:.4f}")
    print(f"Portfolio Volatility: {optimal_portfolio_volatility:.4f}")
    print(f"Portfolio Sharpe Ratio: {optimal_sharpe_ratio:.4f}")
    
    # Print MVP results
    print("\nMinimum Variance Portfolio:")
    print("Weights:")
    for ticker, weight in zip(tickers, mvp_weights):
        print(f"{ticker}: {weight:.4f}")
    print(f"\nMVP Expected Return: {mvp_return:.4f}")
    print(f"MVP Volatility: {mvp_volatility:.4f}")
    print(f"MVP Sharpe Ratio: {mvp_sharpe:.4f}")
    
    # Generate and plot efficient frontier
    print("\nGenerating efficient frontier...")
    eff_returns, eff_vols = generate_efficient_frontier(returns, cov_matrix, mvp_weights, num_points=200)
    
    # Risk-free rate
    risk_free_rate = 0.0193

    plt.figure(figsize=(10, 7))
    
    # Plot efficient frontier
    plt.plot(eff_vols, eff_returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot optimal portfolio (max Sharpe ratio)
    plt.plot(optimal_portfolio_volatility, optimal_portfolio_return, 'g*', 
             markersize=20, label='Optimal Portfolio (Max Sharpe)')
    
    # Plot MVP
    plt.plot(mvp_volatility, mvp_return, 'r*', 
             markersize=20, label='Minimum Variance Portfolio')
    
    # Plot risk-free asset
    plt.plot(0, risk_free_rate, 'ko', markersize=10, label=f'Risk-Free Asset ({risk_free_rate:.2%})')
    
    # Plot Capital Market Line (CML) - line through risk-free rate and tangent portfolio
    max_x = max(eff_vols.max(), optimal_portfolio_volatility) * 1.1
    cml_x = np.array([0, max_x])
    cml_y = risk_free_rate + (optimal_portfolio_return - risk_free_rate) / optimal_portfolio_volatility * cml_x
    plt.plot(cml_x, cml_y, 'g--', linewidth=2, alpha=0.7, label='Capital Market Line (CML)')
    
    # Plot individual stocks
    for i, ticker in enumerate(tickers):
        stock_return = expected_returns(np.eye(len(tickers))[i], returns)
        stock_vol = standard_deviation(np.eye(len(tickers))[i], cov_matrix)
        plt.plot(stock_vol, stock_return, 'o', markersize=8, alpha=0.6)
        plt.annotate(ticker, (stock_vol, stock_return), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier with Capital Market Line', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # create a file to display the chart
    file_path = 'efficient_frontier.png'
    print(f"Saving plot to: {file_path}")
    plt.savefig(file_path, dpi = 300, bbox_inches = 'tight')


    plot_portfolio_weights(optimal_weights, tickers, 'optimal_weights.png')
    portfolioo_df = prepare_portfolio_data()
    plot_industry_weights(optimal_weights, tickers, portfolioo_df, 'industry_weights.png')
    plot_sector_weights(optimal_weights, tickers, portfolioo_df, 'sector_weights.png')

if __name__ == "__main__":
    main()
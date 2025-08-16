import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt

def optimize_portfolio(returns_df, tsla_forecast_returns):
    """
    Calculates expected returns and covariance, then optimizes the portfolio.
    """
    # Historical returns for BND and SPY
    historical_returns = returns_df[['BND', 'SPY']]

    # Calculate expected returns (using historical for BND/SPY, forecast for TSLA)
    mu = expected_returns.mean_historical_return(historical_returns, frequency=252)
    mu['TSLA'] = tsla_forecast_returns.mean() * 252 # Annualize forecast mean

    # Covariance matrix based on historical data
    S = risk_models.sample_cov(returns_df, frequency=252)

    ef = EfficientFrontier(mu, S)

    # Max Sharpe Ratio Portfolio
    max_sharpe_weights = ef.max_sharpe()
    cleaned_max_sharpe_weights = ef.clean_weights()
    max_sharpe_stats = ef.portfolio_performance(verbose=True)

    # Minimum Volatility Portfolio
    min_vol_weights = ef.min_volatility()
    cleaned_min_vol_weights = ef.clean_weights()
    min_vol_stats = ef.portfolio_performance(verbose=True)

    return ef, cleaned_max_sharpe_weights, max_sharpe_stats, cleaned_min_vol_weights, min_vol_stats

def plot_efficient_frontier(ef):
    """
    Generates and plots the Efficient Frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

    # Find and plot the maximum Sharpe and minimum volatility portfolios
    ef.max_sharpe()
    max_sharpe_ret, max_sharpe_vol, _ = ef.portfolio_performance()
    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', s=200, c='red', label='Maximum Sharpe')

    ef.min_volatility()
    min_vol_ret, min_vol_vol, _ = ef.portfolio_performance()
    ax.scatter(min_vol_vol, min_vol_ret, marker='^', s=200, c='blue', label='Minimum Volatility')

    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility (Annualized)")
    ax.set_ylabel("Return (Annualized)")
    ax.legend()
    plt.tight_layout()
    plt.savefig('images/efficient_frontier.png')
    plt.show()
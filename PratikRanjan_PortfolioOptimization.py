import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.dates as mdates
from tabulate import tabulate

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("==== MARKOWITZ PORTFOLIO OPTIMIZATION ====")
print("\n1. ASSET SELECTION AND DATA PREPARATION")

# Define time periods
training_start = '2019-04-01'
training_end = '2022-03-31'
testing_start = '2022-04-01'
testing_end = '2025-03-31'

# Initial investment amount in INR
initial_investment = 100000

# Asset selection from different sectors
assets = {
    'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking',
    'INFY.NS': 'IT',
    'TCS.NS': 'IT',
    'RELIANCE.NS': 'Energy',
    'ITC.NS': 'FMCG',
    'ASIANPAINT.NS': 'Paint',
    'SUNPHARMA.NS': 'Pharma',
    'BHARTIARTL.NS': 'Telecom',
    'TITAN.NS': 'Consumer Durables'
}

print(f"Selected {len(assets)} assets from diverse sectors:")
for asset, sector in assets.items():
    print(f"- {asset}: {sector}")

# Download training data
print(f"\nDownloading training data ({training_start} to {training_end})...")
training_data = yf.download(list(assets.keys()), start=training_start, end=training_end, auto_adjust=True)['Close']
training_returns = training_data.pct_change().dropna()

# Download testing data
print(f"Downloading testing data ({testing_start} to {testing_end})...")
testing_data = yf.download(list(assets.keys()), start=testing_start, end=testing_end, auto_adjust=True)['Close']
testing_returns = testing_data.pct_change().dropna()

# Calculate annualized metrics for individual assets
trading_days = 252
annual_returns = training_returns.mean() * trading_days
annual_volatility = training_returns.std() * np.sqrt(trading_days)
sharpe_ratios = annual_returns / annual_volatility

# Display asset metrics in descending order of Sharpe ratio
asset_metrics = pd.DataFrame({
    'Sector': pd.Series(assets),
    'Annual Return': annual_returns,
    'Annual Volatility': annual_volatility,
    'Sharpe Ratio': sharpe_ratios
})
asset_metrics = asset_metrics.sort_values('Sharpe Ratio', ascending=False)

print("\nIndividual Asset Metrics (Training Period):")
print(tabulate(asset_metrics.round(4), headers='keys', tablefmt='pretty', floatfmt=".4f"))

# Correlation matrix
correlation_matrix = training_returns.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Asset Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')

print("\n2. PORTFOLIO OPTIMIZATION")
print("\nPerforming Markowitz Mean-Variance Optimization...")

# Calculate covariance matrix
cov_matrix = training_returns.cov() * trading_days

# Utility functions for portfolio calculations
def portfolio_return(weights, returns):
    return np.sum(weights * returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_sharpe(weights, returns, cov_matrix, risk_free_rate=0.05):
    port_return = portfolio_return(weights, returns)
    port_volatility = portfolio_volatility(weights, cov_matrix)
    return (port_return - risk_free_rate) / port_volatility

def negative_sharpe(weights, returns, cov_matrix, risk_free_rate=0.05):
    return -portfolio_sharpe(weights, returns, cov_matrix, risk_free_rate)

# Constraints and bounds
num_assets = len(assets)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum of weights = 1
bounds = tuple((0, 1) for _ in range(num_assets))  # no short selling

# Initial guess (equal weights)
initial_weights = np.array([1/num_assets] * num_assets)

# 1. Maximum Sharpe Ratio Portfolio (Optimal Risk-Adjusted Return)
max_sharpe_result = minimize(
    negative_sharpe, 
    initial_weights, 
    args=(annual_returns, cov_matrix), 
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
max_sharpe_weights = max_sharpe_result['x']

# 2. Minimum Volatility Portfolio (Lowest Risk)
min_vol_result = minimize(
    portfolio_volatility, 
    initial_weights, 
    args=(cov_matrix,), 
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
min_vol_weights = min_vol_result['x']

# Generate efficient frontier
target_returns = np.linspace(min(annual_returns), max(annual_returns), 100)
efficient_portfolios = []

for target in target_returns:
    # Constraints: weights sum to 1 and portfolio return = target
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_return(x, annual_returns) - target}
    )
    
    # Optimize for minimum volatility given the target return
    result = minimize(
        portfolio_volatility, 
        initial_weights, 
        args=(cov_matrix,), 
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result['success']:
        volatility = portfolio_volatility(result['x'], cov_matrix)
        efficient_portfolios.append((target, volatility, result['x']))

# Convert to DataFrame for easier handling
efficient_df = pd.DataFrame([
    (target, vol, *weights) for target, vol, weights in efficient_portfolios
], columns=['Return', 'Volatility'] + list(assets.keys()))

# Calculate metrics for optimized portfolios
max_sharpe_return = portfolio_return(max_sharpe_weights, annual_returns)
max_sharpe_volatility = portfolio_volatility(max_sharpe_weights, cov_matrix)
max_sharpe_sharpe = (max_sharpe_return - 0.05) / max_sharpe_volatility

min_vol_return = portfolio_return(min_vol_weights, annual_returns)
min_vol_volatility = portfolio_volatility(min_vol_weights, cov_matrix)
min_vol_sharpe = (min_vol_return - 0.05) / min_vol_volatility

# Create portfolio weights DataFrames
max_sharpe_portfolio = pd.Series(max_sharpe_weights, index=assets.keys())
min_vol_portfolio = pd.Series(min_vol_weights, index=assets.keys())

# Display optimized portfolios
print("\nMaximum Sharpe Ratio Portfolio (Optimal Risk-Adjusted Return):")
print(f"Expected Annual Return: {max_sharpe_return:.4f} ({max_sharpe_return:.2%})")
print(f"Expected Annual Volatility: {max_sharpe_volatility:.4f} ({max_sharpe_volatility:.2%})")
print(f"Sharpe Ratio: {max_sharpe_sharpe:.4f}")
print("\nOptimal Weights:")
for asset, weight in max_sharpe_portfolio.sort_values(ascending=False).items():
    if weight > 0.001:  # Only show non-negligible weights
        print(f"- {asset}: {weight:.4f} ({weight:.2%})")

print("\nMinimum Volatility Portfolio (Lowest Risk):")
print(f"Expected Annual Return: {min_vol_return:.4f} ({min_vol_return:.2%})")
print(f"Expected Annual Volatility: {min_vol_volatility:.4f} ({min_vol_volatility:.2%})")
print(f"Sharpe Ratio: {min_vol_sharpe:.4f}")
print("\nOptimal Weights:")
for asset, weight in min_vol_portfolio.sort_values(ascending=False).items():
    if weight > 0.001:  # Only show non-negligible weights
        print(f"- {asset}: {weight:.4f} ({weight:.2%})")

# Plot the efficient frontier with asset and optimized portfolios
plt.figure(figsize=(12, 8))

# Individual assets
plt.scatter(annual_volatility, annual_returns, s=100, c='blue', alpha=0.7, label='Individual Assets')

# Add asset labels
for i, asset in enumerate(assets.keys()):
    plt.annotate(asset, 
                 (annual_volatility[i], annual_returns[i]), 
                 xytext=(5, 5), 
                 textcoords='offset points',
                 fontsize=8)

# Efficient frontier
plt.plot(efficient_df['Volatility'], efficient_df['Return'], 'r-', linewidth=2, label='Efficient Frontier')

# Maximum Sharpe portfolio
plt.scatter(max_sharpe_volatility, max_sharpe_return, s=150, c='green', marker='*', 
            label=f'Maximum Sharpe Ratio Portfolio (SR: {max_sharpe_sharpe:.4f})')

# Minimum volatility portfolio
plt.scatter(min_vol_volatility, min_vol_return, s=150, c='orange', marker='D',
            label=f'Minimum Volatility Portfolio (Vol: {min_vol_volatility:.4f})')

plt.xlabel('Annual Volatility')
plt.ylabel('Annual Expected Return')
plt.title('Efficient Frontier with Optimized Portfolios', fontsize=16, pad=20)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')

# Pie charts for optimal portfolios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Filter out negligible weights for cleaner visualization
max_sharpe_filtered = max_sharpe_portfolio[max_sharpe_portfolio > 0.005]
if max_sharpe_filtered.sum() < 1:
    max_sharpe_filtered['Others'] = 1 - max_sharpe_filtered.sum()
    
min_vol_filtered = min_vol_portfolio[min_vol_portfolio > 0.005]
if min_vol_filtered.sum() < 1:
    min_vol_filtered['Others'] = 1 - min_vol_filtered.sum()

# Maximum Sharpe Portfolio pie chart
ax1.pie(max_sharpe_filtered, autopct='%1.1f%%', startangle=90, shadow=False, 
       explode=[0.05]*len(max_sharpe_filtered), textprops={'fontsize': 10})
ax1.set_title('Maximum Sharpe Ratio Portfolio Allocation', fontsize=14, pad=20)
ax1.legend(max_sharpe_filtered.index, loc="best", bbox_to_anchor=(0.05, 0.05, 0.9, 0.9))

# Minimum Volatility Portfolio pie chart
ax2.pie(min_vol_filtered, autopct='%1.1f%%', startangle=90, shadow=False, 
       explode=[0.05]*len(min_vol_filtered), textprops={'fontsize': 10})
ax2.set_title('Minimum Volatility Portfolio Allocation', fontsize=14, pad=20)
ax2.legend(min_vol_filtered.index, loc="best", bbox_to_anchor=(0.05, 0.05, 0.9, 0.9))

plt.tight_layout()
plt.savefig('portfolio_allocation.png', dpi=300, bbox_inches='tight')

print("\n3. BACKTESTING")
print("\nBacktesting optimized portfolios using testing period data...")

# Function to calculate portfolio value over time
def calculate_portfolio_value(initial_investment, weights, price_data):
    # Calculate number of shares for each asset
    initial_prices = price_data.iloc[0]
    shares = {}
    
    for asset, weight in weights.items():
        if asset in price_data.columns and weight > 0:
            investment_per_asset = initial_investment * weight
            shares[asset] = investment_per_asset / initial_prices[asset]
    
    # Calculate daily portfolio value
    portfolio_values = pd.Series(index=price_data.index, dtype=float)
    
    for date, prices in price_data.iterrows():
        daily_value = sum(shares.get(asset, 0) * prices[asset] for asset in shares)
        portfolio_values[date] = daily_value
        
    return portfolio_values

# Calculate portfolio values during testing period
max_sharpe_values = calculate_portfolio_value(initial_investment, max_sharpe_portfolio, testing_data)
min_vol_values = calculate_portfolio_value(initial_investment, min_vol_portfolio, testing_data)

# Equal-weighted portfolio for comparison
equal_weights = pd.Series(1/num_assets, index=assets.keys())
equal_values = calculate_portfolio_value(initial_investment, equal_weights, testing_data)

# Calculate cumulative returns
max_sharpe_returns = max_sharpe_values.pct_change().fillna(0)
min_vol_returns = min_vol_values.pct_change().fillna(0)
equal_returns = equal_values.pct_change().fillna(0)

# Calculate performance metrics for the testing period
def calculate_performance_metrics(returns, values, risk_free_rate=0.05/252):
    # Daily metrics
    daily_return = returns.mean()
    daily_volatility = returns.std()
    
    # Annualized metrics
    annual_return = daily_return * trading_days
    annual_volatility = daily_volatility * np.sqrt(trading_days)
    sharpe_ratio = (annual_return - risk_free_rate*trading_days) / annual_volatility
    
    # Total return
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    
    # Final value
    final_value = values.iloc[-1]
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Total Return': total_return,
        'Final Value (INR)': final_value
    }

# Calculate metrics for each portfolio
max_sharpe_metrics = calculate_performance_metrics(max_sharpe_returns, max_sharpe_values)
min_vol_metrics = calculate_performance_metrics(min_vol_returns, min_vol_values)
equal_metrics = calculate_performance_metrics(equal_returns, equal_values)

# Create performance comparison DataFrame
performance_comparison = pd.DataFrame({
    'Maximum Sharpe Portfolio': max_sharpe_metrics,
    'Minimum Volatility Portfolio': min_vol_metrics,
    'Equal-Weight Portfolio': equal_metrics
})

# Display performance comparison
print("\nPortfolio Performance (Testing Period):")
print(tabulate(performance_comparison.T.round(4), headers='keys', tablefmt='pretty', floatfmt=".4f"))

# Plot portfolio values over time
plt.figure(figsize=(12, 8))
plt.plot(max_sharpe_values.index, max_sharpe_values, 'g-', linewidth=2, label='Maximum Sharpe Portfolio')
plt.plot(min_vol_values.index, min_vol_values, 'b-', linewidth=2, label='Minimum Volatility Portfolio')
plt.plot(equal_values.index, equal_values, 'k--', linewidth=1.5, label='Equal-Weight Portfolio')

plt.title('Portfolio Value Over Time (Testing Period)', fontsize=16, pad=20)
plt.xlabel('Date')
plt.ylabel('Portfolio Value (INR)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')

# Plot rolling metrics
window = 30  # 30-day rolling window

plt.figure(figsize=(18, 12))

# Rolling Volatility
rolling_vol_max_sharpe = max_sharpe_returns.rolling(window=window).std() * np.sqrt(trading_days)
rolling_vol_min_vol = min_vol_returns.rolling(window=window).std() * np.sqrt(trading_days)
rolling_vol_equal = equal_returns.rolling(window=window).std() * np.sqrt(trading_days)

plt.plot(rolling_vol_max_sharpe.index, rolling_vol_max_sharpe, 'g-', label='Maximum Sharpe Portfolio', linewidth=2)
plt.plot(rolling_vol_min_vol.index, rolling_vol_min_vol, 'b-', label='Minimum Volatility Portfolio', linewidth=2)
plt.plot(rolling_vol_equal.index, rolling_vol_equal, 'k--', label='Equal-Weight Portfolio', linewidth=1.5)
plt.title(f'{window}-Day Rolling Annualized Volatility', fontsize=14, pad=15)
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_metrics.png', dpi=300, bbox_inches='tight')


print("\n4. SUMMARY AND CONCLUSIONS")

# Summary of results
print("\nSummary of Portfolio Optimization Results:")
best_portfolio = "Maximum Sharpe" if max_sharpe_metrics["Sharpe Ratio"] > min_vol_metrics["Sharpe Ratio"] else "Minimum Volatility"
print(f"1. Best Performing Portfolio: {best_portfolio} Portfolio")
print(f"2. Total Return of Max Sharpe Portfolio: {max_sharpe_metrics['Total Return']:.4f} ({max_sharpe_metrics['Total Return']:.2%})")
print(f"3. Final Portfolio Value (Max Sharpe): INR {max_sharpe_metrics['Final Value (INR)']:.2f}")

print("\nPortfolio optimization completed successfully!")
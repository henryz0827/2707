import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Download SPY data from Yahoo Finance
spy_data = yf.download("SPY", start='1993-02-01', end='2023-10-21')

# printing the top 10 rows
print(spy_data.head(10))

# Calculate daily returns
returns = spy_data['Adj Close'].pct_change().dropna()

# Define the Momentum Strategy Function (Dual Moving Average Crossover)
def momentum_strategy(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['short_mavg'] = data.rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data.rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, -1.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Define a Rate of Change (ROC) Strategy
def roc_strategy(data, window):
    roc = data.pct_change(periods=window)
    signals = pd.DataFrame(index=data.index)
    signals['roc'] = roc
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['roc'] > 0, 1.0, -1.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Logistic Regression Direction Classifier Function
def logistic_regression_direction_classifier(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    return predictions, score

# Function to calculate Cumulative Returns
def calculate_cumulative_returns(data):
    cumulative_returns = (1 + data.pct_change()).cumprod() - 1
    return cumulative_returns

# Drawdowns Calculation

# Function to calculate drawdowns from cumulative returns
def calculate_drawdowns(cumulative_returns):
    # Calculate the running maximum
    running_max = cumulative_returns.cummax()
    # Calculate drawdowns as the difference from the running maximum
    drawdowns = (cumulative_returns - running_max) / running_max
    # Replace NaNs with 0 for the initial part and negative infinity with large negative value
    drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).fillna(0)
    return drawdowns


# Apply Multiple Strategies
short_window = 40
long_window = 100
roc_window = 20
spy_momentum = momentum_strategy(spy_data['Adj Close'], short_window, long_window)
spy_roc = roc_strategy(spy_data['Adj Close'], roc_window)

# Combine Strategies
combined_signal = spy_momentum['signal'] + spy_roc['signal']
combined_signal[combined_signal > 1] = 1
combined_signal[combined_signal < -1] = -1

# Risk Management: Implement a Stop-Loss Strategy
stop_loss_threshold = -0.02  # 2% drop as a threshold
spy_data['daily_return'] = spy_data['Adj Close'].pct_change()
spy_data['cum_return'] = (1 + spy_data['daily_return']).cumprod() - 1
stop_loss_signal = np.where(spy_data['cum_return'] < stop_loss_threshold, -1, 1)

# Incorporate Stop-Loss into Combined Strategy
final_signal = combined_signal * stop_loss_signal
final_signal[final_signal > 1] = 1
final_signal[final_signal < -1] = -1

# Sharpe Ratio Calculation
risk_free_rate = 0.01

# Convert the annual risk-free rate to a daily rate
daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

# Calculate the daily excess returns
daily_excess_returns = returns - daily_risk_free_rate

# daily Sharpe ratio
sharpe_ratio_daily = daily_excess_returns.mean() / daily_excess_returns.std()

# Annualize the Sharpe ratio by multiplying by the square root of the number of trading days
sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(252)  # Assuming 252 trading days in a year




# Alpha Calculation
market_returns = spy_data['Adj Close'].pct_change().mean()

# Calculate Strategy Returns
spy_data['strategy_return'] = market_returns * final_signal.shift()

# Cumulative Strategy Returns
spy_data['cumulative_strategy_return'] = (1 + spy_data['strategy_return']).cumprod()
strategy_return = spy_data['strategy_return']
alpha = strategy_return.mean() - risk_free_rate - (1 * (market_returns - 0.02))  # Beta is 1

sharpe_ratio = (strategy_return.mean() - risk_free_rate) / strategy_return.std()


# Transaction Costs Simulation
transaction_cost_per_trade = 0.0005  
total_trades = np.abs(final_signal.diff()).sum()
total_transaction_costs = total_trades * transaction_cost_per_trade
net_returns = strategy_return.sum() - total_transaction_costs

# Plot the strategy
plt.figure(figsize=(12, 8))
plt.plot(spy_data['Adj Close'], label='SPY Adj Close', alpha=0.5)
plt.plot(spy_momentum['short_mavg'], label='40-day SMA', alpha=0.5)
plt.plot(spy_momentum['long_mavg'], label='100-day SMA', alpha=0.5)
plt.plot(final_signal, label='Final Signal', alpha=0.5)
plt.title('SPY Combined Momentum Strategy')
plt.legend()
plt.show()

# Print summary results
print(f"Sharpe Ratio: {sharpe_ratio_annualized}")
print(f"Alpha: {alpha}")
print(f"Total Transaction Costs: {total_transaction_costs}")
print(f"Net Returns after Costs: {net_returns}")

# Logistic Regression Classifier
# This section can be expanded based on specific analysis needs
features = spy_data[['Adj Close']].pct_change().dropna()
target = strategy_return.shift(-1).dropna() > 0  # Predicting next day's movement

# Ensure features and target have the same length
min_length = min(len(features), len(target))
features, target = features.iloc[:min_length], target.iloc[:min_length]

logistic_predictions, logistic_score = logistic_regression_direction_classifier(features, target)

print(f"Logistic Regression Score: {logistic_score}")

# Apply additional financial analysis
cumulative_returns = calculate_cumulative_returns(spy_data['Adj Close'])
#max_drawdown = calculate_max_drawdown(cumulative_returns)
# Calculate the drawdowns
drawdowns = calculate_drawdowns(cumulative_returns)

# Display additional financial analysis results
print(f"Cumulative Returns: {cumulative_returns[-1]}")
print(f"Max Drawdown: {drawdowns}")

# Exploratory Analysis and Parameter Testing
# Testing different parameter values for the momentum strategy
parameter_test_results = []
for short_window_test in range(20, 61, 10):
    for long_window_test in range(80, 141, 20):
        test_strategy = momentum_strategy(spy_data['Adj Close'], short_window_test, long_window_test)
        test_sharpe_ratio = (test_strategy['signal'].mean() - risk_free_rate) / test_strategy['signal'].std()
        parameter_test_results.append((short_window_test, long_window_test, test_sharpe_ratio))

# Display parameter test results
for result in parameter_test_results:
    print(f"Short Window: {result[0]}, Long Window: {result[1]}, Sharpe Ratio: {result[2]}")

# Further Visualization
# Plot Cumulative Returns and Drawdowns
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(cumulative_returns, label='Cumulative Returns')
plt.title('Cumulative Returns over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(drawdowns, label='Drawdowns')
plt.title('Drawdowns over Time')
plt.legend()
plt.tight_layout()
plt.show()


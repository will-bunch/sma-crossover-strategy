# S&P 500 SMA Crossover Strategy

## Overview

This project investigates whether Simple Moving Average (SMA) crossover trading strategies can outperform a buy-and-hold portfolio of S&P 500 stocks.

Historical price data is downloaded using Yahoo Finance and used to test multiple SMA parameter combinations across a large universe of stocks. A rolling train-test framework is used to optimise parameters out-of-sample and evaluate portfolio performance.

## Methodology

### Data Collection

* S&P 500 constituents sourced from Wikipedia
* Historical price data downloaded using Yahoo Finance
* Stocks with incomplete price histories removed to ensure consistency

### Strategy

* Short-term SMA windows: 20–60 days
* Long-term SMA windows: 180–280 days
* Long position when short SMA exceeds long SMA
* Short position when short SMA falls below long SMA

### Optimisation

* Grid search performed across all SMA combinations
* Best-performing parameter set selected for each stock using training data
* Performance evaluated on unseen test data

### Portfolio Construction

* Equal-weight portfolio of all selected stocks
* Rolling train-test windows used to reduce overfitting
* Buy-and-hold benchmark used for comparison

### Extension

A volatility-filtered version of the strategy was also tested. Positions are reduced when market volatility exceeds a predefined threshold.

## Performance Metrics

The following metrics are calculated:

* Annualised Return
* Annualised Volatility
* Sharpe Ratio
* Maximum Drawdown
* Relative Performance vs Buy-and-Hold

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* yFinance

## Key Findings

The SMA crossover strategy demonstrated different behaviour across market environments. While buy-and-hold generally performed better during strong bull markets, the SMA-based approaches showed improved downside protection during periods of market stress.

The volatility-filtered strategy was included to explore whether risk-adjusted performance could be improved by avoiding high-volatility market conditions.

## Limitations

* No transaction costs included
* No slippage assumptions
* Equal-weight portfolio construction
* Historical performance does not guarantee future results

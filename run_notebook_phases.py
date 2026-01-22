#!/usr/bin/env python3
"""
Run the notebook in phases to show outputs progressively.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

print("=" * 80)
print("PHASE 1: DATA SETUP")
print("=" * 80)

# 1.1 Download Data
print("\n[Step 1.1] Downloading price data...")
tickers = ['SPY', 'QQQ', 'EFA', 'TLT', 'LQD', 'GLD', 'DBC']
start_date = '2010-01-01'
end_date = '2025-12-31'

print(f"  Tickers: {tickers}")
print(f"  Period: {start_date} to {end_date}")

try:
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close']

    print(f"\n  Downloaded data shape: {prices.shape}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Missing values per asset:")
    for ticker in tickers:
        missing = prices[ticker].isna().sum()
        print(f"    {ticker}: {missing} ({missing/len(prices)*100:.2f}%)")

    # Handle missing data
    prices = prices.fillna(method='ffill').fillna(method='bfill')

    print(f"\n  After filling missing values: {prices.isna().sum().sum()} NaN values")

except Exception as e:
    print(f"  ERROR downloading data: {e}")
    sys.exit(1)

# 1.2 Calculate Returns
print("\n[Step 1.2] Calculating returns...")
returns = prices.pct_change().dropna()

# Clip extreme returns to avoid numerical issues
returns = returns.clip(lower=-0.99)

print(f"  Returns shape: {returns.shape}")
print(f"  Returns statistics:")
print(returns.describe())

# 1.3 Split data
print("\n[Step 1.3] Splitting data into train/val/test...")
# Split dates
total_days = len(returns)
train_end_idx = int(total_days * 0.6)
val_end_idx = int(total_days * 0.8)

returns_train = returns.iloc[:train_end_idx]
returns_val = returns.iloc[train_end_idx:val_end_idx]
returns_test = returns.iloc[val_end_idx:]

print(f"  Training set: {returns_train.index[0].date()} to {returns_train.index[-1].date()} ({len(returns_train)} days)")
print(f"  Validation set: {returns_val.index[0].date()} to {returns_val.index[-1].date()} ({len(returns_val)} days)")
print(f"  Test set: {returns_test.index[0].date()} to {returns_test.index[-1].date()} ({len(returns_test)} days)")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE - Data loaded and split")
print("=" * 80)

# Save data for next phases
print("\nSaving data to disk for next phases...")
returns_train.to_pickle('returns_train.pkl')
returns_val.to_pickle('returns_val.pkl')
returns_test.to_pickle('returns_test.pkl')
prices.to_pickle('prices.pkl')

print("\nPhase 1 outputs saved:")
print("  - returns_train.pkl")
print("  - returns_val.pkl")
print("  - returns_test.pkl")
print("  - prices.pkl")
print("\nNext: Run Phase 2 (GARCH and CVaR-RP)")

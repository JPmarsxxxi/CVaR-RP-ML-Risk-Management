# CVaR-RP + ML Risk Management -- Full Notebook Breakdown

## Overall Flow

The notebook implements a 3-phase pipeline:

```
Phase 1: Data Setup → Phase 2: Baseline CVaR-RP Portfolio → Phase 3: ML-Enhanced Portfolio
```

The core idea: start with a **CVaR Risk Parity** portfolio (where every asset contributes equally to tail risk), then use **machine learning** to predict which assets will go up or down, and **tilt the weights** accordingly -- zeroing out assets predicted to decline.

---

## PHASE 1: DATA SETUP (Cells 0-10)

### Cell 0 [Markdown] -- Title & Overview

Sets the stage. Lists the 5 phases (Data Setup, Baseline CVaR-RP, ML Layer, Backtesting, Robustness) and the 7 assets: SPY, QQQ, EFA (international equity), TLT (long bonds), LQD (corporate bonds), GLD (gold), DBC (commodities). This is a deliberately diversified mix spanning equities, fixed income, and real assets.

### Cell 1 [Markdown] -- Environment Setup Instructions

Tells you to create a conda environment with Python 3.10. This is important because TensorFlow and some other dependencies have version-specific requirements.

### Cell 2 [Code] -- Package Installation

Installs all dependencies via `pip`: numpy, pandas, yfinance, scipy, arch, scikit-learn, xgboost, tensorflow, matplotlib, seaborn, tqdm, numba, vectorbt, numexpr, joblib. Then verifies every import works.

**Problem addressed:** Dependency conflicts between TensorFlow and other packages. The cell pins to Python 3.10 and installs everything in one shot to avoid version mismatches.

### Cell 3 [Markdown] -- Speed Optimizations Documentation

Documents that the raw implementation would take ~2 hours, so four optimizations were added:
- **Numba JIT** for the CVaR-RP inner loop (33x faster)
- **VectorBT** for backtesting (60x faster)
- **joblib** for parallel ML training (6x faster)
- **NumExpr** for matrix math (2x faster)

**Problem addressed:** The CVaR-RP iterative solver runs thousands of iterations per rebalancing date, across many dates. Without JIT compilation, this is painfully slow in pure Python.

### Cell 4 [Code] -- Library Imports

Imports everything needed: numpy, pandas, yfinance, scipy.stats (for the normal distribution CDF/PDF used in the CVaR formula), arch (GARCH models), all the sklearn classifiers and utilities, xgboost, TensorFlow/Keras layers (LSTM, SimpleRNN, Conv1D, Dense, Dropout, MultiHeadAttention, LayerNormalization), matplotlib, seaborn, tqdm, and suppresses warnings.

### Cell 5 [Markdown] -- Phase 1 Header

Describes the data download: 7 ETFs, daily OHLCV, from 2010-01-01 to 2025-12-31 (~15 years).

### Cell 6 [Code] -- Download Price Data (Phase 1.1)

Downloads daily historical data for all 7 tickers using `yf.Ticker(...).history()`. Organizes data into 5 aligned DataFrames: `closes`, `opens`, `highs`, `lows`, `volumes`. Reports shape and missing values.

**Problem addressed:** yfinance sometimes returns data with different date ranges per ticker (some ETFs launched at different times). The code handles this by aligning on the date index and reporting any gaps.

### Cell 7 [Markdown] -- Empty (spacer)

### Cell 8 [Code] -- Data Preprocessing (Phase 1.2)

This cell does four critical things:
1. Computes **daily returns** via `pct_change()` on closing prices
2. Handles **missing data** with forward-fill (weekends, holidays, gaps)
3. Creates **binary labels**: `1` if return > 0, `0` otherwise -- these are the ML classification targets
4. Performs a **time-based train/val/test split** (60%/20%/20%) using `TimeSeriesSplit`

**Problem addressed:** A naive random split would cause **look-ahead bias** (training on future data). The time-based split ensures the model only ever trains on past data, which is essential for any financial ML application.

### Cell 9 [Markdown] -- EDA Header

### Cell 10 [Code] -- Exploratory Data Analysis (Phase 1.3)

Generates 8 diagnostic plots, each saved as a PNG:
1. **Normalized price evolution** -- all assets rebased to 1 for comparison
2. **Return distributions** -- histograms + KDE + normal overlay, showing fat tails
3. **Correlation matrix heatmap** -- Pearson correlations of daily returns
4. **Rolling 30-day volatility** -- shows volatility clustering over time
5. **Cumulative returns** -- compounded growth per asset
6. **Box plots** -- spread, median, outliers per asset
7. **Summary statistics** -- risk-return scatter, Sharpe ratios, skewness, kurtosis
8. **Drawdown analysis** -- peak-to-trough declines

**Why this matters:** The correlation matrix directly informs whether CVaR-RP diversification will work. Low/negative correlations between assets (e.g., SPY vs TLT) mean risk parity can meaningfully reduce portfolio risk. The fat tails visible in the return distributions justify using CVaR (which focuses on tail risk) rather than simple variance.

---

## PHASE 2: BASELINE CVaR-RP PORTFOLIO (Cells 11-19)

### Cell 11 [Markdown] -- Empty (spacer)

### Cell 12 [Code] -- GARCH(1,1) Volatility Estimation (Phase 2.1)

This is the foundation of the CVaR-RP optimizer: **forecasting forward-looking risk**.

For each asset, a **GARCH(1,1)** model is fit on monthly returns using a rolling window of 3 months. The variance equation:

```
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```

This captures **volatility clustering** (big moves follow big moves). The model produces a 1-month-ahead variance forecast.

The covariance matrix is then built as: **Omega = D x R x D**, where D is the diagonal matrix of GARCH-forecasted standard deviations and R is the rolling correlation matrix. This is a simplified DCC (Dynamic Conditional Correlation) approach.

**Problems addressed:**
- Using **monthly** data for GARCH, not daily. Daily GARCH on a small rolling window is noisy and often fails to converge. The cell header explicitly says "CORRECTED - Monthly Data," meaning an earlier version likely tried daily and ran into convergence issues.
- The rolling window of only 3 months is very short -- this is a deliberate choice to make the model adaptive to changing market conditions, at the cost of less stable estimates.

### Cell 13 [Markdown] -- CVaR Formula Explanation

Documents the CVaR calculation under a normal distribution assumption:

```
CVaR_alpha = mu + sigma * phi(Phi^{-1}(alpha)) / alpha
```

where alpha = 0.05 (5% tail). This is the analytical formula that avoids Monte Carlo simulation.

### Cell 14 [Code] -- Numba JIT CVaR-RP Core

Defines `_cvar_rp_core()`, the inner loop of the CVaR-RP optimizer, compiled with `@njit` (Numba's no-Python JIT). This function:
1. Starts with equal weights (1/N)
2. Computes each asset's **Marginal Risk Contribution (MRC)** beta (Equation 3 from the paper)
3. Updates weights: `x_i = (1/beta_i) / sum(1/beta_j)` -- assets with higher risk contribution get lower weights
4. Checks convergence via RMSE of weight changes

**Problem addressed:** This loop runs up to 3000 iterations per rebalancing date, across potentially hundreds of dates. Pure Python would make this a bottleneck. Numba compiles it to machine code for a ~33x speedup.

### Cell 15 [Code] -- Portfolio CVaR Calculation (Phase 2.2)

Implements two approaches to computing portfolio CVaR:
1. **Quantile Regression** (paper's preferred method): Uses `statsmodels.QuantReg` to estimate VaR non-parametrically, then derives CVaR.
2. **Parametric Normal** (fallback): Uses the closed-form formula from Cell 13.

**Problem addressed:** Quantile regression can fail to converge in some periods (especially with short windows or extreme market conditions). The fallback ensures the pipeline never crashes -- it just uses the simpler normal assumption when the more sophisticated method fails.

### Cell 16 [Markdown] -- CVaR-RP Optimization Description

Describes the iterative algorithm: equal-weight initialization, MRC beta calculation, inverse-beta weight update, convergence at RMSE < 1e-6 or 3000 iterations. References Equations 3, 8, 9, 10 from the paper.

### Cell 17 [Code] -- CVaR-RP Optimization (Phase 2.3)

The full optimizer, wrapping the Numba core. Three parts:
1. **`optimize_cvar_rp()`** -- takes a mean vector and covariance matrix, returns risk-parity weights
2. **Training-period test** -- runs on the full training sample to verify convergence
3. **Rolling out-of-sample** -- at each month in the val+test period, feeds in the GARCH-forecasted covariance and solves for weights

The title says "CORRECTED - Following Paper Exactly," suggesting earlier versions had deviations from the paper's algorithm that were fixed.

**Problems addressed:**
- The optimizer must handle **degenerate covariance matrices** (e.g., when assets have near-zero variance or near-perfect correlation). The code includes safeguards for this.
- Convergence is not guaranteed -- the 3000-iteration cap prevents infinite loops.

### Cell 18 [Markdown] -- Empty (spacer)

### Cell 19 [Code] -- Backtest Baseline CVaR-RP (Phase 2.4)

Evaluates the baseline (no ML) portfolio with a realistic backtest:
- **`apply_transaction_costs()`** -- proportional costs based on turnover (10 bps default)
- **`calculate_monthly_returns()`** -- converts daily to monthly using log returns for numerical stability
- **`backtest_cvar_rp()`** -- walk-forward backtest with monthly rebalancing
- **`get_risk_free_rate()`** -- fetches 3-month T-bill rate from Yahoo Finance for Sharpe ratio
- **`calculate_performance_metrics()`** -- computes cumulative return, annualized return/vol, Sharpe, max drawdown, Calmar ratio

**Problems addressed:**
- **Log returns vs simple returns:** The code uses log returns for aggregation (`log(1+r)`) which is more numerically stable when compounding over long periods, then converts back.
- **Risk-free rate:** Rather than hardcoding, it fetches the actual T-bill rate, with a default of 2.5% as fallback if the API call fails.
- This baseline is critical -- it is the benchmark the ML layer must beat.

---

## PHASE 3: MACHINE LEARNING LAYER (Cells 20-27)

### Cell 20 [Markdown] -- Phase 3 Header

Lists the feature categories: price-based, volume, volatility, and technical indicators (RSI, MACD, Bollinger Bands).

### Cell 21 [Code] -- Feature Engineering (Phase 3.1)

Builds the feature matrix for each asset. Helper functions:
- **`calculate_rsi()`** -- 14-day Relative Strength Index (momentum)
- **`calculate_macd()`** -- Moving Average Convergence Divergence (12/26/9 EMA crossover)
- **`calculate_bollinger_bands()`** -- where price sits relative to 2-std bands

**`create_features()`** combines everything per asset:
- Multi-horizon returns (1d, 5d, 20d, 1m, 3m)
- OHLC ratios (Open/Close, High/Close, Low/Close -- from Section 4.4.2 of the paper)
- Moving average ratios (5d, 20d, 60d MA vs price)
- Volume ratio (current / 20d average)
- Rolling volatility (10d, 20d, 30d windows)
- RSI, MACD, Bollinger position

Features are aligned with binary labels and NaN rows (from rolling window warmup) are dropped.

**Problem addressed:** Feature engineering for financial ML is tricky because of **stationarity**. Raw prices are non-stationary, so the code uses returns, ratios, and normalized indicators rather than price levels. This is essential for ML models to generalize.

### Cell 22 [Markdown] -- Model Selection Header

Explains the two-stage approach: first select the best model per asset via cross-validation, then retrain on full in-sample data.

### Cell 23 [Code] -- Data Preparation for ML (Phase 3.2)

Prepares the data for training:
1. **`prepare_sequences()`** -- creates sliding-window 3D arrays for deep learning models (LSTM, RNN, CNN expect `(samples, timesteps, features)`)
2. **Date-based splitting** -- matches the train/val/test boundaries from Phase 1.2
3. **Feature scaling** -- `StandardScaler` fit on training data ONLY, then transform val and test

**Problem addressed:** **Data leakage** is the #1 pitfall in financial ML. If you fit the scaler on the full dataset, information from the future leaks into training. This cell is careful to fit only on training data and store the scalers for later reuse.

### Cell 24 [Code] -- Model Definitions (Phase 3.3)

Defines all 8 classification models:

**Traditional ML:**
1. **Logistic Regression** -- L2 regularized, balanced class weights, LBFGS solver
2. **SVC** -- RBF kernel with probability calibration, balanced weights
3. **Random Forest** -- 100 trees, max depth 10, balanced weights
4. **Gradient Boosting (XGBoost)** -- 100 estimators, learning rate 0.1, max depth 5

**Deep Learning:**
5. **CNN** -- Conv1D(64) -> Conv1D(32) -> Dense(32) -> sigmoid
6. **RNN** -- SimpleRNN(64) -> SimpleRNN(32) -> Dense(16) -> sigmoid
7. **LSTM** -- LSTM(64) -> LSTM(32) -> Dense(16) -> sigmoid
8. **Transformer** -- MultiHeadAttention(4 heads, key_dim=32) -> Feed-Forward -> GlobalAvgPool -> Dense(32) -> sigmoid

All deep learning models use **EarlyStopping** (patience=10) and **ReduceLROnPlateau** to prevent overfitting. Training config: 50 epochs max, batch size 32.

**Problem addressed:** The balanced class weights in traditional models and the binary crossentropy with sigmoid in deep learning models handle **class imbalance** (markets go up slightly more than they go down, roughly 53/47).

### Cell 25 [Code] -- Stage 1 Training (Phase 3.4)

The most compute-intensive cell. For each of the 7 assets, it trains all 8 models:
- **`train_traditional_ml()`** -- fits sklearn/XGBoost models on 2D arrays
- **`train_deep_learning()`** -- trains Keras models on 3D sequences with callbacks
- **`cross_validate_model()`** -- wraps everything in `TimeSeriesSplit` cross-validation (typically 5 folds)

The best model per asset is selected by highest mean CV accuracy. Results go into `stage1_summary`.

**Problems addressed:**
- **Time-series cross-validation** is used instead of standard k-fold. In standard k-fold, fold 3 might train on 2023 data and validate on 2020 data -- that is look-ahead bias. `TimeSeriesSplit` ensures each validation fold is always chronologically after its training fold.
- Some models (especially SVC and deep learning) can be slow, hence the joblib parallelization mentioned in Cell 3.

### Cell 26 [Code] -- Stage 2 Retraining (Phase 3.5)

After identifying the best model per asset, this cell:
1. **Retrains** the winning model on the full in-sample data (train + validation combined) -- this gives the model more data to learn from
2. **Generates test predictions** -- binary (1=UP, 0=DOWN) and probability outputs
3. Creates `predictions_matrix` (dates x assets) and `probabilities_matrix`

**Problem addressed:** The two-stage approach prevents a subtle form of overfitting: if you used the same validation set for both model selection and performance reporting, your reported accuracy would be optimistically biased. By selecting on validation, then evaluating on a separate test set, the results are more honest.

### Cell 27 [Code] -- ML-Enhanced CVaR-RP Portfolio (Phase 3.6)

The payoff of the entire pipeline. This cell combines the CVaR-RP weights with ML predictions:

```
x'_i = x_i * Y_hat_i      (multiply weight by prediction: 0 or 1)
x'_i = x'_i / sum(x'_j)   (re-normalize to sum to 1)
```

If ML predicts DOWN for an asset, its weight goes to **zero** and the remaining weight is redistributed to assets predicted UP. If all assets are predicted DOWN, it defaults to equal weights (a safety fallback).

The cell then:
1. Computes baseline and ML-enhanced portfolio returns over the test period
2. Calculates performance metrics for both strategies
3. Generates 3 comparison plots (cumulative returns, monthly returns, weight evolution)
4. Prints improvement analysis (how much better/worse the ML version did)

**Problem addressed:** The "all predicted DOWN" edge case is handled with the equal-weight fallback. Without it, the portfolio would have zero allocation and produce no returns for that period.

---

## Summary of Key Problems & Solutions

| Problem | Where | Solution |
|---------|-------|----------|
| Slow CVaR-RP convergence loop | Cells 14, 17 | Numba JIT compilation (33x speedup) |
| Look-ahead bias in data splits | Cell 8 | Chronological train/val/test split |
| GARCH failing on daily data | Cell 12 | Switched to monthly aggregation |
| Quantile regression non-convergence | Cell 15 | Parametric normal fallback |
| Data leakage from scaling | Cell 23 | Fit scaler on training data only |
| Non-stationary features | Cell 21 | Returns, ratios, and normalized indicators instead of raw prices |
| Class imbalance (up vs down) | Cell 24 | Balanced class weights, binary crossentropy |
| Model selection overfitting | Cells 25-26 | Two-stage approach (select on CV, retrain on full in-sample) |
| All-DOWN prediction edge case | Cell 27 | Equal-weight fallback |
| Degenerate covariance matrices | Cell 17 | Iteration cap (3000) + convergence tolerance |

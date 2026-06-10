# Option Pricing via FFT

Implementation of the Carr-Madan (1999) FFT-based option pricing framework for European options under various stochastic models, calibrated tolive SPX market data.

## Models implemented
- Black-Scholes (analytical + FFT, used as sanity check)
- Variance Gamma (VG) — per-maturity calibration to SPX IV surface
- Heston stochastic volatility — joint calibration across maturities using the $\phi_2$ formulation (Albrecher et al. 2007) for numerical stability

## Results
Calibration of model parameters by minimizing IV RMSE against SPX mid-price implied volatilities (3-month T-bill rate, 3 maturities). RMSE < 0.002 for VG, RMSE < 0.001 for Heston. Market data fetched live via yfinance.

## Backtesting
Historical SPX options data (Jan–Jun 2026) sourced via the Massive API. Joint Heston calibration run daily across all available maturities, parallelised across trading days using `joblib`.

A volatility trading signal is defined as the difference between Heston-implied IV (using lagged parameters) and market IV, normalized to a per-contract z-score using an expanding window. Positions are taken when |z| > 1, delta hedged using lagged BS delta to avoid lookahead bias.

## Key findings
- Unhedged Sharpe of 4.65, increasing to 6.6 after delta hedging — delta exposure was hurting the raw strategy
- Sensitivity analysis shows Sharpe drops from 6.66 to 2.88 at 1% bid-ask spread, strategy unprofitable above ~1.5%
- Transaction cost model is a simplification — illiquidity and market impact not accounted for

## Roadmap
- Source real bid-ask data to replace modeled spread
- Integration with HAR-RV realized variance forecasts
- Merton jump-diffusion extension
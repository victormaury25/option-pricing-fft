# Option Pricing via FFT

Implementation of the Carr-Madan (1999) FFT-based option pricing framework 
for European options under various stochastic models, calibrated to live SPX 
market data.

## Models implemented
- Black-Scholes (analytical + FFT, used as sanity check)
- Variance Gamma (VG) — calibrated to SPX IV surface across 3 maturities

## Results
Per-maturity calibration of VG parameters (θ, σ, v) by minimizing IV RMSE 
against SPX mid-price implied volatilities. RMSE < 0.002 across all maturities. 
Market data fetched live via yfinance (SPX options + 3-month T-bill rate).

## Roadmap
- Heston stochastic volatility model with branch-cut fix (Gao-Hyndman 2025)
- Integration with HAR-RV realized variance forecasts

If the notebook fails to render on GitHub, view it on [nbviewer](https://nbviewer.org/github/victormaury25/option-pricing-fft/blob/main/FFT_option_pricing.ipynb).
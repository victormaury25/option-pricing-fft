# Option Pricing via FFT

Implementation of the Carr-Madan (1999) FFT-based option pricing framework 
for European options under various stochastic models, calibrated to live SPX 
market data.

## Models implemented
- Black-Scholes (analytical + FFT, used as sanity check)
- Variance Gamma (VG) — per-maturity calibration to SPX IV surface
- Heston stochastic volatility — per-maturity and joint calibration across 
  maturities, using the φ₂ formulation (Albrecher et al. 2007) for numerical 
  stability

## Results
Calibration of model parameters by minimizing IV RMSE against SPX mid-price 
implied volatilities (3-month T-bill rate, 3 maturities). RMSE < 0.002 for VG, 
RMSE < 0.001 for Heston. Market data fetched live via yfinance.

## Backtesting
Historical SPX options data (Jan–Jun 2026) sourced via the Massive API and 
pre-processed by `fetch_data.py`. Joint Heston calibration is run daily across 
all available maturities, parallelised across trading days using `joblib`.

## Roadmap
- Volatility trading signal based on model-implied vs market IV
- Integration with HAR-RV realized variance forecasts

If the notebook fails to render on GitHub, view it on [nbviewer](https://nbviewer.org/github/victormaury25/option-pricing-fft/blob/main/FFT%20option%20pricing.ipynb).
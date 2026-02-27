# Enhanced Binomial Options Pricing Model

A sophisticated **binomial tree options pricing model** with real market data integration, GARCH volatility forecasting, and comprehensive risk metrics. Built for pricing both European and American options across multiple global markets.

![Options Pricing Dashboard](options_pricing_dashboard.pdf)

##  Key Features

- **Real Market Data Integration**: Automatically fetches current stock prices, dividend yields, and market data from Yahoo Finance
- **Multi-Country Support**: Handles US, NZ, European, UK, Australian, and Japanese markets with country-specific risk-free rates
- **Advanced Volatility Estimation**: Implements Historical, EWMA, GARCH(1,1), and Implied volatility methods
- **Complete Greeks Calculation**: Delta, Gamma, Theta, Vega, and Rho using finite differences
- **American & European Options**: Handles both exercise styles with early exercise logic for American options
- **Black-Scholes Validation**: Compares binomial tree results against closed-form solutions
- **Comprehensive Visualization**: 4-panel dashboard showing price history, returns distribution, volatility comparison, and option price surfaces
- **Convergence Analysis**: Shows how price accuracy improves with more time steps

## Performance Highlights

| Metric | GOOGL Example | FBU.NZ Example |
|--------|---------------|----------------|
| **Option Type** | American Put | American Call |
| **Stock Price** | $307.38 | $3.50 |
| **Volatility (GARCH)** | 29.37% | 31.56% |
| **Option Price** | $69.27 | $0.0621 |
| **Delta** | -0.55 | 0.16 |
| **Gamma** | 0.0025 | 0.41 |
| **Black-Scholes Error** | N/A (American) | N/A (American) |

##  Mathematical Foundation

### Binomial Tree Parameters

For each time step Δt = T/N:

- **Up factor**: u = e^{(r - q)Δt + σ√Δt}
- **Down factor**: d = e^{(r - q)Δt - σ√Δt}
- **Risk-neutral probability**: p = (e^{(r - q)Δt} - d) / (u - d)

Where:
- r = risk-free rate
- q = dividend yield
- σ = volatility
- Δt = time step

### Backward Induction

**European options**: V = e^{-rΔt}[p·V_up + (1-p)·V_down]

**American options**: V = max(hold_value, exercise_value)

##  Technical Architecture
enhanced_options_model.py
├── VolatilityEstimator
│ ├── historical_vol() - Simple historical volatility
│ ├── ewma_vol() - RiskMetrics EWMA model
│ ├── garch_vol() - GARCH(1,1) forecasting
│ └── implied_vol() - Market-implied volatility
│
├── RiskFreeRate
│ ├── get_rate() - Country-specific reference rates
│ └── fetch_live_rate() - FRED API integration
│
├── BinomialOptionsPricing
│ ├── build_tree() - Construct price lattice
│ ├── price_option() - Main pricing engine
│ ├── black_scholes() - Validation benchmark
│ └── calculate_greeks() - Risk metrics
│
└── RealMarketOptionsPricing
├── Real-time data integration
├── Automatic parameter estimation
└── Visualization dashboard


## Example output:

============================================================
OPTION PRICING SUMMARY
============================================================
Underlying: $307.38 (US)
Strike: $305.00
Time to expiry: 1.00 years
Risk-free rate: 3.53%
Volatility: 29.37%
Dividend yield: 27.00%
Model: 100-step binomial tree (american put)
------------------------------------------------------------
Option price: $69.2721
------------------------------------------------------------
Greeks:
  Delta: -0.5462
  Gamma: 0.0025
  Theta: -17771.4970
  Vega: 8125.0304
  Rho: -23400.2023
============================================================

==================================================
Volatility Estimates for GOOGL
==================================================
Historical: 29.84%
EWMA      : 24.30%
GARCH     : 29.37%
Implied   : 216.27%

Recommendation:
  • For pricing: Use GARCH (29.37%)
  • For trading: Market expects 216.27%

Impact on Option Pricing:
----------------------------------------
Historical vol: 29.84% → price: $29.54
EWMA       vol: 24.30% → price: $24.83
GARCH      vol: 29.37% → price: $29.14
Implied    vol: 216.27% → price: $172.25

---


## Visualization Dashboard
The model generates a 4-panel visualization showing:

Historical Prices: Time series of the underlying stock

Returns Distribution: Histogram with normality reference

Volatility Comparison: Bar chart comparing estimation methods

Option Price Surface: Call and put prices across strike prices


## Known Limitations & Future Improvements

Fix Yahoo Finance dividend yield scaling (currently over 100% for some stocks)
Add trinomial tree for better convergence
implement Monte Carlo simulation for validation
Add support for exotic options (barriers, Asians)
Create web interface using Streamlit
Add historical backtesting framework


END

# options pricing model using black scholes
#22 feb 2026
#Hayley Falloon
"""
Enhanced Binomial Options Pricing Model
----------------------------------------
Features:
- Real market data integration (Yahoo Finance, FRED)
- GARCH volatility forecasting
- Country-specific risk-free rates (US, NZ, Europe, UK)
- Dividend yield from company data
- Multiple volatility estimation methods
- Comprehensive error handling
- Fixed MultiIndex column handling for yfinance data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force backend that works better on Mac
from scipy.stats import norm
from math import exp, sqrt
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False
    print("Note: arch package not installed. GARCH volatility not available.")
    print("Install with: pip install arch")

try:
    import pandas_datareader.data as web
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError:
    PANDAS_DATAREADER_AVAILABLE = False
    print("Note: pandas_datareader not installed. Live risk-free rates not available.")
    print("Install with: pip install pandas-datareader")


# ============================================================================
# VOLATILITY ESTIMATION MODULE
# ============================================================================

class VolatilityEstimator:
    """
    Advanced volatility estimation using multiple methods
    """
    
    def __init__(self, ticker, lookback_years=2):
        """
        Initialize with stock ticker
        
        Parameters:
        ticker: Stock symbol (e.g., 'GOOGL', 'FBU.NZ')
        lookback_years: Years of historical data to use
        """
        self.ticker = ticker
        self.lookback_years = lookback_years
        
        # Download data
        period = f'{lookback_years}y'
        print(f"Downloading {lookback_years} years of data for {ticker}...")
        self.data = yf.download(ticker, period=period, progress=False)
        
        if len(self.data) == 0:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Handle MultiIndex columns if present (critical fix!)
        if isinstance(self.data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns - extracting price data...")
            # Try to get 'Adj Close' from the MultiIndex
            if ('Adj Close', ticker) in self.data.columns:
                self.prices = self.data[('Adj Close', ticker)]
                print("Using Adj Close from MultiIndex")
            elif ('Close', ticker) in self.data.columns:
                self.prices = self.data[('Close', ticker)]
                print("Using Close from MultiIndex")
            else:
                # If all else fails, take the first column
                print("Using first available column")
                self.prices = self.data.iloc[:, 0]
        else:
            # Simple columns
            if 'Adj Close' in self.data.columns:
                self.prices = self.data['Adj Close']
                print("Using Adj Close column")
            elif 'Close' in self.data.columns:
                self.prices = self.data['Close']
                print("Using Close column")
            else:
                # Take the first column
                print("Using first column")
                self.prices = self.data.iloc[:, 0]
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        print(f"Loaded {len(self.returns)} daily returns for {ticker}")
        print(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
    
    def historical_vol(self, method='equal'):
        """
        Simple historical volatility
        
        Parameters:
        method: 'equal' or 'exp' (exponentially weighted)
        """
        if method == 'exp':
            # Exponentially weighted (more weight to recent)
            span = 60  # 60-day half-life
            weights = np.exp(-np.arange(len(self.returns)) / span)
            weights = weights / weights.sum()
            # Calculate weighted variance
            weighted_avg = np.sum(weights * self.returns)
            weighted_var = np.sum(weights * (self.returns - weighted_avg)**2)
            vol = np.sqrt(weighted_var * 252)
        else:
            # Equal weight
            vol = self.returns.std() * np.sqrt(252)
        
        return vol
    
    def ewma_vol(self, lambda_param=0.94):
        """
        RiskMetrics EWMA (Exponentially Weighted Moving Average)
        
        Parameters:
        lambda_param: Decay factor (0.94 is standard for daily data)
        """
        variance = np.zeros_like(self.returns)
        variance[0] = self.returns.iloc[0]**2
        
        for i in range(1, len(self.returns)):
            variance[i] = ((1 - lambda_param) * self.returns.iloc[i]**2 + 
                          lambda_param * variance[i-1])
        
        # Current variance (most recent)
        current_var = variance[-1]
        annualized_vol = np.sqrt(current_var * 252)
        
        return annualized_vol
    
    def garch_vol(self, horizon=30):
        """
        GARCH(1,1) volatility forecast
        
        Parameters:
        horizon: Forecast horizon in days
        """
        if not GARCH_AVAILABLE:
            print("GARCH not available, falling back to EWMA")
            return self.ewma_vol()
        
        try:
            # Fit GARCH model (multiply by 100 for numerical stability)
            model = arch_model(self.returns * 100, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            
            # Forecast
            forecasts = results.forecast(horizon=horizon)
            
            # Get forecasted variance and convert back
            forecast_var = forecasts.variance.iloc[-1].values[0] / 10000
            forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
            
            return forecast_vol
        except Exception as e:
            print(f"GARCH failed: {e}, falling back to EWMA")
            return self.ewma_vol()
    
    def implied_vol(self):
        """
        Get implied volatility from options market (if available)
        """
        try:
            stock = yf.Ticker(self.ticker)
            options = stock.option_chain()
            
            if len(options.calls) > 0:
                # Average implied vol from near-the-money options
                calls_iv = options.calls['impliedVolatility'].mean()
                puts_iv = options.puts['impliedVolatility'].mean()
                return (calls_iv + puts_iv) / 2
            else:
                return None
        except:
            return None
    
    def get_best_vol(self, method='garch'):
        """
        Get the best available volatility estimate
        """
        if method == 'garch' and GARCH_AVAILABLE:
            try:
                return self.garch_vol()
            except:
                return self.ewma_vol()
        elif method == 'ewma':
            return self.ewma_vol()
        else:
            return self.historical_vol()
    
    def compare_methods(self):
        """
        Compare all volatility estimates
        """
        methods = {
            'Historical': self.historical_vol(),
            'EWMA': self.ewma_vol(),
        }
        
        if GARCH_AVAILABLE:
            methods['GARCH'] = self.garch_vol()
        
        implied = self.implied_vol()
        if implied is not None:
            methods['Implied'] = implied
        
        print(f"\n{'='*50}")
        print(f"Volatility Estimates for {self.ticker}")
        print('='*50)
        for method, vol in methods.items():
            print(f"{method:10s}: {vol:.2%}")
        
        # Recommendation
        if 'Implied' in methods:
            print(f"\nRecommendation:")
            print(f"  • For pricing: Use GARCH ({methods['GARCH']:.2%})")
            print(f"  • For trading: Market expects {methods['Implied']:.2%}")
        elif GARCH_AVAILABLE:
            print(f"\nRecommendation: Use GARCH ({methods['GARCH']:.2%})")
        else:
            print(f"\nRecommendation: Use EWMA ({methods['EWMA']:.2%})")
        
        return methods


# ============================================================================
# RISK-FREE RATE MODULE
# ============================================================================

class RiskFreeRate:
    """
    Fetch risk-free rates for different countries
    """
    
    # Reference rates by country (you'd normally fetch these from APIs)
    # Format: {country: {maturity_years: rate}}
    RATES = {
        'US': {
            0.25: 0.052,   # 3-month T-bill
            1: 0.048,      # 1-year Treasury
            5: 0.045,      # 5-year Treasury
            10: 0.043      # 10-year Treasury
        },
        'NZ': {
            0.25: 0.055,   # 3-month rate
            1: 0.052,      # 1-year rate  
            5: 0.048,      # 5-year rate
            10: 0.045      # 10-year rate
        },
        'Germany': {       # Eurozone benchmark
            0.25: 0.035,
            1: 0.032,
            5: 0.028,
            10: 0.025
        },
        'UK': {
            0.25: 0.045,
            1: 0.042,
            5: 0.038,
            10: 0.035
        },
        'Australia': {
            0.25: 0.048,
            1: 0.045,
            5: 0.042,
            10: 0.040
        },
        'Japan': {
            0.25: 0.001,  # Near-zero rates
            1: 0.002,
            5: 0.005,
            10: 0.008
        }
    }
    
    @classmethod
    def get_rate(cls, country, maturity_years):
        """
        Get risk-free rate for country and maturity
        
        Parameters:
        country: Country code ('US', 'NZ', 'Germany', 'UK', etc.)
        maturity_years: Time to maturity in years
        """
        if country not in cls.RATES:
            print(f"Unknown country {country}, defaulting to US")
            country = 'US'
        
        # Find closest available maturity
        available = list(cls.RATES[country].keys())
        closest = min(available, key=lambda x: abs(x - maturity_years))
        
        return cls.RATES[country][closest]
    
    @classmethod
    def infer_country(cls, ticker):
        """
        Infer country from ticker suffix
        """
        if ticker.endswith('.NZ'):
            return 'NZ'
        elif ticker.endswith('.AU'):
            return 'Australia'
        elif ticker.endswith('.L'):
            return 'UK'
        elif ticker.endswith('.DE'):
            return 'Germany'
        elif ticker.endswith('.F'):
            return 'Germany'  # Eurozone
        elif ticker.endswith('.PA'):
            return 'Germany'  # Eurozone
        elif ticker.endswith('.TO'):
            return 'US'  # Canada (using US as proxy)
        elif ticker.endswith('.T'):
            return 'Japan'
        else:
            return 'US'  # Default
    
    @classmethod
    def fetch_live_rate(cls, country, maturity_years):
        """
        Attempt to fetch live rate from FRED or other sources
        """
        if not PANDAS_DATAREADER_AVAILABLE:
            return cls.get_rate(country, maturity_years)
        
        try:
            # FRED series IDs for different countries
            fred_ids = {
                'US': {
                    0.25: 'DTB3',      # 3-month T-bill
                    1: 'DGS1',         # 1-year Treasury
                    5: 'DGS5',         # 5-year Treasury
                    10: 'DGS10'        # 10-year Treasury
                },
                # Add more FRED series for other countries
            }
            
            if country in fred_ids:
                # Find closest maturity
                avail = list(fred_ids[country].keys())
                closest = min(avail, key=lambda x: abs(x - maturity_years))
                series_id = fred_ids[country][closest]
                
                # Fetch from FRED
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                data = web.DataReader(series_id, 'fred', start_date, end_date)
                
                if len(data) > 0:
                    latest = data.iloc[-1].values[0]
                    if not np.isnan(latest) and latest is not None:
                        return latest / 100
        except Exception as e:
            print(f"Could not fetch live rate: {e}")
        
        # Fallback to reference rates
        return cls.get_rate(country, maturity_years)


# ============================================================================
# ENHANCED BINOMIAL OPTIONS PRICING MODEL
# ============================================================================

class BinomialOptionsPricing:
    """
    Binomial Tree Model for pricing European and American options
    Enhanced with real market data capabilities
    """
    
    def __init__(self, S, K, T, r, sigma, N=100, 
                 option_type='call', exercise='european', 
                 dividend_yield=0, country='US'):
        """
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        N: Number of time steps
        option_type: 'call' or 'put'
        exercise: 'european' or 'american'
        dividend_yield: Continuous dividend yield
        country: Country for context (used in reporting)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()
        self.exercise = exercise.lower()
        self.dividend_yield = dividend_yield
        self.country = country
        
        # Validate inputs
        self._validate_inputs()
        
        # Binomial tree parameters
        self.dt = T / N
        self.u = exp((r - dividend_yield) * self.dt + sigma * sqrt(self.dt))
        self.d = exp((r - dividend_yield) * self.dt - sigma * sqrt(self.dt))
        self.p = (exp((r - dividend_yield) * self.dt) - self.d) / (self.u - self.d)
        
        # For convergence tracking
        self.prices_history = []
        self.deltas_history = []
        
        # Validate tree parameters
        self._validate_tree_parameters()
    
    def _validate_inputs(self):
        """Validate basic inputs"""
        if self.S <= 0:
            raise ValueError(f"Stock price must be positive: {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive: {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity must be positive: {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive: {self.sigma}")
        if self.N <= 0:
            raise ValueError(f"Number of steps must be positive: {self.N}")
    
    def _validate_tree_parameters(self):
        """Ensure tree parameters are valid"""
        if self.p < 0 or self.p > 1:
            print(f"\n⚠️  Warning: Risk-neutral probability p={self.p:.4f} is outside [0,1]")
            print(f"   This may indicate unrealistic parameters.")
            print(f"   Consider adjusting: r={self.r:.2%}, sigma={self.sigma:.2%}, dt={self.dt:.4f}")
    
    def build_tree(self):
        """
        Build the binomial price tree
        Returns: 2D array of stock prices at each node
        """
        tree = np.zeros((self.N + 1, self.N + 1))
        
        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        
        return tree
    
    def price_option(self):
        """
        Main pricing function
        Returns: option price, delta, stock_tree, option_tree
        """
        # Build stock price tree
        stock_tree = self.build_tree()
        
        # Initialize option value tree
        option_tree = np.zeros((self.N + 1, self.N + 1))
        
        # Calculate option values at maturity
        if self.option_type == 'call':
            option_tree[:, self.N] = np.maximum(0, stock_tree[:, self.N] - self.K)
        else:  # put
            option_tree[:, self.N] = np.maximum(0, self.K - stock_tree[:, self.N])
        
        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                # Discounted expected value
                hold_value = exp(-self.r * self.dt) * (
                    self.p * option_tree[j, i + 1] + 
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                
                if self.exercise == 'american':
                    # Check early exercise
                    if self.option_type == 'call':
                        exercise_value = stock_tree[j, i] - self.K
                    else:
                        exercise_value = self.K - stock_tree[j, i]
                    
                    option_tree[j, i] = max(hold_value, exercise_value)
                else:
                    option_tree[j, i] = hold_value
        
        # Option price is at the root
        option_price = option_tree[0, 0]
        
        # Calculate delta (sensitivity to stock price)
        if self.N > 0:
            delta = ((option_tree[0, 1] - option_tree[1, 1]) / 
                    (stock_tree[0, 1] - stock_tree[1, 1]))
        else:
            delta = 0
        
        # Store for convergence tracking
        self.prices_history.append((self.N, option_price))
        self.deltas_history.append((self.N, delta))
        
        return option_price, delta, stock_tree, option_tree
    
    def black_scholes(self):
        """
        Black-Scholes formula for European options (validation)
        """
        if self.exercise != 'european':
            return None
        
        d1 = (np.log(self.S / self.K) + 
              (self.r - self.dividend_yield + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            price = (self.S * np.exp(-self.dividend_yield * self.T) * norm.cdf(d1) - 
                    self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
                    self.S * np.exp(-self.dividend_yield * self.T) * norm.cdf(-d1))
        
        return price
    
    def calculate_greeks(self):
        """
        Calculate option Greeks using finite differences
        """
        # Small shifts for numerical Greeks
        dS = self.S * 0.01
        dT = 1/365  # One day
        dSigma = 0.01
        dr = 0.01
        
        # Base price
        price, delta_base, _, _ = self.price_option()
        
        # Create shifted instances
        up = BinomialOptionsPricing(
            self.S + dS, self.K, self.T, self.r, self.sigma, self.N,
            self.option_type, self.exercise, self.dividend_yield
        )
        down = BinomialOptionsPricing(
            self.S - dS, self.K, self.T, self.r, self.sigma, self.N,
            self.option_type, self.exercise, self.dividend_yield
        )
        time_up = BinomialOptionsPricing(
            self.S, self.K, max(self.T - dT, 0.01), self.r, self.sigma, self.N,
            self.option_type, self.exercise, self.dividend_yield
        )
        sigma_up = BinomialOptionsPricing(
            self.S, self.K, self.T, self.r, self.sigma + dSigma, self.N,
            self.option_type, self.exercise, self.dividend_yield
        )
        r_up = BinomialOptionsPricing(
            self.S, self.K, self.T, self.r + dr, self.sigma, self.N,
            self.option_type, self.exercise, self.dividend_yield
        )
        
        # Get prices
        price_up, _, _, _ = up.price_option()
        price_down, _, _, _ = down.price_option()
        price_time_up, _, _, _ = time_up.price_option()
        price_sigma_up, _, _, _ = sigma_up.price_option()
        price_r_up, _, _, _ = r_up.price_option()
        
        # Calculate Greeks
        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up - 2 * price + price_down) / (dS ** 2)
        theta = (price_time_up - price) / dT
        vega = (price_sigma_up - price) / dSigma
        rho = (price_r_up - price) / dr
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta * 365,  # Annualized
            'Vega': vega * 100,    # Per 1% vol change
            'Rho': rho * 100       # Per 1% rate change
        }
    
    def convergence_analysis(self, max_steps=500, step_size=10):
        """
        Analyze convergence as number of steps increases
        """
        print(f"\n{'='*50}")
        print(f"Convergence Analysis")
        print('='*50)
        
        steps = list(range(10, max_steps + 1, step_size))
        prices = []
        bs_price = self.black_scholes() if self.exercise == 'european' else None
        
        for n in steps:
            # Create new instance with different N
            opt = BinomialOptionsPricing(
                self.S, self.K, self.T, self.r, self.sigma, n,
                self.option_type, self.exercise, self.dividend_yield
            )
            price, _, _, _ = opt.price_option()
            prices.append(price)
            
            if n % 50 == 0:
                if bs_price:
                    error = abs(price - bs_price) / bs_price * 100
                    print(f"N={n:3d}: ${price:.4f} (error: {error:.4f}%)")
                else:
                    print(f"N={n:3d}: ${price:.4f}")
        
        return steps, prices, bs_price
    
    def summary(self):
        """
        Print comprehensive option summary
        """
        price, delta, _, _ = self.price_option()
        bs_price = self.black_scholes()
        greeks = self.calculate_greeks()
        
        print(f"\n{'='*60}")
        print(f"OPTION PRICING SUMMARY")
        print('='*60)
        print(f"Underlying: ${self.S:.2f} ({self.country})")
        print(f"Strike: ${self.K:.2f}")
        print(f"Time to expiry: {self.T:.2f} years")
        print(f"Risk-free rate: {self.r:.2%}")
        print(f"Volatility: {self.sigma:.2%}")
        print(f"Dividend yield: {self.dividend_yield:.2%}")
        print(f"Model: {self.N}-step binomial tree ({self.exercise} {self.option_type})")
        print('-' * 60)
        print(f"Option price: ${price:.4f}")
        
        if bs_price is not None:
            error = abs(price - bs_price) / bs_price * 100
            print(f"Black-Scholes: ${bs_price:.4f}")
            print(f"Difference: ${price - bs_price:.6f} ({error:.4f}%)")
        
        print('-' * 60)
        print("Greeks:")
        for greek, value in greeks.items():
            print(f"  {greek}: {value:.4f}")
        print('=' * 60)


# ============================================================================
# MARKET DATA INTEGRATION
# ============================================================================

class RealMarketOptionsPricing(BinomialOptionsPricing):
    """
    Enhanced version that fetches real market data
    """
    
    def __init__(self, ticker, expiry_date, strike=None, 
                 N=100, option_type='call', exercise='european',
                 vol_method='garch'):
        """
        Initialize with real market data
        
        Parameters:
        ticker: Stock symbol (e.g., 'GOOGL', 'FBU.NZ')
        expiry_date: Option expiry date (YYYY-MM-DD)
        strike: Strike price (None for at-the-money)
        N: Number of binomial steps
        option_type: 'call' or 'put'
        exercise: 'european' or 'american'
        vol_method: 'garch', 'ewma', or 'historical'
        """
        self.ticker = ticker
        self.expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        self.vol_method = vol_method
        
        print(f"\n{'='*60}")
        print(f"FETCHING REAL MARKET DATA FOR {ticker}")
        print('='*60)
        
        # 1. Get current stock price
        stock = yf.Ticker(ticker)
        self.stock_info = stock.info
        
        S = self.stock_info.get('regularMarketPrice', 
                                 self.stock_info.get('currentPrice', None))
        if S is None:
            # Fallback to recent close
            hist = stock.history(period='1d')
            if len(hist) > 0:
                S = hist['Close'].iloc[-1]
            else:
                raise ValueError(f"Could not get current price for {ticker}")
        
        print(f"Current price: ${S:.2f}")
        
        # 2. Calculate time to expiry
        today = datetime.now()
        days = (self.expiry - today).days
        if days <= 0:
            raise ValueError(f"Expiry date {expiry_date} is in the past")
        
        T = days / 365.25
        print(f"Time to expiry: {days} days ({T:.2f} years)")
        
        # 3. Infer country and get risk-free rate
        self.country = RiskFreeRate.infer_country(ticker)
        r = RiskFreeRate.fetch_live_rate(self.country, T)
        print(f"Country: {self.country}, Risk-free rate: {r:.2%}")
        
        # 4. Estimate volatility
        try:
            vol_est = VolatilityEstimator(ticker)
            if vol_method == 'garch':
                sigma = vol_est.garch_vol()
            elif vol_method == 'ewma':
                sigma = vol_est.ewma_vol()
            else:
                sigma = vol_est.historical_vol()
            
            print(f"Volatility ({vol_method}): {sigma:.2%}")
            self.vol_estimates = vol_est.compare_methods()
        except Exception as e:
            print(f"Could not estimate volatility: {e}")
            print("Using default volatility of 25%")
            sigma = 0.25
            self.vol_estimates = {'Default': 0.25}
        
        # 5. Get dividend yield
        dividend_yield = self.stock_info.get('dividendYield', 0)
        if dividend_yield is None:
            dividend_yield = 0
        print(f"Dividend yield: {dividend_yield:.2%}")
        
        # 6. Set strike price (default to ATM)
        if strike is None:
            # Round to nearest 5 for nicer strikes
            strike = round(S / 5) * 5
        print(f"Strike price: ${strike:.2f}")
        
        # Initialize parent class
        super().__init__(S, strike, T, r, sigma, N, 
                        option_type, exercise, dividend_yield, self.country)
   ######### 
    def plot_data_summary(self):
        """
        Plot historical data and volatility estimates
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Historical prices
            print("Generating price history plot...")
            vol_est = VolatilityEstimator(self.ticker)
            axes[0,0].plot(vol_est.prices.index, vol_est.prices, 'b-', linewidth=1)
            axes[0,0].set_title(f'{self.ticker} - Historical Prices')
            axes[0,0].set_ylabel('Price ($)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Returns distribution
            print("Generating returns distribution plot...")
            axes[0,1].hist(vol_est.returns, bins=50, edgecolor='black', alpha=0.7)
            axes[0,1].axvline(x=0, color='r', linestyle='--')
            axes[0,1].set_title('Returns Distribution')
            axes[0,1].set_xlabel('Daily Return')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Volatility comparison
            print("Generating volatility comparison plot...")
            methods = list(self.vol_estimates.keys())
            values = list(self.vol_estimates.values())
            colors = ['blue', 'green', 'orange', 'red'][:len(methods)]
            
            bars = axes[1,0].bar(methods, [v*100 for v in values], color=colors, alpha=0.7)
            axes[1,0].set_title('Volatility Estimates Comparison')
            axes[1,0].set_ylabel('Volatility (%)')
            axes[1,0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{val:.1%}', ha='center', fontsize=9)
            
            # Plot 4: Option price surface
            print("Generating option price surface plot...")
            strikes = np.arange(self.S * 0.7, self.S * 1.3, self.S * 0.05)
            prices_call = []
            prices_put = []
            
            for k in strikes:
                opt_call = BinomialOptionsPricing(
                    self.S, k, self.T, self.r, self.sigma, self.N,
                    'call', self.exercise, self.dividend_yield
                )
                opt_put = BinomialOptionsPricing(
                    self.S, k, self.T, self.r, self.sigma, self.N,
                    'put', self.exercise, self.dividend_yield
                )
                prices_call.append(opt_call.price_option()[0])
                prices_put.append(opt_put.price_option()[0])
            
            axes[1,1].plot(strikes, prices_call, 'b-', linewidth=2, label='Call')
            axes[1,1].plot(strikes, prices_put, 'r-', linewidth=2, label='Put')
            axes[1,1].axvline(x=self.S, color='black', linestyle='--', alpha=0.5, label='Current Price')
            axes[1,1].set_title('Option Prices vs Strike')
            axes[1,1].set_xlabel('Strike Price ($)')
            axes[1,1].set_ylabel('Option Price ($)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            print("\n" + "="*60)
            print("DISPLAYING PLOTS - Please check for a matplotlib window")
            print("Close the plot window to continue...")
            print("="*60)
            plt.show(block=True)  # This will wait until the user closes the plot
            print("Plot closed - continuing analysis...")
            
        except Exception as e:
            print(f"Could not generate plots: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def demonstrate_nz_option():
    """
    Example using NZ market parameters (FBU)
    """
    print("\n" + "="*60)
    print("NZ MARKET EXAMPLE: Fletcher Building (FBU.NZ)")
    print("="*60)
    
    try:
        # Use a date 1 year in the future
        future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Try with real data
        option = RealMarketOptionsPricing(
            ticker='FBU.NZ',
            expiry_date=future_date,
            option_type='call',
            exercise='american',
            vol_method='ewma'
        )
        
        option.summary()
        option.plot_data_summary()
        
        # Convergence analysis
        steps, prices, bs = option.convergence_analysis(max_steps=200)
        
    except Exception as e:
        print(f"\nCould not fetch real data: {e}")
        print("Using default NZ parameters instead...")
        
        # Fallback to default NZ parameters
        option = BinomialOptionsPricing(
            S=5.20, K=5.00, T=0.5, r=0.055, sigma=0.25, N=100,
            option_type='call', exercise='american', dividend_yield=0.04,
            country='NZ'
        )
        
        option.summary()
        
        # Convergence analysis
        steps, prices, bs = option.convergence_analysis(max_steps=200)

###
def demonstrate_us_option():
    """
    Example using US market parameters (GOOGL)
    """
    print("\n" + "="*60)
    print("US MARKET EXAMPLE: Google (GOOGL)")
    print("="*60)
    
    # Use a date 1 year in the future
    future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        option = RealMarketOptionsPricing(
            ticker='GOOGL',
            expiry_date=future_date,
            option_type='put',
            exercise='american',
            vol_method='garch'
        )
        
        option.summary()
        option.plot_data_summary()
    except Exception as e:
        print(f"Error with GOOGL: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_european_option():
    """
    Example using European market parameters (SAP.DE)
    """
    print("\n" + "="*60)
    print("EUROPEAN MARKET EXAMPLE: SAP (SAP.DE)")
    print("="*60)
    
    # Use a date 1 year in the future
    future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        option = RealMarketOptionsPricing(
            ticker='SAP.DE',
            expiry_date=future_date,
            option_type='call',
            exercise='european',
            vol_method='ewma'
        )
        
        option.summary()
    except Exception as e:
        print(f"Error with SAP.DE: {e}")


def compare_volatility_methods(ticker='GOOGL'):
    """
    Compare different volatility estimation methods
    """
    print("\n" + "="*60)
    print(f"VOLATILITY METHOD COMPARISON: {ticker}")
    print("="*60)
    
    try:
        estimator = VolatilityEstimator(ticker)
        methods = estimator.compare_methods()
        
        # Test impact on option pricing
        stock = yf.Ticker(ticker)
        S = stock.info.get('regularMarketPrice', 100)
        
        print("\nImpact on Option Pricing:")
        print("-" * 40)
        
        for method, vol in methods.items():
            if vol is not None:
                option = BinomialOptionsPricing(
                    S=S, K=S, T=0.5, r=0.05, sigma=vol, N=100,
                    option_type='call', exercise='european'
                )
                price, _, _, _ = option.price_option()
                print(f"{method:10s} vol: {vol:.2%} → price: ${price:.4f}")
    except Exception as e:
        print(f"Error comparing volatility methods: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENHANCED BINOMIAL OPTIONS PRICING MODEL")
    print("with Real Market Data Integration")
    print("="*60)
    
    # Run demonstrations
    demonstrate_nz_option()
    demonstrate_us_option()
    demonstrate_european_option()
    
    # Compare volatility methods
    compare_volatility_methods('GOOGL')
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)

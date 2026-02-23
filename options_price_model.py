# options pricing model using black scholes
#22 feb 2026
#Hayley Falloon

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force a specific backend that works better on Mac
from scipy.stats import norm
from math import exp, sqrt

class BinomialOptionsPricing:
    """
    Binomial Tree Model for pricing European and American options
    Validates against Black-Scholes formula
    """
    
    def __init__(self, S, K, T, r, sigma, N, option_type='call', exercise='european', dividend_yield=0):
        """
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        N: Number of time steps in binomial tree
        option_type: 'call' or 'put'
        exercise: 'european' or 'american'
        dividend_yield: Continuous dividend yield (for NZ stocks)
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
        
        # Binomial tree parameters
        self.dt = T / N  # Time step
        self.u = exp((r - dividend_yield) * self.dt + sigma * sqrt(self.dt))  # Up factor
        self.d = exp((r - dividend_yield) * self.dt - sigma * sqrt(self.dt))  # Down factor
        self.p = (exp((r - dividend_yield) * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability
        
        # For convergence tracking
        self.prices_history = []
        self.deltas_history = []
        
    def validate_parameters(self):
        """Ensure parameters are arbitrage-free"""
        if self.p < 0 or self.p > 1:
            print(f"Warning: Risk-neutral probability p={self.p:.4f} is outside [0,1]. Consider adjusting parameters.")
        return self.p >= 0 and self.p <= 1
    
    def build_tree(self):
        """
        Build the binomial price tree
        Returns: 2D array of stock prices at each node
        """
        # Initialize tree
        tree = np.zeros((self.N + 1, self.N + 1))
        
        # Fill the tree with stock prices
        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        
        return tree
    
    def price_option(self):
        """
        Main pricing function
        Returns: option price, delta, and the tree
        """
        # Validate parameters
        self.validate_parameters()
        
        # Build stock price tree
        stock_tree = self.build_tree()
        
        # Initialize option value tree
        option_tree = np.zeros((self.N + 1, self.N + 1))
        
        # Calculate option values at maturity (last column)
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
                    # For American options, check early exercise
                    if self.option_type == 'call':
                        exercise_value = stock_tree[j, i] - self.K
                    else:  # put
                        exercise_value = self.K - stock_tree[j, i]
                    
                    option_tree[j, i] = max(hold_value, exercise_value)
                else:
                    # European options can't be exercised early
                    option_tree[j, i] = hold_value
        
        # Option price is at the root [0,0]
        option_price = option_tree[0, 0]
        
        # Calculate delta (sensitivity to stock price)
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
        
        return option_price, delta, stock_tree, option_tree
    
    def black_scholes(self):
        """
        Black-Scholes formula for validation
        Returns: Black-Scholes price
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.dividend_yield + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            price = self.S * np.exp(-self.dividend_yield * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.dividend_yield * self.T) * norm.cdf(-d1)
        
        return price


def run_complete_analysis():
    """
    Run all analyses and display results in a single figure
    """
    print("=" * 70)
    print("OPTIONS PRICING MODEL: COMPLETE ANALYSIS")
    print("=" * 70)
    
    # Create a single figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ==================== 1. NZ OPTION EXAMPLE (Text Table) ====================
    print("\n--- NZ MARKET EXAMPLE (FBU: Fletcher Building) ---")
    
    # NZ Market Parameters
    S, K, T, r, sigma, div = 5.20, 5.00, 0.5, 0.055, 0.25, 0.04
    N = 100
    
    # Create a text table in the first subplot
    ax1 = plt.subplot(3, 3, 1)
    ax1.axis('tight')
    ax1.axis('off')
    
    # Calculate prices
    results_data = []
    for option_type in ['call', 'put']:
        for exercise in ['european', 'american']:
            binomial = BinomialOptionsPricing(S, K, T, r, sigma, N, option_type, exercise, div)
            price, delta, _, _ = binomial.price_option()
            
            if exercise == 'european':
                bs_price = binomial.black_scholes()
                diff_pct = (price - bs_price) / bs_price * 100
                results_data.append([
                    f"{exercise.capitalize()} {option_type.capitalize()}",
                    f"${price:.4f}",
                    f"${bs_price:.4f}",
                    f"{diff_pct:.3f}%"
                ])
            else:
                results_data.append([
                    f"{exercise.capitalize()} {option_type.capitalize()}",
                    f"${price:.4f}",
                    "N/A",
                    "N/A"
                ])
    
    # Create table
    table = ax1.table(cellText=results_data,
                      colLabels=['Option Type', 'Binomial', 'Black-Scholes', 'Diff'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax1.set_title('NZ Market Example: FBU @ $5.20', fontsize=10, fontweight='bold')
    
    # ==================== 2. CONVERGENCE ANALYSIS ====================
    ax2 = plt.subplot(3, 3, 2)
    
    S2, K2, T2, r2, sigma2, div2 = 100, 100, 1, 0.05, 0.2, 0.02
    N_values = [5, 10, 20, 50, 100, 200]
    binomial_prices = []
    
    for n in N_values:
        b = BinomialOptionsPricing(S2, K2, T2, r2, sigma2, n, 'call', 'european', div2)
        price, _, _, _ = b.price_option()
        binomial_prices.append(price)
    
    bs_price = BinomialOptionsPricing(S2, K2, T2, r2, sigma2, 1000, 'call', 'european', div2).black_scholes()
    
    ax2.plot(N_values, binomial_prices, 'bo-', linewidth=2, markersize=6, label='Binomial')
    ax2.axhline(y=bs_price, color='r', linestyle='--', label=f'BS: {bs_price:.2f}')
    ax2.set_xlabel('Time Steps (N)')
    ax2.set_ylabel('Option Price')
    ax2.set_title('Convergence to Black-Scholes')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_xscale('log')
    
    # ==================== 3. TREE STRUCTURE VISUALIZATION ====================
    ax3 = plt.subplot(3, 3, 3)
    
    N_small = 5
    b_small = BinomialOptionsPricing(100, 100, 1, 0.05, 0.2, N_small, 'call', 'european')
    tree = b_small.build_tree()
    
    for i in range(N_small + 1):
        for j in range(i + 1):
            x = i
            y = j - i/2
            ax3.plot(x, y, 'bo', markersize=4)
            if i < N_small:
                ax3.plot([x, x+1], [y, y-0.5], 'b-', alpha=0.2)
                ax3.plot([x, x+1], [y, y+0.5], 'b-', alpha=0.2)
    ax3.set_title(f'Binomial Tree (N={N_small})')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Node Position')
    ax3.set_xticks(range(N_small+1))
    
    # ==================== 4. STOCK PRICE SENSITIVITY ====================
    ax4 = plt.subplot(3, 3, 4)
    
    S_range = np.arange(80, 121, 5)
    call_prices, put_prices = [], []
    
    for s in S_range:
        call = BinomialOptionsPricing(s, 100, 1, 0.05, 0.2, 100, 'call', 'european', 0.02)
        put = BinomialOptionsPricing(s, 100, 1, 0.05, 0.2, 100, 'put', 'european', 0.02)
        call_prices.append(call.price_option()[0])
        put_prices.append(put.price_option()[0])
    
    ax4.plot(S_range, call_prices, 'b-', linewidth=2, label='Call')
    ax4.plot(S_range, put_prices, 'r-', linewidth=2, label='Put')
    ax4.axvline(x=100, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Stock Price ($)')
    ax4.set_ylabel('Option Price')
    ax4.set_title('Price vs Stock Price')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # ==================== 5. VOLATILITY SENSITIVITY ====================
    ax5 = plt.subplot(3, 3, 5)
    
    sigma_range = np.arange(0.1, 0.51, 0.05)
    call_sigma, put_sigma = [], []
    
    for sig in sigma_range:
        call = BinomialOptionsPricing(100, 100, 1, 0.05, sig, 100, 'call', 'european', 0.02)
        put = BinomialOptionsPricing(100, 100, 1, 0.05, sig, 100, 'put', 'european', 0.02)
        call_sigma.append(call.price_option()[0])
        put_sigma.append(put.price_option()[0])
    
    ax5.plot(sigma_range, call_sigma, 'b-', linewidth=2, label='Call')
    ax5.plot(sigma_range, put_sigma, 'r-', linewidth=2, label='Put')
    ax5.set_xlabel('Volatility')
    ax5.set_ylabel('Option Price')
    ax5.set_title('Price vs Volatility')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    # ==================== 6. TIME SENSITIVITY ====================
    ax6 = plt.subplot(3, 3, 6)
    
    T_range = np.arange(0.1, 2.1, 0.2)
    call_T, put_T = [], []
    
    for t in T_range:
        call = BinomialOptionsPricing(100, 100, t, 0.05, 0.2, 100, 'call', 'european', 0.02)
        put = BinomialOptionsPricing(100, 100, t, 0.05, 0.2, 100, 'put', 'european', 0.02)
        call_T.append(call.price_option()[0])
        put_T.append(put.price_option()[0])
    
    ax6.plot(T_range, call_T, 'b-', linewidth=2, label='Call')
    ax6.plot(T_range, put_T, 'r-', linewidth=2, label='Put')
    ax6.set_xlabel('Time to Maturity (years)')
    ax6.set_ylabel('Option Price')
    ax6.set_title('Price vs Time')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)
    
    # ==================== 7. DIVIDEND SENSITIVITY (NZ FOCUS) ====================
    ax7 = plt.subplot(3, 3, 7)
    
    div_range = np.arange(0, 0.1, 0.01)
    call_div, put_div = [], []
    
    for d in div_range:
        call = BinomialOptionsPricing(100, 100, 1, 0.05, 0.2, 100, 'call', 'european', d)
        put = BinomialOptionsPricing(100, 100, 1, 0.05, 0.2, 100, 'put', 'european', d)
        call_div.append(call.price_option()[0])
        put_div.append(put.price_option()[0])
    
    ax7.plot(div_range, call_div, 'b-', linewidth=2, label='Call')
    ax7.plot(div_range, put_div, 'r-', linewidth=2, label='Put')
    ax7.set_xlabel('Dividend Yield')
    ax7.set_ylabel('Option Price')
    ax7.set_title('Price vs Dividend (NZ Focus)')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=8)
    
    # ==================== 8. AMERICAN vs EUROPEAN PREMIUM ====================
    ax8 = plt.subplot(3, 3, 8)
    
    S_range2 = np.arange(80, 121, 5)
    euro_puts, amer_puts = [], []
    
    for s in S_range2:
        euro = BinomialOptionsPricing(s, 100, 1, 0.05, 0.2, 100, 'put', 'european', 0.02)
        amer = BinomialOptionsPricing(s, 100, 1, 0.05, 0.2, 100, 'put', 'american', 0.02)
        euro_puts.append(euro.price_option()[0])
        amer_puts.append(amer.price_option()[0])
    
    premium = [a - e for a, e in zip(amer_puts, euro_puts)]
    ax8.plot(S_range2, premium, 'g-', linewidth=2)
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax8.set_xlabel('Stock Price')
    ax8.set_ylabel('Premium')
    ax8.set_title('American Put Premium')
    ax8.grid(True, alpha=0.3)
    
    # ==================== 9. ERROR ANALYSIS ====================
    ax9 = plt.subplot(3, 3, 9)
    
    N_range = [10, 20, 50, 100, 200, 500]
    errors = []
    
    for n in N_range:
        b = BinomialOptionsPricing(100, 100, 1, 0.05, 0.2, n, 'call', 'european', 0.02)
        price, _, _, _ = b.price_option()
        bs = b.black_scholes()
        errors.append(abs(price - bs) / bs * 100)
    
    ax9.plot(N_range, errors, 'ro-', linewidth=2)
    ax9.set_xlabel('Time Steps (N)')
    ax9.set_ylabel('Error %')
    ax9.set_title('Pricing Error vs Black-Scholes')
    ax9.grid(True, alpha=0.3)
    ax9.set_xscale('log')
    ax9.set_yscale('log')
    
    # Final touches
    plt.suptitle('OPTIONS PRICING MODEL: COMPLETE ANALYSIS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE: All visualizations shown in single figure")
    print("=" * 70)
    
    # Optional: Ask if user wants to save
    save_option = input("\nSave figure to file? (y/n): ").lower()
    if save_option == 'y':
        filename = input("Enter filename (default: options_dashboard.png): ") or "options_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved to {filename}")


if __name__ == "__main__":
    try:
        run_complete_analysis()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close('all')  # Ensure figures are closed
        print("Done - you can safely exit")
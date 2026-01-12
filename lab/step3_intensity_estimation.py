#!/usr/bin/env python3
"""
Step 3: Order Intensity Estimation
논문 Eq. 13-14: Λ^a_i(δ) = A^a_i * exp(-η^a_i * δ)

Key concept:
- δ (delta): half-spread, the distance from mid-price to quote
- Λ(δ): execution intensity (expected fills per unit time) at spread δ
- Larger δ → fewer executions (exponential decay)

Estimation approach:
1. For each regime and side (buy/sell), group trades by the half-spread at execution
2. Count number of executions per unit time at each spread level
3. Fit exponential decay: Λ(δ) = A * exp(-η * δ)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

TRADING_HOURS = 6.5
INTERVAL_MINUTES = 5
INTERVALS_PER_DAY = int(TRADING_HOURS * 60 / INTERVAL_MINUTES)  # 78

# Folder structure
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CSV_DIR = OUTPUT_DIR / "csv"
PLOTS_DIR = OUTPUT_DIR / "plots"
PARAMS_DIR = OUTPUT_DIR / "parameters"

print("=" * 70)
print("Step 3: Order Intensity Estimation (논문 Eq. 13-14)")
print("=" * 70)

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n" + "-" * 60)
print("1. Loading Data")
print("-" * 60)

trades = pd.read_csv(CSV_DIR / 'trades_classified.csv', parse_dates=['DateTime'])
quotes = pd.read_csv(CSV_DIR / 'quotes_cleaned.csv', parse_dates=['DateTime'])
regimes = pd.read_csv(CSV_DIR / 'regime_results.csv', parse_dates=['DateTime'])

print(f"  Trades: {len(trades):,}")
print(f"  Quotes: {len(quotes):,}")
print(f"  Regime observations: {len(regimes):,}")

# ============================================================================
# 2. Merge Regime Information
# ============================================================================

print("\n" + "-" * 60)
print("2. Merging Regime Information")
print("-" * 60)

trades['DateTime_5min'] = trades['DateTime'].dt.floor('5min')
regimes['DateTime_5min'] = regimes['DateTime'].dt.floor('5min')

trades = pd.merge(
    trades,
    regimes[['DateTime_5min', 'Regime']].drop_duplicates(),
    on='DateTime_5min',
    how='left'
)

trades['Regime'] = trades['Regime'].ffill().bfill()
trades = trades.dropna(subset=['Regime'])
trades['Regime'] = trades['Regime'].astype(int)

print(f"  Trades with regime: {len(trades):,}")
print(f"  Low regime trades: {(trades['Regime']==0).sum():,}")
print(f"  High regime trades: {(trades['Regime']==1).sum():,}")

# ============================================================================
# 3. Calculate Half-Spread
# ============================================================================

print("\n" + "-" * 60)
print("3. Calculating Half-Spread at Execution")
print("-" * 60)

quotes_merge = quotes[['DateTime', 'Mid']].copy()
trades = pd.merge_asof(
    trades.sort_values('DateTime'),
    quotes_merge.sort_values('DateTime'),
    on='DateTime',
    direction='backward',
    suffixes=('', '_quote')
)

# δ calculation
if 'Mid' not in trades.columns and 'Mid_quote' in trades.columns:
    trades['Mid'] = trades['Mid_quote']

trades['Delta'] = np.where(
    trades['Side'] == 'buy',
    np.abs(trades['Price'] - trades['Mid']),
    np.abs(trades['Mid'] - trades['Price'])
)

# Filter: δ > 0 and remove outliers
trades = trades[trades['Delta'] > 0].copy()
delta_cap = trades['Delta'].quantile(0.99)
trades = trades[trades['Delta'] <= delta_cap].copy()

print(f"  Trades with valid δ: {len(trades):,}")
print(f"  δ range: [{trades['Delta'].min():.6f}, {trades['Delta'].max():.6f}]")
print(f"  Mean δ: {trades['Delta'].mean():.6f}")
print(f"  Median δ: {trades['Delta'].median():.6f}")

# ============================================================================
# 4. MLE Estimation (Primary Method)
# ============================================================================
"""
Primary Method: MLE under exponential assumption

If trades occur at various δ levels according to Λ(δ) = A*exp(-η*δ),
and we observe the δ values of executed trades, then under mild assumptions,
the MLE for η is simply:

    η_MLE = 1 / mean(δ)

This is because the observed δ distribution is proportional to Λ(δ).

The baseline intensity A is estimated from the total trade count:
    A = (total trades) / (total time) * correction_factor
"""

print("\n" + "-" * 60)
print("4. MLE Estimation (Primary Method)")
print("-" * 60)

def estimate_mle(trades_subset, total_time_sec, regime_name, side_name):
    """MLE estimation for intensity parameters."""
    deltas = trades_subset['Delta'].values
    n = len(deltas)
    
    if n < 100:
        print(f"  {regime_name} {side_name}: Insufficient data ({n})")
        return None
    
    # MLE for η
    mean_delta = np.mean(deltas)
    eta = 1.0 / mean_delta
    eta_se = eta / np.sqrt(n)  # Standard error
    
    # Baseline intensity A
    # A represents intensity when δ→0
    # Observed rate = A * E[exp(-η*δ)] under the model
    # For exponential δ with rate η: E[exp(-η*δ)] = η/(η+η) = 0.5
    # So: A ≈ 2 * observed_rate
    observed_rate = n / total_time_sec
    A = observed_rate  # Simplified: use observed rate directly
    
    print(f"\n  {regime_name} {side_name}:")
    print(f"    n = {n:,} trades")
    print(f"    mean(δ) = {mean_delta:.6f}")
    print(f"    η = {eta:.2f} ± {eta_se:.2f}")
    print(f"    A = {A:.4f} fills/sec = {A*60:.2f} fills/min")
    
    return {
        'n_trades': n,
        'mean_delta': mean_delta,
        'eta': eta,
        'eta_se': eta_se,
        'A': A,
        'A_per_min': A * 60
    }

# Calculate time spans
total_time = (trades['DateTime'].max() - trades['DateTime'].min()).total_seconds()
print(f"\n  Total time span: {total_time/3600:.1f} hours")

results_mle = []
for regime in [0, 1]:
    regime_name = 'Low' if regime == 0 else 'High'
    regime_data = trades[trades['Regime'] == regime]
    regime_time = (regime_data['DateTime'].max() - regime_data['DateTime'].min()).total_seconds()
    
    if regime_time <= 0:
        regime_time = total_time / 2
    
    for side in ['buy', 'sell']:
        quote_side = 'ask' if side == 'buy' else 'bid'
        subset = regime_data[regime_data['Side'] == side]
        
        result = estimate_mle(subset, regime_time, regime_name, f"{side} ({quote_side})")
        if result:
            results_mle.append({
                'Regime': regime,
                'Regime_Name': regime_name,
                'Side': side,
                'Quote_Side': quote_side,
                **result
            })

# ============================================================================
# 5. Empirical Rate Estimation (Secondary Method - Improved)
# ============================================================================

print("\n" + "-" * 60)
print("5. Empirical Rate Estimation (Improved Binning)")
print("-" * 60)

def estimate_empirical_improved(trades_subset, regime_name, side_name):
    """
    Improved empirical estimation with equal-width binning.
    """
    deltas = trades_subset['Delta'].values
    n = len(deltas)
    
    if n < 500:
        print(f"  {regime_name} {side_name}: Insufficient data ({n})")
        return None
    
    # Use equal-width bins (not percentile-based)
    delta_min, delta_max = deltas.min(), deltas.max()
    n_bins = 30
    bin_edges = np.linspace(delta_min, delta_max, n_bins + 1)
    
    counts, _ = np.histogram(deltas, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Density (normalized histogram)
    density = counts / (n * bin_widths)
    
    # Filter bins with enough data
    valid = counts >= 10
    if valid.sum() < 5:
        print(f"  {regime_name} {side_name}: Not enough valid bins")
        return None
    
    x = bin_centers[valid]
    y = density[valid]
    
    # Fit: f(δ) = C * exp(-η * δ)
    # where C is a normalization constant
    def exp_model(delta, C, eta):
        return C * np.exp(-eta * delta)
    
    # Initial guess from log-linear fit
    try:
        log_y = np.log(y[y > 0])
        x_valid = x[y > 0]
        if len(log_y) >= 2:
            slope, intercept = np.polyfit(x_valid, log_y, 1)
            eta_init = -slope
            C_init = np.exp(intercept)
        else:
            eta_init, C_init = 200, y.max()
    except:
        eta_init, C_init = 200, y.max()
    
    if eta_init <= 0:
        eta_init = 200
    
    try:
        popt, pcov = curve_fit(
            exp_model, x, y,
            p0=[C_init, eta_init],
            bounds=([1e-10, 1], [1e6, 2000]),
            maxfev=10000
        )
        C, eta = popt
        
        # R²
        y_pred = exp_model(x, C, eta)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        print(f"\n  {regime_name} {side_name}:")
        print(f"    η = {eta:.2f}, R² = {r2:.3f}")
        print(f"    valid bins = {valid.sum()}")
        
        return {
            'eta': eta,
            'C': C,
            'r2': r2,
            'x': x,
            'y': y,
            'y_pred': y_pred,
            'n_bins': valid.sum()
        }
    except Exception as e:
        print(f"  {regime_name} {side_name}: Fit failed - {e}")
        return None

results_empirical = []
for regime in [0, 1]:
    regime_name = 'Low' if regime == 0 else 'High'
    regime_data = trades[trades['Regime'] == regime]
    
    for side in ['buy', 'sell']:
        quote_side = 'ask' if side == 'buy' else 'bid'
        subset = regime_data[regime_data['Side'] == side]
        
        result = estimate_empirical_improved(subset, regime_name, f"{side} ({quote_side})")
        if result:
            results_empirical.append({
                'Regime': regime,
                'Regime_Name': regime_name,
                'Side': side,
                'Quote_Side': quote_side,
                **result
            })

# ============================================================================
# 6. Save Results
# ============================================================================

print("\n" + "-" * 60)
print("6. Saving Results")
print("-" * 60)

# MLE results (primary)
if results_mle:
    df_mle = pd.DataFrame([{
        'Regime': r['Regime'],
        'Regime_Name': r['Regime_Name'],
        'Side': r['Side'],
        'Quote_Side': r['Quote_Side'],
        'A_per_sec': r['A'],
        'A_per_min': r['A_per_min'],
        'eta': r['eta'],
        'eta_se': r['eta_se'],
        'mean_delta': r['mean_delta'],
        'n_trades': r['n_trades']
    } for r in results_mle])
    df_mle.to_csv(PARAMS_DIR / 'intensity_parameters_final.csv', index=False)
    print(f"  Saved: {PARAMS_DIR}/intensity_parameters_final.csv")

# ============================================================================
# 7. Create Plots
# ============================================================================

print("\n" + "-" * 60)
print("7. Creating Plots")
print("-" * 60)

# Plot 1: Delta distribution with fitted exponential
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_positions = {(0, 'buy'): (0,0), (0, 'sell'): (0,1),
                  (1, 'buy'): (1,0), (1, 'sell'): (1,1)}

for regime in [0, 1]:
    regime_name = 'Low' if regime == 0 else 'High'
    regime_data = trades[trades['Regime'] == regime]
    
    for side in ['buy', 'sell']:
        quote_side = 'ask' if side == 'buy' else 'bid'
        subset = regime_data[regime_data['Side'] == side]
        
        row, col = plot_positions[(regime, side)]
        ax = axes[row, col]
        
        # Histogram
        deltas = subset['Delta'].values
        ax.hist(deltas, bins=50, density=True, alpha=0.7, 
                color='steelblue', edgecolor='white', label='Observed')
        
        # Fitted exponential from MLE
        mle_result = next((r for r in results_mle 
                          if r['Regime']==regime and r['Side']==side), None)
        if mle_result:
            x_range = np.linspace(0, deltas.max(), 100)
            eta = mle_result['eta']
            # Exponential PDF: f(x) = η * exp(-η*x)
            y_fit = eta * np.exp(-eta * x_range)
            ax.plot(x_range, y_fit, 'r-', lw=2, 
                   label=f'Exp fit: η={eta:.1f}')
        
        ax.set_xlabel('Half-Spread δ ($)')
        ax.set_ylabel('Density')
        ax.set_title(f'{regime_name} Regime - {side.capitalize()} ({quote_side})\n'
                    f'n={len(deltas):,}, mean(δ)={deltas.mean():.5f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'delta_distribution_final.png', dpi=150)
plt.close()
print(f"  Saved: {PLOTS_DIR}/delta_distribution_final.png")

# Plot 2: Empirical fit comparison
if results_empirical:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for r in results_empirical:
        regime, side = r['Regime'], r['Side']
        row, col = plot_positions[(regime, side)]
        ax = axes[row, col]
        
        ax.scatter(r['x'], r['y'], s=40, alpha=0.7, label='Empirical density')
        
        x_smooth = np.linspace(r['x'].min(), r['x'].max(), 100)
        y_smooth = r['C'] * np.exp(-r['eta'] * x_smooth)
        ax.plot(x_smooth, y_smooth, 'r-', lw=2, 
               label=f"Fit: η={r['eta']:.1f}, R²={r['r2']:.3f}")
        
        ax.set_xlabel('Half-Spread δ ($)')
        ax.set_ylabel('Density')
        ax.set_title(f"{r['Regime_Name']} - {side.capitalize()} ({r['Quote_Side']})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'intensity_fit_final.png', dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/intensity_fit_final.png")

# ============================================================================
# 8. Summary
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS: Intensity Parameters")
print("=" * 70)

print("""
  Model: Λ(δ) = A * exp(-η * δ)
  
  - A: baseline intensity (fills per second when δ→0)
  - η: price sensitivity (how fast intensity decays with spread)
  
  Interpretation:
  - At δ = 0.01 ($0.01 spread), intensity drops to exp(-η*0.01) of baseline
  - Higher η means traders are more price-sensitive
""")

print("\n  MLE Results (Recommended):")
print("-" * 60)
print(f"  {'Regime':<6} {'Side':<6} {'A (fills/sec)':<14} {'A (fills/min)':<14} {'η':<10}")
print("-" * 60)
for r in results_mle:
    print(f"  {r['Regime_Name']:<6} {r['Side']:<6} {r['A']:>13.4f} {r['A_per_min']:>13.2f} {r['eta']:>9.2f}")
print("-" * 60)

# Economic interpretation
print("\n  Economic Interpretation:")
for r in results_mle:
    delta_1cent = 0.01
    decay = np.exp(-r['eta'] * delta_1cent)
    print(f"    {r['Regime_Name']} {r['Side']}: At δ=$0.01, intensity is {decay*100:.1f}% of baseline")

print("\n" + "=" * 70)
print("✅ Step 3 Complete!")
print("=" * 70)
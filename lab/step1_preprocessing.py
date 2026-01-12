#!/usr/bin/env python3
"""
Step 1: Data Preprocessing
논문: Optimal Market Making under Regime-Switching Heston

Key corrections:
1. Trading day = 6.5 hours (9:30-16:00)
2. 5-min intervals: 78 per day
3. Realized Variance: Sum of squared returns within each 5-min window
4. No overnight returns (each day independent)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import time

# ============================================================================
# Configuration
# ============================================================================

# Time constants (논문 Section 3.1: T = 6.5 hours for US equity markets)
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)
TRADING_HOURS = 6.5  # hours
MINUTES_PER_DAY = TRADING_HOURS * 60  # 390 minutes
INTERVAL_MINUTES = 5
INTERVALS_PER_DAY = int(MINUTES_PER_DAY / INTERVAL_MINUTES)  # 78

# Folder structure
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CSV_DIR = OUTPUT_DIR / "csv"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Step 1: Data Preprocessing")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Trading hours: {TRADING_START} - {TRADING_END} ({TRADING_HOURS}h)")
print(f"  Interval: {INTERVAL_MINUTES} min")
print(f"  Intervals per day: {INTERVALS_PER_DAY}")

# ============================================================================
# 1. Load Raw Data
# ============================================================================

print("\n" + "-" * 60)
print("1. Loading Raw Data")
print("-" * 60)

quotes = pd.read_csv(DATA_DIR / "MSFT_quotes_combined.csv", parse_dates=['DateTime'])
quotes = quotes.sort_values('DateTime').reset_index(drop=True)
print(f"  Quotes: {len(quotes):,}")

trades = pd.read_csv(DATA_DIR / "MSFT_trades_combined.csv", parse_dates=['DateTime'])
trades = trades.sort_values('DateTime').reset_index(drop=True)
print(f"  Trades: {len(trades):,}")

# ============================================================================
# 2. Filter Trading Hours
# ============================================================================

print("\n" + "-" * 60)
print("2. Filtering Trading Hours")
print("-" * 60)

quotes_time = quotes['DateTime'].dt.time
quotes = quotes[(quotes_time >= TRADING_START) & (quotes_time <= TRADING_END)].copy()

trades_time = trades['DateTime'].dt.time
trades = trades[(trades_time >= TRADING_START) & (trades_time <= TRADING_END)].copy()

quotes['Date'] = quotes['DateTime'].dt.date
trades['Date'] = trades['DateTime'].dt.date
trading_days = sorted(quotes['Date'].unique())

print(f"  Quotes after filter: {len(quotes):,}")
print(f"  Trades after filter: {len(trades):,}")
print(f"  Trading days: {len(trading_days)}")

# ============================================================================
# 3. Clean Data (Per Day)
# ============================================================================

print("\n" + "-" * 60)
print("3. Cleaning Data")
print("-" * 60)

quotes_list = []
trades_list = []

for day in trading_days:
    q = quotes[quotes['Date'] == day].copy()
    t = trades[trades['Date'] == day].copy()
    
    # Clean quotes: positive spread, remove outliers
    q['Spread'] = q['Ask'] - q['Bid']
    q = q[(q['Spread'] > 0) & (q['Bid'] > 0) & (q['Ask'] > 0)]
    q = q[q['Spread'] <= q['Spread'].quantile(0.99)]
    q['Mid'] = (q['Bid'] + q['Ask']) / 2
    
    # Clean trades: positive price, remove outliers
    t = t[t['Price'] > 0]
    t = t[(t['Price'] >= t['Price'].quantile(0.01)) & 
          (t['Price'] <= t['Price'].quantile(0.99))]
    
    quotes_list.append(q)
    trades_list.append(t)

quotes = pd.concat(quotes_list, ignore_index=True)
trades = pd.concat(trades_list, ignore_index=True)

print(f"  Quotes after cleaning: {len(quotes):,}")
print(f"  Trades after cleaning: {len(trades):,}")

# ============================================================================
# 4. Trade Classification (Lee-Ready)
# ============================================================================

print("\n" + "-" * 60)
print("4. Trade Classification (Lee-Ready)")
print("-" * 60)

trades_list = []
for day in trading_days:
    q = quotes[quotes['Date'] == day].sort_values('DateTime')
    t = trades[trades['Date'] == day].sort_values('DateTime')
    
    # Match each trade to the most recent quote
    t = pd.merge_asof(t, q[['DateTime', 'Mid']], on='DateTime', direction='backward')
    
    # Classify: buy if trade price >= mid, sell otherwise
    t['Direction'] = np.where(t['Price'] >= t['Mid'], 1, -1)
    t['Side'] = np.where(t['Direction'] == 1, 'buy', 'sell')
    
    trades_list.append(t.dropna(subset=['Mid']))

trades = pd.concat(trades_list, ignore_index=True)

buy_count = (trades['Side'] == 'buy').sum()
sell_count = (trades['Side'] == 'sell').sum()
print(f"  Buy trades: {buy_count:,} ({buy_count/len(trades)*100:.1f}%)")
print(f"  Sell trades: {sell_count:,} ({sell_count/len(trades)*100:.1f}%)")

# ============================================================================
# 5. Realized Variance Calculation
# ============================================================================
"""
논문 Eq. 5-6: dS_t = √V_t dW^S_t

Realized Variance (5-min):
RV_t = Σ r_i² where r_i = log(P_i) - log(P_{i-1})

We keep RV in its raw form (not annualized) to match the Heston model's
instantaneous variance V_t. The HMM will identify regimes based on
relative variance levels.

Note: Different scaling choices are possible:
- Raw RV: directly comparable to instantaneous variance
- Annualized: RV * (intervals_per_day * trading_days_per_year)
"""

print("\n" + "-" * 60)
print("5. Realized Variance Calculation (5-min)")
print("-" * 60)

variance_list = []

for day in trading_days:
    t = trades[trades['Date'] == day].copy()
    t = t.sort_values('DateTime')
    
    # Log returns (within day only - no overnight)
    t['LogPrice'] = np.log(t['Price'])
    t['Return'] = t['LogPrice'].diff()
    
    # 5-min buckets
    t['DateTime_5min'] = t['DateTime'].dt.floor('5min')
    
    # Realized variance = sum of squared returns in each bucket
    rv = t.groupby('DateTime_5min').agg({
        'Return': lambda x: (x**2).sum(),  # RV
        'Price': 'count'  # Number of trades
    }).reset_index()
    rv.columns = ['DateTime', 'RV_raw', 'NumTrades']
    rv['Date'] = day
    
    variance_list.append(rv)

variance_5min = pd.concat(variance_list, ignore_index=True)

# Remove zero-variance intervals (no price movement)
variance_5min = variance_5min[variance_5min['RV_raw'] > 0].copy()

# Scale to make values more interpretable
# Option 1: Keep raw (for HMM regime identification)
variance_5min['Variance'] = variance_5min['RV_raw']

# Option 2: Annualized variance (uncomment if preferred)
# variance_5min['Variance'] = variance_5min['RV_raw'] * INTERVALS_PER_DAY * 252

print(f"  Total observations: {len(variance_5min):,}")
print(f"  Per day: ~{len(variance_5min)/len(trading_days):.0f}")
print(f"\n  Variance statistics (raw):")
print(f"    Mean: {variance_5min['Variance'].mean():.6f}")
print(f"    Std:  {variance_5min['Variance'].std():.6f}")
print(f"    Min:  {variance_5min['Variance'].min():.6f}")
print(f"    Max:  {variance_5min['Variance'].max():.6f}")

# ============================================================================
# 6. Aggregate Mid Prices and Spreads
# ============================================================================

print("\n" + "-" * 60)
print("6. Aggregating Mid Prices and Spreads")
print("-" * 60)

mid_list, spread_list = [], []

for day in trading_days:
    q = quotes[quotes['Date'] == day].copy()
    q['DateTime_5min'] = q['DateTime'].dt.floor('5min')
    
    # Mid price at end of each interval
    mid = q.groupby('DateTime_5min')['Mid'].last().reset_index()
    mid.columns = ['DateTime', 'Mid']
    mid['Date'] = day
    mid_list.append(mid)
    
    # Spread statistics
    spread = q.groupby('DateTime_5min')['Spread'].agg(['mean', 'median', 'std']).reset_index()
    spread.columns = ['DateTime', 'Spread_Mean', 'Spread_Median', 'Spread_Std']
    spread['Date'] = day
    spread_list.append(spread)

mid_5min = pd.concat(mid_list, ignore_index=True)
spread_5min = pd.concat(spread_list, ignore_index=True)

print(f"  Mid price observations: {len(mid_5min):,}")
print(f"  Spread observations: {len(spread_5min):,}")

# ============================================================================
# 7. Save Outputs
# ============================================================================

print("\n" + "-" * 60)
print("7. Saving Outputs")
print("-" * 60)

variance_5min.to_csv(CSV_DIR / 'realized_variance_5min.csv', index=False)
mid_5min.to_csv(CSV_DIR / 'mid_prices_5min.csv', index=False)
spread_5min.to_csv(CSV_DIR / 'spreads_5min.csv', index=False)

trades[['DateTime', 'Date', 'Price', 'Size', 'Direction', 'Side']].to_csv(
    CSV_DIR / 'trades_classified.csv', index=False
)
quotes[['DateTime', 'Date', 'Bid', 'Ask', 'Mid', 'Spread']].to_csv(
    CSV_DIR / 'quotes_cleaned.csv', index=False
)

print(f"  Saved to {CSV_DIR}/")

# ============================================================================
# 8. Diagnostic Plots
# ============================================================================

print("\n" + "-" * 60)
print("8. Creating Diagnostic Plots")
print("-" * 60)

# Plot 1: Realized Variance Time Series
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes[0]
for i, day in enumerate(trading_days):
    data = variance_5min[variance_5min['Date'] == day]
    ax1.plot(data['DateTime'], data['Variance'], 'b-', alpha=0.7, lw=0.8)
    if i < len(trading_days) - 1:
        ax1.axvline(data['DateTime'].max(), color='gray', ls='--', alpha=0.3)

ax1.set_ylabel('Realized Variance (5-min)')
ax1.set_title('Realized Variance Time Series')
ax1.grid(True, alpha=0.3)

# Plot 2: Variance Distribution
ax2 = axes[1]
ax2.hist(variance_5min['Variance'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(variance_5min['Variance'].mean(), color='r', ls='--', label='Mean')
ax2.axvline(variance_5min['Variance'].median(), color='g', ls='--', label='Median')
ax2.set_xlabel('Realized Variance')
ax2.set_ylabel('Frequency')
ax2.set_title('Variance Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'realized_variance.png', dpi=150)
plt.close()

# Plot 2: Spread Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(spread_5min['Spread_Mean'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(spread_5min['Spread_Mean'].mean(), color='r', ls='--', label='Mean')
ax.set_xlabel('Average Spread ($)')
ax.set_ylabel('Frequency')
ax.set_title('5-min Average Spread Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'spread.png', dpi=150)
plt.close()

print(f"  Plots saved to {PLOTS_DIR}/")

print("\n" + "=" * 70)
print("✅ Step 1 Complete!")
print("=" * 70)
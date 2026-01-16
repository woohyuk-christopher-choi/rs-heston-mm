#!/usr/bin/env python3
"""
Unknown Trades 분석
- Unknown이 발생하는 이유 파악
- 데이터 품질 문제인지, 실제로 다른 곳에서 체결인지 확인
"""

import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "output" / "csv"

print("=" * 70)
print("Unknown Trades Analysis")
print("=" * 70)

# Load data
quotes = pd.read_csv(CSV_DIR / 'quotes_cleaned.csv', parse_dates=['DateTime'])
trades = pd.read_csv(CSV_DIR / 'trades_classified.csv', parse_dates=['DateTime'])

print(f"\nLoaded: {len(quotes):,} quotes, {len(trades):,} trades")

# Match trades to quotes
trades = trades.sort_values('DateTime').reset_index(drop=True)
quotes = quotes.sort_values('DateTime').reset_index(drop=True)

trades = pd.merge_asof(
    trades,
    quotes[['DateTime', 'Bid', 'Ask', 'Mid', 'Spread']].rename(
        columns={'Bid': 'Q_Bid', 'Ask': 'Q_Ask', 'Mid': 'Q_Mid', 'Spread': 'Q_Spread'}
    ),
    on='DateTime',
    direction='backward'
)

# Classification
tol = 0.0001
trades['AtAsk'] = (trades['Price'] >= trades['Q_Ask'] - tol)
trades['AtBid'] = (trades['Price'] <= trades['Q_Bid'] + tol)
trades['Unknown'] = ~(trades['AtAsk'] | trades['AtBid'])

# ============================================================================
# 1. Basic Statistics
# ============================================================================

print("\n" + "-" * 60)
print("1. Basic Classification")
print("-" * 60)

n_ask = trades['AtAsk'].sum()
n_bid = trades['AtBid'].sum()
n_unknown = trades['Unknown'].sum()
n_total = len(trades)

print(f"  At Ask:    {n_ask:>10,} ({n_ask/n_total*100:>5.1f}%)")
print(f"  At Bid:    {n_bid:>10,} ({n_bid/n_total*100:>5.1f}%)")
print(f"  Unknown:   {n_unknown:>10,} ({n_unknown/n_total*100:>5.1f}%)")

# ============================================================================
# 2. Where are unknown trades relative to bid/ask?
# ============================================================================

print("\n" + "-" * 60)
print("2. Unknown Trades Location")
print("-" * 60)

unknown = trades[trades['Unknown']].copy()

# Position relative to mid
unknown['RelativeToMid'] = unknown['Price'] - unknown['Q_Mid']
unknown['RelativeToAsk'] = unknown['Price'] - unknown['Q_Ask']
unknown['RelativeToBid'] = unknown['Price'] - unknown['Q_Bid']

print(f"\n  Unknown trades relative to quote:")
print(f"    vs Mid:  mean = ${unknown['RelativeToMid'].mean():.6f}")
print(f"    vs Ask:  mean = ${unknown['RelativeToAsk'].mean():.6f}")
print(f"    vs Bid:  mean = ${unknown['RelativeToBid'].mean():.6f}")

# Categories
unknown['BetweenBidAsk'] = (unknown['Price'] > unknown['Q_Bid']) & (unknown['Price'] < unknown['Q_Ask'])
unknown['BelowBid'] = unknown['Price'] < unknown['Q_Bid']
unknown['AboveAsk'] = unknown['Price'] > unknown['Q_Ask']

n_between = unknown['BetweenBidAsk'].sum()
n_below = unknown['BelowBid'].sum()
n_above = unknown['AboveAsk'].sum()

print(f"\n  Unknown trades breakdown:")
print(f"    Between Bid-Ask: {n_between:>10,} ({n_between/n_unknown*100:>5.1f}%)")
print(f"    Below Bid:       {n_below:>10,} ({n_below/n_unknown*100:>5.1f}%)")
print(f"    Above Ask:       {n_above:>10,} ({n_above/n_unknown*100:>5.1f}%)")

# ============================================================================
# 3. Time lag analysis
# ============================================================================

print("\n" + "-" * 60)
print("3. Quote-Trade Time Lag")
print("-" * 60)

# Time between trade and matched quote
trades['QuoteAge_ms'] = (trades['DateTime'] - trades['DateTime']).dt.total_seconds() * 1000

# For unknown trades, check if a closer quote exists
# Sample analysis on first 10000 unknown trades
sample = unknown.head(10000).copy()

# For each unknown trade, find the quote age
quote_times = quotes['DateTime'].values

def find_quote_age(trade_time):
    """Find time since last quote"""
    idx = np.searchsorted(quote_times, trade_time, side='right') - 1
    if idx >= 0:
        return (trade_time - quote_times[idx]).total_seconds() * 1000
    return np.nan

sample['QuoteAge'] = sample['DateTime'].apply(find_quote_age)

print(f"\n  Quote age for unknown trades (sample of 10,000):")
print(f"    Mean:   {sample['QuoteAge'].mean():.2f} ms")
print(f"    Median: {sample['QuoteAge'].median():.2f} ms")
print(f"    Max:    {sample['QuoteAge'].max():.2f} ms")

# ============================================================================
# 4. Price improvement analysis
# ============================================================================

print("\n" + "-" * 60)
print("4. Price Improvement Analysis")
print("-" * 60)

between = unknown[unknown['BetweenBidAsk']].copy()

if len(between) > 0:
    # How much inside the spread?
    between['SpreadPosition'] = (between['Price'] - between['Q_Bid']) / between['Q_Spread']
    
    print(f"\n  Trades between bid-ask ({len(between):,} trades):")
    print(f"    Position in spread (0=bid, 1=ask):")
    print(f"      Mean:   {between['SpreadPosition'].mean():.3f}")
    print(f"      Median: {between['SpreadPosition'].median():.3f}")
    
    # At midpoint?
    at_mid = ((between['SpreadPosition'] > 0.45) & (between['SpreadPosition'] < 0.55)).sum()
    print(f"    At midpoint (45-55%): {at_mid:,} ({at_mid/len(between)*100:.1f}%)")

# ============================================================================
# 5. Conclusion
# ============================================================================

print("\n" + "-" * 60)
print("5. Conclusion")
print("-" * 60)

print(f"""
  Unknown trades ({n_unknown/n_total*100:.1f}%) breakdown:
  
  1. Between Bid-Ask ({n_between/n_unknown*100:.1f}%):
     - Likely: Midpoint executions, price improvement, hidden liquidity
     - These are NOT hitting the MM's posted quotes
     - 논문 모델에서 제외하는 것이 맞음
     
  2. Outside Bid-Ask ({(n_below+n_above)/n_unknown*100:.1f}%):
     - Likely: Quote stale (already changed), reporting delay
     - 데이터 동기화 문제일 수 있음
     
  Recommendation:
  - Between Bid-Ask trades: 제외 (MM quote hit이 아님)
  - Outside trades: 가까운 quote로 재매칭 시도 가능
""")

# ============================================================================
# 6. Alternative: Relaxed matching
# ============================================================================

print("\n" + "-" * 60)
print("6. Relaxed Matching Test")
print("-" * 60)

# Try different tolerances
for tol in [0.001, 0.005, 0.01]:
    trades['AtAsk_relaxed'] = (trades['Price'] >= trades['Q_Ask'] - tol)
    trades['AtBid_relaxed'] = (trades['Price'] <= trades['Q_Bid'] + tol)
    trades['Matched'] = trades['AtAsk_relaxed'] | trades['AtBid_relaxed']
    
    matched = trades['Matched'].sum()
    print(f"  Tolerance ${tol}: {matched:,} matched ({matched/n_total*100:.1f}%)")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
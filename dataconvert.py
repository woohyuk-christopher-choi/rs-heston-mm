#!/usr/bin/env python3
"""
MSFT CSV 파일을 패턴으로 자동 합치기
"""

import pandas as pd
import glob

# ============== 설정 ==============
quote_pattern = "MSFT_quotes_2013_05_*.csv"  # 모든 quote 파일
trade_pattern = "MSFT_trades_2013_05_*.csv"  # 모든 trade 파일

output_quote = "MSFT_quotes_combined.csv"
output_trade = "MSFT_trades_combined.csv"


# ==================================


def merge_by_pattern(pattern, output_file, data_type):
    """패턴으로 파일 찾아서 합치기"""

    print(f"\n{'=' * 60}")
    print(f"Merging {data_type.upper()}")
    print(f"{'=' * 60}")

    # 파일 찾기
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ No files found matching: {pattern}")
        return None

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")

    # 파일 읽고 합치기
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, parse_dates=['DateTime'])
            dfs.append(df)
            print(f"✓ {file}: {len(df):,} rows")
        except Exception as e:
            print(f"✗ {file}: Error - {e}")

    if not dfs:
        return None

    # 합치기
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('DateTime').reset_index(drop=True)

    # 중복 제거
    combined = combined.drop_duplicates(subset=['DateTime', 'SequenceNumber'], keep='first')

    # 저장
    combined.to_csv(output_file, index=False)

    print(f"\n✓ Saved: {output_file}")
    print(f"  Total: {len(combined):,} rows")
    print(f"  Period: {combined['DateTime'].min()} to {combined['DateTime'].max()}")

    # 날짜별 카운트
    print(f"\n  Daily breakdown:")
    for date, count in combined.groupby(combined['DateTime'].dt.date).size().items():
        print(f"    {date}: {count:,}")

    return combined


if __name__ == "__main__":
    print("MSFT Data Auto-Merger")

    # Quotes
    quotes = merge_by_pattern(quote_pattern, output_quote, "quotes")

    # Trades
    trades = merge_by_pattern(trade_pattern, output_trade, "trades")

    if quotes is not None and trades is not None:
        print(f"\n{'=' * 60}")
        print("✅ Merge completed!")
        print(f"{'=' * 60}")
        print(f"Quote/Trade ratio: {len(quotes) / len(trades):.1f}:1")
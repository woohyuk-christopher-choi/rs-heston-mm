#!/usr/bin/env python3
"""
MSFT Trade ASC 파일들을 CSV로 일괄 변환
"""

import pandas as pd
from pathlib import Path
import glob

# ============== 폴더 경로 ==============
TRADES_DIR = Path("Marketmaking/data/raw/MSFT/TRADE")
OUTPUT_DIR = Path("Marketmaking/data/processed/MSFT/trades")
# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_trade_to_csv(input_path, output_path):
    """Trade ASC를 CSV로 변환"""
    
    columns = [
        'Date', 'Time', 'Price', 'Size',
        'Exchange', 'SaleCondition', 'Field6', 'SequenceNumber',
        'Field8', 'Field9', 'Field10', 'Field11', 'Field12'
    ]
    
    # 파일 읽기
    df = pd.read_csv(
        input_path,
        header=None,
        names=columns,
        parse_dates=False,
        low_memory=False,
        on_bad_lines='warn'
    )
    
    # 데이터 타입 변환
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    
    # Datetime 생성
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], 
        format='%m/%d/%Y %H:%M:%S.%f',
        errors='coerce'
    )
    
    # 필터링
    initial = len(df)
    df = df.dropna(subset=['Price', 'Size', 'DateTime'])
    df = df[(df['Price'] > 0) & (df['Size'] > 0)]
    
    # Regular trading hours (09:30 - 16:00)
    df = df[(df['DateTime'].dt.hour >= 9) & 
            ((df['DateTime'].dt.hour < 16) | 
             ((df['DateTime'].dt.hour == 16) & (df['DateTime'].dt.minute == 0)))]
    df = df[~((df['DateTime'].dt.hour == 9) & (df['DateTime'].dt.minute < 30))]
    
    # 주요 컬럼만 선택
    output_columns = [
        'DateTime', 'Date', 'Time',
        'Price', 'Size',
        'Exchange', 'SaleCondition', 'SequenceNumber'
    ]
    
    df_clean = df[output_columns].copy()
    df_clean = df_clean.sort_values('DateTime').reset_index(drop=True)
    
    # Dollar Volume 계산
    df_clean['DollarVolume'] = df_clean['Price'] * df_clean['Size']
    
    # CSV 저장
    df_clean.to_csv(output_path, index=False)
    
    return df_clean, initial, len(df_clean)


if __name__ == "__main__":
    print("=" * 70)
    print("MSFT Trade Batch Converter")
    print("=" * 70)
    
    # 모든 Trade ASC 파일 찾기
    pattern = str(TRADES_DIR / "MSFT_*_X_T.asc")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"\n❌ No files found matching: {pattern}")
        exit(1)
    
    print(f"\nFound {len(files)} trade files:")
    for f in files:
        print(f"  - {Path(f).name}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print()
    
    total_initial = 0
    total_cleaned = 0
    results = []
    
    for i, input_file in enumerate(files, 1):
        input_path = Path(input_file)
        
        # 날짜 추출 (예: MSFT_2013_05_01_X_T.asc → 2013_05_01)
        filename = input_path.stem  # MSFT_2013_05_01_X_T
        date_part = '_'.join(filename.split('_')[1:4])  # 2013_05_01
        
        output_filename = f"MSFT_trades_{date_part}.csv"
        output_path = OUTPUT_DIR / output_filename
        
        print(f"[{i}/{len(files)}] Processing: {input_path.name}")
        
        try:
            df, initial, cleaned = convert_trade_to_csv(input_path, output_path)
            
            total_initial += initial
            total_cleaned += cleaned
            
            print(f"  ✓ {initial:,} → {cleaned:,} trades")
            print(f"  ✓ Saved: {output_filename}")
            
            results.append({
                'file': input_path.name,
                'date': date_part,
                'initial': initial,
                'cleaned': cleaned,
                'removed': initial - cleaned,
                'time_range': f"{df['DateTime'].min()} to {df['DateTime'].max()}",
                'total_volume': df['Size'].sum(),
                'mean_price': df['Price'].mean()
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # 요약
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nProcessed {len(results)} files:")
    for r in results:
        print(f"  {r['date']}: {r['cleaned']:,} trades "
              f"(vol: {r['total_volume']:,} shares, avg: ${r['mean_price']:.2f})")
    
    print(f"\nTotal:")
    print(f"  Initial rows: {total_initial:,}")
    print(f"  Cleaned rows: {total_cleaned:,}")
    print(f"  Removed: {total_initial - total_cleaned:,} ({(total_initial - total_cleaned)/total_initial*100:.1f}%)")
    
    # 전체 합친 파일 생성
    if results:
        print(f"\nCreating combined file...")
        
        all_files = [OUTPUT_DIR / f"MSFT_trades_{r['date']}.csv" for r in results]
        dfs = [pd.read_csv(f, parse_dates=['DateTime']) for f in all_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('DateTime').reset_index(drop=True)
        
        combined_path = OUTPUT_DIR.parent / "MSFT_trades_combined.csv"
        combined.to_csv(combined_path, index=False)
        
        print(f"  ✓ Combined: {combined_path}")
        print(f"  ✓ Total rows: {len(combined):,}")
        print(f"  ✓ Total volume: {combined['Size'].sum():,.0f} shares")
        print(f"  ✓ Total dollar volume: ${combined['DollarVolume'].sum():,.0f}")
    
    print("\n✅ All conversions completed!")
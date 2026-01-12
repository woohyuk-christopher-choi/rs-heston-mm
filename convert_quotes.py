#!/usr/bin/env python3
"""
MSFT Quote ASC 파일들을 CSV로 일괄 변환
"""

import pandas as pd
from pathlib import Path
import glob

# ============== 폴더 경로 ==============
QUOTES_DIR = Path("Marketmaking/data/raw/MSFT/QUOTES")
OUTPUT_DIR = Path("Marketmaking/data/processed/MSFT/quotes")
# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_quote_to_csv(input_path, output_path):
    """Quote ASC를 CSV로 변환"""
    
    columns = [
        'Date', 'Time', 'Exchange', 'Bid', 'Ask', 
        'BidSize', 'AskSize', 'QuoteCondition', 'Field8',
        'SequenceNumber', 'BidExchange', 'AskExchange',
        'Field12', 'Field13', 'Field14', 'NationalBBO',
        'Field16', 'Field17', 'Field18', 'Field19', 'Field20'
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
    df['Bid'] = pd.to_numeric(df['Bid'], errors='coerce')
    df['Ask'] = pd.to_numeric(df['Ask'], errors='coerce')
    df['BidSize'] = pd.to_numeric(df['BidSize'], errors='coerce')
    df['AskSize'] = pd.to_numeric(df['AskSize'], errors='coerce')
    
    # Datetime 생성
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], 
        format='%m/%d/%Y %H:%M:%S.%f',
        errors='coerce'
    )
    
    # Mid price와 Spread 계산
    df['Mid'] = (df['Bid'] + df['Ask']) / 2
    df['Spread'] = df['Ask'] - df['Bid']
    
    # 필터링
    initial = len(df)
    df = df.dropna(subset=['Bid', 'Ask', 'DateTime'])
    df = df[(df['Bid'] > 0) & (df['Ask'] > 0) & (df['Spread'] >= 0)]
    df = df[df['Spread'] < 1.0]
    
    # Regular trading hours (09:30 - 16:00)
    df = df[(df['DateTime'].dt.hour >= 9) & 
            ((df['DateTime'].dt.hour < 16) | 
             ((df['DateTime'].dt.hour == 16) & (df['DateTime'].dt.minute == 0)))]
    df = df[~((df['DateTime'].dt.hour == 9) & (df['DateTime'].dt.minute < 30))]
    
    # 주요 컬럼만 선택
    output_columns = [
        'DateTime', 'Date', 'Time', 'Exchange',
        'Bid', 'Ask', 'Mid', 'Spread',
        'BidSize', 'AskSize',
        'QuoteCondition', 'BidExchange', 'AskExchange',
        'NationalBBO', 'SequenceNumber'
    ]
    
    df_clean = df[output_columns].copy()
    df_clean = df_clean.sort_values('DateTime').reset_index(drop=True)
    
    # CSV 저장
    df_clean.to_csv(output_path, index=False)
    
    return df_clean, initial, len(df_clean)


if __name__ == "__main__":
    print("=" * 70)
    print("MSFT Quote Batch Converter")
    print("=" * 70)
    
    # 모든 Quote ASC 파일 찾기
    pattern = str(QUOTES_DIR / "MSFT_*_X_Q.asc")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"\n❌ No files found matching: {pattern}")
        exit(1)
    
    print(f"\nFound {len(files)} quote files:")
    for f in files:
        print(f"  - {Path(f).name}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print()
    
    total_initial = 0
    total_cleaned = 0
    results = []
    
    for i, input_file in enumerate(files, 1):
        input_path = Path(input_file)
        
        # 날짜 추출 (예: MSFT_2013_05_01_X_Q.asc → 2013_05_01)
        filename = input_path.stem  # MSFT_2013_05_01_X_Q
        date_part = '_'.join(filename.split('_')[1:4])  # 2013_05_01
        
        output_filename = f"MSFT_quotes_{date_part}.csv"
        output_path = OUTPUT_DIR / output_filename
        
        print(f"[{i}/{len(files)}] Processing: {input_path.name}")
        
        try:
            df, initial, cleaned = convert_quote_to_csv(input_path, output_path)
            
            total_initial += initial
            total_cleaned += cleaned
            
            print(f"  ✓ {initial:,} → {cleaned:,} quotes")
            print(f"  ✓ Saved: {output_filename}")
            
            results.append({
                'file': input_path.name,
                'date': date_part,
                'initial': initial,
                'cleaned': cleaned,
                'removed': initial - cleaned,
                'time_range': f"{df['DateTime'].min()} to {df['DateTime'].max()}",
                'mean_spread': df['Spread'].mean()
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
        print(f"  {r['date']}: {r['cleaned']:,} quotes (spread: ${r['mean_spread']:.4f})")
    
    print(f"\nTotal:")
    print(f"  Initial rows: {total_initial:,}")
    print(f"  Cleaned rows: {total_cleaned:,}")
    print(f"  Removed: {total_initial - total_cleaned:,} ({(total_initial - total_cleaned)/total_initial*100:.1f}%)")
    
    # 전체 합친 파일 생성
    if results:
        print(f"\nCreating combined file...")
        
        all_files = [OUTPUT_DIR / f"MSFT_quotes_{r['date']}.csv" for r in results]
        dfs = [pd.read_csv(f, parse_dates=['DateTime']) for f in all_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('DateTime').reset_index(drop=True)
        
        combined_path = OUTPUT_DIR.parent / "MSFT_quotes_combined.csv"
        combined.to_csv(combined_path, index=False)
        
        print(f"  ✓ Combined: {combined_path}")
        print(f"  ✓ Total rows: {len(combined):,}")
    
    print("\n✅ All conversions completed!")
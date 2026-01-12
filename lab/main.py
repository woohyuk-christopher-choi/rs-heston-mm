#!/usr/bin/env python3
"""
Master Script: Run Complete Experiment
Regime-Switching Market Making
"""

import sys
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def main():
    print_section("REGIME-SWITCHING MARKET MAKING")

    print("논문: Optimal Market Making under Regime-Switching Heston")
    print("저자: Woohyuk Choi (2026)\n")

    # Check data directory (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please create 'data/' folder and add CSV files")
        sys.exit(1)

    # Check data files
    required_files = ['MSFT_quotes_combined.csv', 'MSFT_trades_combined.csv']
    
    print("Checking data files...")
    for file in required_files:
        filepath = data_dir / file
        if not filepath.exists():
            print(f"  ✗ {file} not found in data/")
            sys.exit(1)
        print(f"  ✓ {file}")
    
    # Create output structure
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "csv").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "parameters").mkdir(exist_ok=True)
    
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── csv/          (cleaned data & aggregated results)")
    print(f"  ├── plots/        (visualizations)")
    print(f"  └── parameters/   (estimated parameters)")
    
    print("\n" + "-" * 80 + "\n")
    
    # Run steps
    steps = [
        ('step1_preprocessing.py', 'Data Preprocessing'),
        ('step2_regime_identification.py', 'Regime Identification'),
        ('step3_intensity_estimation.py', 'Intensity Estimation')
    ]
    
    for script, name in steps:
        print(f"▶ Running: {name}...")

        try:
            script_path = script_dir / script
            # Use subprocess to run each script in its own process
            import subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_dir),
                capture_output=False,
                text=True
            )
            if result.returncode != 0:
                print(f"\n❌ {name} failed with exit code {result.returncode}")
                sys.exit(1)
            print(f"\n✅ {name} complete!\n")
            print("-" * 80)
        except FileNotFoundError as e:
            print(f"\n❌ Script not found: {script_path}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Summary
    print_section("EXPERIMENT COMPLETE")
    
    print("📊 Generated Files:\n")
    
    # List all output files
    output_structure = {
        'csv': [
            'mid_prices_5min.csv',
            'realized_variance_5min.csv',
            'spreads_5min.csv',
            'trades_classified.csv',
            'quotes_cleaned.csv',
            'regime_results.csv'
        ],
        'plots': [
            'realized_variance.png',
            'spread.png',
            'regime.png',
            'intensity_curves.png'
        ],
        'parameters': [
            'heston_parameters.csv',
            'intensity_parameters.csv'
        ]
    }
    
    for folder, files in output_structure.items():
        print(f"\n{folder}/:")
        for file in files:
            filepath = output_dir / folder / file
            if filepath.exists():
                size = filepath.stat().st_size / 1024
                print(f"  ✓ {file} ({size:.1f} KB)")
            else:
                print(f"  ✗ {file} (not generated)")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("""
1. Review results in output/:
   - plots/regime.png: Identified regimes
   - plots/intensity_curves.png: Order arrival functions

2. Check parameters in output/parameters/:
   - heston_parameters.csv: κ, θ, ξ, λ
   - intensity_parameters.csv: A, η

3. Use cleaned data in output/csv/ for further analysis

4. TODO (Steps 4-8):
   - HJB Solver (CI)
   - Wonham Filter (PI)
   - Backtesting
   - CI vs PI Comparison
    """)
    
    print("✅ All experiments completed successfully!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
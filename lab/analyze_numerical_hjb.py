#!/usr/bin/env python3
"""
Analyze and Plot Numerical HJB Results from step4_CI_Problem.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ['lab', 'src'] else SCRIPT_DIR
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PARAMS_DIR = OUTPUT_DIR / "parameters"

def load_numerical_results(rho=0.0):
    """Load numerical HJB results"""
    tag = f"rho_{rho:.2f}".replace(".", "p").replace("-", "m")
    param_dir = PARAMS_DIR / f"ci_{tag}"
    
    print(f"Loading from: {param_dir}")
    
    # Load U array
    U_path = param_dir / "ci_U.npy"
    if not U_path.exists():
        print(f"❌ File not found: {U_path}")
        return None
    
    U_all = np.load(U_path)
    print(f"✅ Loaded U: shape={U_all.shape}")
    
    # Load metadata
    meta_path = param_dir / "ci_hjb_solution_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"✅ Loaded metadata")
    else:
        meta = None
        print("⚠️  No metadata found")
    
    # Load spreads
    spreads_path = param_dir / "ci_spreads_t0.csv"
    if spreads_path.exists():
        df_spreads = pd.read_csv(spreads_path)
        print(f"✅ Loaded spreads: {len(df_spreads)} rows")
    else:
        df_spreads = None
        print("⚠️  No spreads found")
    
    # Load diagnostics
    diag_path = param_dir / "ci_diagnostics.json"
    if diag_path.exists():
        with open(diag_path) as f:
            diagnostics = json.load(f)
        print(f"✅ Loaded diagnostics")
    else:
        diagnostics = None
        print("⚠️  No diagnostics found")
    
    return {
        'U': U_all,
        'meta': meta,
        'spreads': df_spreads,
        'diagnostics': diagnostics,
    }

def analyze_U_structure(U_all, meta):
    """Analyze the structure of U"""
    print("\n" + "="*70)
    print("VALUE FUNCTION U ANALYSIS")
    print("="*70)
    
    if meta:
        cfg = meta.get('cfg', {})
        print(f"\nGrid structure:")
        print(f"  Time steps (Nt): {cfg.get('Nt', 'unknown')}")
        print(f"  Inventory (Q_max): {cfg.get('Q_max', 'unknown')}")
        print(f"  Variance grid (Nv): {cfg.get('Nv', 'unknown')}")
        print(f"  v range: [{cfg.get('v_min', '?'):.2e}, {cfg.get('v_max', '?'):.2e}]")
    
    print(f"\nU array shape: {U_all.shape}")
    print(f"  Interpretation: (Nt+1, 2 regimes, Nq, Nv)")
    
    Nt_plus_1, N_regimes, Nq, Nv = U_all.shape
    
    print(f"\nU statistics:")
    print(f"  Overall range: [{np.min(U_all):.6f}, {np.max(U_all):.6f}]")
    print(f"  Mean: {np.mean(U_all):.6f}")
    print(f"  Std: {np.std(U_all):.6f}")
    
    # Terminal condition (t=T, should be close to 1)
    U_T = U_all[-1]  # Last time step
    print(f"\nTerminal condition (t=T):")
    print(f"  U(T) range: [{np.min(U_T):.6f}, {np.max(U_T):.6f}]")
    print(f"  Should be ≈ 1.0: {'✅' if np.abs(np.mean(U_T) - 1.0) < 0.01 else '❌'}")
    
    # Initial time (t=0)
    U_0 = U_all[0]
    print(f"\nInitial time (t=0):")
    print(f"  U(0) range: [{np.min(U_0):.6f}, {np.max(U_0):.6f}]")
    
    # Check regime differences
    print(f"\nRegime comparison at t=0:")
    qi_mid = Nq // 2  # q=0 position
    vi_mid = Nv // 2  # Middle variance
    
    U_low = U_0[0, qi_mid, vi_mid]
    U_high = U_0[1, qi_mid, vi_mid]
    
    print(f"  Low regime (q=0, v=middle): {U_low:.6f}")
    print(f"  High regime (q=0, v=middle): {U_high:.6f}")
    print(f"  Ratio (High/Low): {U_high/U_low:.4f}")
    
    return U_0

def plot_value_function(U_0, meta, save_name="ci_numerical_U_analysis.png"):
    """Create comprehensive plots of U"""
    
    cfg = meta.get('cfg', {}) if meta else {}
    Q_max = cfg.get('Q_max', 10)
    Nv = cfg.get('Nv', 220)
    v_min = cfg.get('v_min', 1e-8)
    v_max = cfg.get('v_max', 1e-3)
    
    # Grids
    q_grid = np.arange(-Q_max, Q_max + 1)
    if cfg.get('v_grid') == 'log':
        v_grid = np.logspace(np.log10(v_min), np.log10(v_max), Nv)
    else:
        v_grid = np.linspace(v_min, v_max, Nv)
    
    Nq = len(q_grid)
    qi_mid = Nq // 2
    vi_mid = Nv // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. U vs q (at middle v)
    ax = axes[0, 0]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        ax.plot(q_grid, U_0[regime, :, vi_mid], '-o', ms=4, label=name)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("U(t=0, v=mid, q)")
    ax.set_title("Value Function vs Inventory")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. -ln(U) vs q (should be approximately quadratic)
    ax = axes[0, 1]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        U_slice = np.clip(U_0[regime, :, vi_mid], 1e-10, np.inf)
        ax.plot(q_grid, -np.log(U_slice), '-o', ms=4, label=name)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("-ln(U)")
    ax.set_title("-ln(U) vs q (should be ≈ quadratic)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. U vs v (at q=0)
    ax = axes[0, 2]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        ax.plot(v_grid * 1e5, U_0[regime, qi_mid, :], '-', lw=2, label=name)
    ax.set_xlabel("Variance (×10⁻⁵)")
    ax.set_ylabel("U(t=0, q=0, v)")
    ax.set_title("Value Function vs Variance")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # 4. U heatmap - Low regime
    ax = axes[1, 0]
    im = ax.contourf(q_grid, v_grid * 1e5, U_0[0, :, :].T, levels=20, cmap='viridis')
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Variance (×10⁻⁵)")
    ax.set_title("U(t=0) - Low Regime")
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax)
    
    # 5. U heatmap - High regime
    ax = axes[1, 1]
    im = ax.contourf(q_grid, v_grid * 1e5, U_0[1, :, :].T, levels=20, cmap='viridis')
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Variance (×10⁻⁵)")
    ax.set_title("U(t=0) - High Regime")
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax)
    
    # 6. Regime difference
    ax = axes[1, 2]
    diff = U_0[1, :, :] - U_0[0, :, :]
    im = ax.contourf(q_grid, v_grid * 1e5, diff.T, levels=20, cmap='RdBu_r')
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Variance (×10⁻⁵)")
    ax.set_title("U(High) - U(Low)")
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {PLOTS_DIR / save_name}")

def analyze_spreads(df_spreads):
    """Analyze optimal spreads"""
    if df_spreads is None:
        print("\n❌ No spreads data available")
        return
    
    print("\n" + "="*70)
    print("OPTIMAL SPREADS ANALYSIS")
    print("="*70)
    
    print(f"\nDataFrame shape: {df_spreads.shape}")
    print(f"Columns: {df_spreads.columns.tolist()}")
    
    # Spreads at q=0
    print("\nSpreads at q=0:")
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        df_regime = df_spreads[(df_spreads['regime'] == regime) & (df_spreads['q'] == 0)]
        
        if len(df_regime) > 0:
            # Take middle variance point
            vi_mid = len(df_regime) // 2
            row = df_regime.iloc[vi_mid]
            
            ask = row.get('delta_ask_cents', row.get('ask_cents', np.nan))
            bid = row.get('delta_bid_cents', row.get('bid_cents', np.nan))
            
            print(f"  {name}: ask={ask:.4f}¢, bid={bid:.4f}¢, total={ask+bid:.4f}¢")
    
    # Range of spreads
    print("\nSpread ranges:")
    for col in ['delta_ask_cents', 'delta_bid_cents', 'ask_cents', 'bid_cents']:
        if col in df_spreads.columns:
            print(f"  {col}: [{df_spreads[col].min():.4f}, {df_spreads[col].max():.4f}]¢")

def plot_spreads(df_spreads, meta, save_name="ci_numerical_spreads_analysis.png"):
    """Plot optimal spreads"""
    if df_spreads is None:
        print("\n❌ Cannot plot spreads - no data")
        return
    
    cfg = meta.get('cfg', {}) if meta else {}
    Q_max = cfg.get('Q_max', 10)
    Nv = cfg.get('Nv', 220)
    
    q_grid = np.arange(-Q_max, Q_max + 1)
    qi_mid = len(q_grid) // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get middle variance index for each regime
    vi_mid = Nv // 2
    
    # 1. Ask spreads vs q
    ax = axes[0, 0]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        color = 'blue' if regime == 0 else 'red'
        
        df_regime = df_spreads[df_spreads['regime'] == regime]
        
        # Group by q and take middle v
        spreads_q = []
        q_vals = []
        for q in q_grid:
            df_q = df_regime[df_regime['q'] == q]
            if len(df_q) > 0:
                vi = len(df_q) // 2
                ask = df_q.iloc[vi].get('delta_ask_cents', df_q.iloc[vi].get('ask_cents', np.nan))
                if not np.isnan(ask):
                    spreads_q.append(ask)
                    q_vals.append(q)
        
        if spreads_q:
            ax.plot(q_vals, spreads_q, '-o', color=color, ms=4, label=name)
    
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Ask Half-Spread (cents)")
    ax.set_title("Optimal Ask Spreads")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Bid spreads vs q
    ax = axes[0, 1]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        color = 'blue' if regime == 0 else 'red'
        
        df_regime = df_spreads[df_spreads['regime'] == regime]
        
        spreads_q = []
        q_vals = []
        for q in q_grid:
            df_q = df_regime[df_regime['q'] == q]
            if len(df_q) > 0:
                vi = len(df_q) // 2
                bid = df_q.iloc[vi].get('delta_bid_cents', df_q.iloc[vi].get('bid_cents', np.nan))
                if not np.isnan(bid):
                    spreads_q.append(bid)
                    q_vals.append(q)
        
        if spreads_q:
            ax.plot(q_vals, spreads_q, '--s', color=color, ms=4, alpha=0.7, label=name)
    
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Bid Half-Spread (cents)")
    ax.set_title("Optimal Bid Spreads")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Total spread
    ax = axes[1, 0]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        color = 'blue' if regime == 0 else 'red'
        
        df_regime = df_spreads[df_spreads['regime'] == regime]
        
        total_q = []
        q_vals = []
        for q in q_grid:
            df_q = df_regime[df_regime['q'] == q]
            if len(df_q) > 0:
                vi = len(df_q) // 2
                row = df_q.iloc[vi]
                ask = row.get('delta_ask_cents', row.get('ask_cents', np.nan))
                bid = row.get('delta_bid_cents', row.get('bid_cents', np.nan))
                if not np.isnan(ask) and not np.isnan(bid):
                    total_q.append(ask + bid)
                    q_vals.append(q)
        
        if total_q:
            ax.plot(q_vals, total_q, '-o', color=color, ms=4, label=name)
    
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Total Spread (cents)")
    ax.set_title("Total Spread = Ask + Bid")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Spread asymmetry
    ax = axes[1, 1]
    for regime in [0, 1]:
        name = "Low" if regime == 0 else "High"
        color = 'blue' if regime == 0 else 'red'
        
        df_regime = df_spreads[df_spreads['regime'] == regime]
        
        asymm_q = []
        q_vals = []
        for q in q_grid:
            df_q = df_regime[df_regime['q'] == q]
            if len(df_q) > 0:
                vi = len(df_q) // 2
                row = df_q.iloc[vi]
                ask = row.get('delta_ask_cents', row.get('ask_cents', np.nan))
                bid = row.get('delta_bid_cents', row.get('bid_cents', np.nan))
                if not np.isnan(ask) and not np.isnan(bid):
                    asymm_q.append(ask - bid)
                    q_vals.append(q)
        
        if asymm_q:
            ax.plot(q_vals, asymm_q, '-o', color=color, ms=4, label=name)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Ask - Bid (cents)")
    ax.set_title("Spread Asymmetry")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {PLOTS_DIR / save_name}")

def main():
    """Main analysis"""
    print("="*70)
    print("NUMERICAL HJB RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    results = load_numerical_results(rho=0.0)
    
    if results is None or results['U'] is None:
        print("\n❌ Failed to load results")
        return
    
    # Analyze U
    U_0 = analyze_U_structure(results['U'], results['meta'])
    
    # Plot U
    plot_value_function(U_0, results['meta'])
    
    # Analyze spreads
    if results['spreads'] is not None:
        analyze_spreads(results['spreads'])
        plot_spreads(results['spreads'], results['meta'])
    
    # Diagnostics
    if results['diagnostics']:
        print("\n" + "="*70)
        print("SOLVER DIAGNOSTICS")
        print("="*70)
        diag = results['diagnostics']
        print(f"\nPolicy iteration:")
        print(f"  Mean iterations: {diag.get('policy_iters_mean', 'N/A'):.1f}")
        print(f"  Max iterations: {diag.get('policy_iters_max', 'N/A')}")
        print(f"  Final error (mean): {diag.get('policy_final_error_mean', 'N/A'):.2e}")
        
        print(f"\nU range:")
        print(f"  Min: {diag.get('U_min_overall', 'N/A'):.6f}")
        print(f"  Max: {diag.get('U_max_overall', 'N/A'):.6f}")
        
        if 'residual_t0' in diag and diag['residual_t0'] is not None:
            res = diag['residual_t0']
            if isinstance(res, dict):
                print(f"\nHJB residual at t=0:")
                for k, v in res.items():
                    print(f"  {k}: {v:.2e}")
            else:
                print(f"\nHJB residual at t=0: {res:.2e}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
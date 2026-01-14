#!/usr/bin/env python3
"""
Step 4: CI HJB Solver - FINAL CORRECT VERSION

Paper Theorem 7.8, Equation (59):
0 = ∂_t U + drift + diffusion + (γ²q²v/2)U + regime - H_ask - H_bid

This is a FORWARD PDE. For backward time stepping from T to 0:
- Time index: n runs from Nt (T) down to 0 (t=0)
- At step n: we have U^{n+1} and compute U^n
- Discretization: ∂_t U ≈ (U^{n+1} - U^n)/dt

Rearranging:
U^n = U^{n+1} - dt*[drift + diffusion + (γ²q²v/2)U + regime - H]

Key points:
1. Reaction term (γ²q²v/2)U goes on RHS with NEGATIVE sign
2. Hamiltonians are SUBTRACTED (negative sign in PDE)
3. Use semi-implicit: reaction and diffusion on RHS evaluated at U^{n+1}
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name != 'home' else Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PARAMS_DIR = OUTPUT_DIR / "parameters"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "T_hours": 6.5,
    "dt_seconds": 300.0,    # 5 minutes for stability
    
    "Q_max": 5,
    
    "v_min": 2e-5,          # Increased minimum
    "v_max": 1.5e-4,        # Reduced maximum  
    "Nv": 15,
    
    "gamma": 0.01,
    
    "kappa_L": 10.0105,
    "theta_L": 0.000016,
    "kappa_H": 10.2948,
    "theta_H": 0.000107,
    "xi": 0.0305,
    "rho": -0.5,
    
    "lambda_LH": 2.3032 / (6.5 * 3600),
    "lambda_HL": 5.6140 / (6.5 * 3600),
    
    "eps": 1e-12,
    "max_delta": 1.0,
}

def load_intensity_params(path_csv):
    df = pd.read_csv(path_csv)
    params = {}
    for _, r in df.iterrows():
        regime = int(r['Regime']) if 'Regime' in df.columns else int(r['regime'])
        side = str(r['Side'] if 'Side' in df.columns else r['side']).strip().lower()
        A = float(r['A'] if 'A' in df.columns else r['a'])
        eta = float(r['eta'] if 'eta' in df.columns else r['Eta'])
        params[(regime, side)] = {"A": A, "eta": eta}
    return params

def optimal_spread_ask(U_q, U_qm1, gamma, eta):
    eps = CONFIG["eps"]
    U_q = max(U_q, eps)
    U_qm1 = max(U_qm1, eps)
    
    base = (1.0/gamma) * np.log(1.0 + gamma/eta)
    ratio = np.clip(U_qm1 / U_q, eps, 1.0/eps)
    inventory_adj = (1.0/gamma) * np.log(ratio)
    
    delta = base + inventory_adj
    return max(0.0, min(delta, CONFIG["max_delta"]))

def optimal_spread_bid(U_q, U_qp1, gamma, eta):
    eps = CONFIG["eps"]
    U_q = max(U_q, eps)
    U_qp1 = max(U_qp1, eps)
    
    base = (1.0/gamma) * np.log(1.0 + gamma/eta)
    ratio = np.clip(U_qp1 / U_q, eps, 1.0/eps)
    inventory_adj = (1.0/gamma) * np.log(ratio)
    
    delta = base + inventory_adj
    return max(0.0, min(delta, CONFIG["max_delta"]))

def hamiltonian_ask(U_q, U_qm1, A, eta, gamma):
    eps = CONFIG["eps"]
    U_q = max(U_q, eps)
    U_qm1 = max(U_qm1, eps)
    
    delta = optimal_spread_ask(U_q, U_qm1, gamma, eta)
    lam = A * np.exp(-eta * delta)
    gain = U_q - np.exp(-gamma * delta) * U_qm1
    
    H = lam * gain
    return max(H, 0.0) if np.isfinite(H) else 0.0

def hamiltonian_bid(U_q, U_qp1, A, eta, gamma):
    eps = CONFIG["eps"]
    U_q = max(U_q, eps)
    U_qp1 = max(U_qp1, eps)
    
    delta = optimal_spread_bid(U_q, U_qp1, gamma, eta)
    lam = A * np.exp(-eta * delta)
    gain = U_q - np.exp(-gamma * delta) * U_qp1
    
    H = lam * gain
    return max(H, 0.0) if np.isfinite(H) else 0.0

def run_ci_hjb(intensity_params):
    """
    Solve CI HJB backward from T to 0
    
    Paper Eq. 59 (forward PDE):
    0 = ∂_t U + [diffusion] + (γ²q²v/2)U + [regime] - H_ask - H_bid
    
    Backward stepping:
    U^n = U^{n+1} - dt*{[diffusion at n+1] + (γ²q²v/2)U^{n+1} 
                        + [regime at n+1] - H[U^{n+1}]}
    """
    print("=" * 70)
    print("Step 4: CI HJB Solver - Equation 59")
    print("=" * 70)
    
    T = CONFIG["T_hours"] * 3600
    dt = CONFIG["dt_seconds"]
    Nt = int(T / dt)
    
    Q = CONFIG["Q_max"]
    q_grid = np.arange(-Q, Q + 1)
    Nq = len(q_grid)
    
    v_grid = np.linspace(CONFIG["v_min"], CONFIG["v_max"], CONFIG["Nv"])
    Nv = len(v_grid)
    dv = v_grid[1] - v_grid[0]
    
    gamma = CONFIG["gamma"]
    xi = CONFIG["xi"]
    rho = CONFIG["rho"]
    
    kappa = {0: CONFIG["kappa_L"], 1: CONFIG["kappa_H"]}
    theta = {0: CONFIG["theta_L"], 1: CONFIG["theta_H"]}
    
    lam_LH = CONFIG["lambda_LH"]
    lam_HL = CONFIG["lambda_HL"]
    
    A_a = {0: intensity_params[(0, 'buy')]['A'], 1: intensity_params[(1, 'buy')]['A']}
    eta_a = {0: intensity_params[(0, 'buy')]['eta'], 1: intensity_params[(1, 'buy')]['eta']}
    A_b = {0: intensity_params[(0, 'sell')]['A'], 1: intensity_params[(1, 'sell')]['A']}
    eta_b = {0: intensity_params[(0, 'sell')]['eta'], 1: intensity_params[(1, 'sell')]['eta']}
    
    print(f"\nConfig:")
    print(f"  T = {T/3600:.1f}h = {Nt} steps (dt = {dt/60:.1f}min)")
    print(f"  Inventory: q ∈ [{-Q}, {Q}] ({Nq} points)")
    print(f"  Variance: v ∈ [{CONFIG['v_min']:.2e}, {CONFIG['v_max']:.2e}] ({Nv} points, dv={dv:.2e})")
    print(f"  γ = {gamma}")
    
    print(f"\nIntensities:")
    for i in [0, 1]:
        name = "Low" if i == 0 else "High"
        print(f"  {name}: A_a={A_a[i]:.4f}, η_a={eta_a[i]:.2f}, A_b={A_b[i]:.4f}, η_b={eta_b[i]:.2f}")
    
    cfl = 0.5 * xi**2 * v_grid.max() * dt / dv**2
    print(f"\nCFL = {cfl:.1f}")
    
    # Initialize
    U = np.ones((2, Nq, Nv))
    
    print(f"\nBackward solve: T={T/3600:.1f}h → t=0")
    report_interval = max(1, Nt // 10)
    
    for n in range(Nt):
        U_new = np.zeros_like(U)
        
        for regime in [0, 1]:
            other = 1 - regime
            q_ij = lam_LH if regime == 0 else lam_HL
            
            for qi in range(Nq):
                q = q_grid[qi]
                
                for vi in range(Nv):
                    v = v_grid[vi]
                    
                    # Start with U^{n+1}
                    U_curr = U[regime, qi, vi]
                    
                    # Diffusion term: (1/2)ξ²v ∂_vv U
                    # Use central difference
                    if 0 < vi < Nv - 1:
                        d2U_dv2 = (U[regime, qi, vi+1] - 2*U[regime, qi, vi] + U[regime, qi, vi-1]) / dv**2
                        diffusion = 0.5 * xi**2 * v * d2U_dv2
                    else:
                        diffusion = 0.0
                    
                    # Drift term: κ(θ-v) ∂_v U
                    drift_coeff = kappa[regime] * (theta[regime] - v)
                    if 0 < vi < Nv - 1:
                        if drift_coeff >= 0:
                            dU_dv = (U[regime, qi, vi+1] - U[regime, qi, vi]) / dv
                        else:
                            dU_dv = (U[regime, qi, vi] - U[regime, qi, vi-1]) / dv
                        drift = drift_coeff * dU_dv
                    else:
                        drift = 0.0
                    
                    # Cross term: -ρξγqv ∂_v U
                    cross_coeff = -rho * xi * gamma * q * v
                    if 0 < vi < Nv - 1:
                        dU_dv_central = (U[regime, qi, vi+1] - U[regime, qi, vi-1]) / (2*dv)
                        cross = cross_coeff * dU_dv_central
                    else:
                        cross = 0.0
                    
                    # Reaction: (1/2)γ²q²v U
                    reaction = 0.5 * gamma**2 * q**2 * v * U_curr
                    
                    # Regime switching: Σ q_ij(U_j - U_i)
                    regime_switch = q_ij * (U[other, qi, vi] - U_curr)
                    
                    # Hamiltonians (evaluated at current U)
                    H_ask = 0.0
                    H_bid = 0.0
                    
                    if qi > 0:
                        H_ask = hamiltonian_ask(U_curr, U[regime, qi-1, vi],
                                               A_a[regime], eta_a[regime], gamma)
                    
                    if qi < Nq - 1:
                        H_bid = hamiltonian_bid(U_curr, U[regime, qi+1, vi],
                                               A_b[regime], eta_b[regime], gamma)
                    
                    # Paper Eq. 59: 0 = ∂_t U + [terms] - H_ask - H_bid
                    # So: ∂_t U = -[terms] + H_ask + H_bid
                    # Backward: U^n = U^{n+1} - dt*(∂_t U)
                    #         = U^{n+1} - dt*{-[terms] + H}
                    #         = U^{n+1} + dt*[terms] - dt*H
                    
                    RHS_terms = diffusion + drift + cross + reaction + regime_switch
                    
                    U_new[regime, qi, vi] = U_curr - dt * RHS_terms + dt * (H_ask + H_bid)
                    
                    # Enforce positivity
                    U_new[regime, qi, vi] = max(U_new[regime, qi, vi], CONFIG["eps"])
        
        U = U_new
        
        # Check
        if np.any(~np.isfinite(U)):
            print(f"\n❌ NaN at step {n+1}/{Nt}")
            return None, q_grid, v_grid
        
        if np.max(U) > 1e10:
            print(f"\n❌ Explosion at step {n+1}/{Nt}, max={np.max(U):.2e}")
            return None, q_grid, v_grid
        
        if np.max(U) < 1e-10:
            print(f"\n❌ Collapse to zero at step {n+1}/{Nt}")
            return None, q_grid, v_grid
        
        if (n + 1) % report_interval == 0:
            pct = 100 * (n + 1) / Nt
            print(f"  {pct:5.1f}%: U ∈ [{np.min(U):.6f}, {np.max(U):.6f}], mean={np.mean(U):.6f}")
    
    print(f"\n✅ Complete: U ∈ [{np.min(U):.6f}, {np.max(U):.6f}]")
    return U, q_grid, v_grid

def compute_optimal_spreads(U, q_grid, v_grid, intensity_params, gamma):
    Nq, Nv = len(q_grid), len(v_grid)
    
    eta_a = {0: intensity_params[(0, 'buy')]['eta'], 1: intensity_params[(1, 'buy')]['eta']}
    eta_b = {0: intensity_params[(0, 'sell')]['eta'], 1: intensity_params[(1, 'sell')]['eta']}
    
    spreads_ask = np.full((2, Nq, Nv), np.nan)
    spreads_bid = np.full((2, Nq, Nv), np.nan)
    
    for regime in [0, 1]:
        for qi in range(Nq):
            for vi in range(Nv):
                if qi > 0:
                    spreads_ask[regime, qi, vi] = optimal_spread_ask(
                        U[regime, qi, vi], U[regime, qi-1, vi], gamma, eta_a[regime])
                
                if qi < Nq - 1:
                    spreads_bid[regime, qi, vi] = optimal_spread_bid(
                        U[regime, qi, vi], U[regime, qi+1, vi], gamma, eta_b[regime])
    
    return spreads_ask, spreads_bid

def plot_results(U, q_grid, v_grid, spreads_ask, spreads_bid):
    vi_mid = len(v_grid) // 2
    qi_zero = np.where(q_grid == 0)[0][0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Value vs inventory
    ax = axes[0, 0]
    for r in [0, 1]:
        ax.plot(q_grid, U[r, :, vi_mid], '-o', ms=4, label=["Low", "High"][r])
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("U(t=0)")
    ax.set_title("Value Function vs Inventory")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Value vs variance
    ax = axes[0, 1]
    for r in [0, 1]:
        ax.plot(v_grid * 1e4, U[r, qi_zero, :], '-', lw=2, label=["Low", "High"][r])
    ax.set_xlabel("Variance (×10⁻⁴)")
    ax.set_ylabel("U(t=0, q=0)")
    ax.set_title("Value Function vs Variance")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Spreads
    ax = axes[1, 0]
    for r in [0, 1]:
        color = ['blue', 'red'][r]
        name = ["Low", "High"][r]
        
        mask_a = np.isfinite(spreads_ask[r, :, vi_mid])
        mask_b = np.isfinite(spreads_bid[r, :, vi_mid])
        
        ax.plot(q_grid[mask_a], spreads_ask[r, mask_a, vi_mid] * 100,
                '-o', color=color, ms=4, label=f"{name} ask")
        ax.plot(q_grid[mask_b], spreads_bid[r, mask_b, vi_mid] * 100,
                '--s', color=color, ms=4, alpha=0.7, label=f"{name} bid")
    
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Half-Spread (cents)")
    ax.set_title("Optimal Spreads")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Total spread
    ax = axes[1, 1]
    for r in [0, 1]:
        total = spreads_ask[r, :, vi_mid] + spreads_bid[r, :, vi_mid]
        valid = np.isfinite(total)
        ax.plot(q_grid[valid], total[valid] * 100, '-o', ms=4, label=["Low", "High"][r])
    
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Total Spread (cents)")
    ax.set_title("Total Spread")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ci_hjb_final.png", dpi=150)
    plt.close()
    print(f"\nPlot: {PLOTS_DIR}/ci_hjb_final.png")

def main():
    intensity_csv = PARAMS_DIR / "intensity_parameters_side_eta.csv"
    
    if not intensity_csv.exists():
        print(f"❌ Missing {intensity_csv}")
        return
    
    intensity_params = load_intensity_params(intensity_csv)
    
    print("\nIntensity parameters loaded:")
    for (r, s), p in intensity_params.items():
        print(f"  {'Low' if r==0 else 'High'} {s}: A={p['A']:.4f}, η={p['eta']:.2f}")
    
    result = run_ci_hjb(intensity_params)
    
    if result[0] is None:
        print("\n❌ Solver failed")
        return
    
    U, q_grid, v_grid = result
    
    spreads_ask, spreads_bid = compute_optimal_spreads(
        U, q_grid, v_grid, intensity_params, CONFIG["gamma"])
    
    # Report
    vi_mid = len(v_grid) // 2
    qi_zero = np.where(q_grid == 0)[0][0]
    
    print("\n" + "=" * 70)
    print("RESULTS at t=0, v=v_mid")
    print("=" * 70)
    
    print(f"\nAt q=0:")
    for r in [0, 1]:
        name = "Low" if r == 0 else "High"
        ask = spreads_ask[r, qi_zero, vi_mid] * 100
        bid = spreads_bid[r, qi_zero, vi_mid] * 100
        u = U[r, qi_zero, vi_mid]
        print(f"  {name}: U={u:.6f}, ask={ask:.4f}¢, bid={bid:.4f}¢, total={ask+bid:.4f}¢")
    
    # Save
    results = []
    for r in [0, 1]:
        for qi, q in enumerate(q_grid):
            results.append({
                'regime': r,
                'regime_name': 'Low' if r == 0 else 'High',
                'q': q,
                'U_v_mid': U[r, qi, vi_mid],
                'ask_cents': spreads_ask[r, qi, vi_mid] * 100,
                'bid_cents': spreads_bid[r, qi, vi_mid] * 100,
            })
    
    df = pd.DataFrame(results)
    df.to_csv(PARAMS_DIR / "ci_optimal_spreads_final.csv", index=False)
    print(f"\nSaved: {PARAMS_DIR}/ci_optimal_spreads_final.csv")
    
    plot_results(U, q_grid, v_grid, spreads_ask, spreads_bid)
    
    print("\n" + "=" * 70)
    print("✅ Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
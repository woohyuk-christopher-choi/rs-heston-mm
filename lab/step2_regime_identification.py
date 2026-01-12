#!/usr/bin/env python3
"""
Step 2: Regime Identification with HMM
논문 Section 4.1: Hidden Regime Process

Key corrections:
1. dt = 5 min / (6.5 hours * 60 min) for trading hours
2. Proper transition rate calculation from HMM transition matrix
3. CIR parameter estimation with appropriate bounds
4. Feller condition verification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy.optimize import minimize
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Time constants (논문 Section 3.1)
TRADING_HOURS = 6.5
INTERVAL_MINUTES = 5
INTERVALS_PER_DAY = int(TRADING_HOURS * 60 / INTERVAL_MINUTES)  # 78

# dt in units of trading days
# 1 trading day = 6.5 hours = 390 minutes = 78 intervals
dt = 1.0 / INTERVALS_PER_DAY  # ≈ 0.0128 days per interval

# Folder structure
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CSV_DIR = OUTPUT_DIR / "csv"
PLOTS_DIR = OUTPUT_DIR / "plots"
PARAMS_DIR = OUTPUT_DIR / "parameters"

PARAMS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Step 2: Regime Identification (HMM)")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Trading hours: {TRADING_HOURS}h")
print(f"  Interval: {INTERVAL_MINUTES} min")
print(f"  Intervals per day: {INTERVALS_PER_DAY}")
print(f"  dt (trading days): {dt:.6f}")

# ============================================================================
# 1. Load Variance Data
# ============================================================================

print("\n" + "-" * 60)
print("1. Loading Variance Data")
print("-" * 60)

variance = pd.read_csv(CSV_DIR / 'realized_variance_5min.csv', parse_dates=['DateTime'])
print(f"  Loaded {len(variance):,} observations")
print(f"  Raw variance range: [{variance['Variance'].min():.6e}, {variance['Variance'].max():.6e}]")

# Scale variance for better HMM performance
# Use log transformation to handle the wide range and small values
variance['Variance_raw'] = variance['Variance'].copy()
variance['LogVariance'] = np.log(variance['Variance'] + 1e-10)

print(f"  Log variance range: [{variance['LogVariance'].min():.2f}, {variance['LogVariance'].max():.2f}]")

# ============================================================================
# 2. Fit Hidden Markov Model
# ============================================================================
"""
논문 Definition 4.1: X = {X_t} is a continuous-time Markov chain on S = {H, L}

HMM identifies two regimes based on variance levels:
- Low variance regime (calm market)
- High variance regime (volatile market)

We use LOG(variance) for HMM fitting because:
1. Variance has a wide range with many small values
2. Log transformation makes the distribution more Gaussian-like
3. Better separation of regimes
"""

print("\n" + "-" * 60)
print("2. Fitting Hidden Markov Model")
print("-" * 60)

# Use log variance for HMM
X = variance['LogVariance'].values.reshape(-1, 1)

# Initialize HMM with better starting values based on percentiles
low_init = np.percentile(X, 30)
high_init = np.percentile(X, 70)
print(f"\n  Initial estimates (log scale):")
print(f"    Low mean guess:  {low_init:.2f}")
print(f"    High mean guess: {high_init:.2f}")

# Fit 2-state Gaussian HMM with custom initialization
model = hmm.GaussianHMM(
    n_components=2, 
    covariance_type="full", 
    n_iter=1000, 
    random_state=42,
    init_params='stc'  # Initialize startprob, transmat, covars; set means manually
)

# Set initial means
model.means_ = np.array([[low_init], [high_init]])

model.fit(X)
states = model.predict(X)

# Ensure state 0 = Low variance, state 1 = High variance
if model.means_[0] > model.means_[1]:
    states = 1 - states
    model.means_ = model.means_[::-1]
    model.covars_ = model.covars_[::-1]
    model.transmat_ = model.transmat_[::-1, :][:, ::-1]

variance['Regime'] = states

# Report regime statistics
n_low = (states == 0).sum()
n_high = (states == 1).sum()
print(f"\n  Regime distribution:")
print(f"    Low (0):  {n_low:,} ({n_low/len(states)*100:.1f}%)")
print(f"    High (1): {n_high:,} ({n_high/len(states)*100:.1f}%)")

print(f"\n  Regime means (log variance):")
print(f"    Low:  {model.means_[0][0]:.2f} → exp = {np.exp(model.means_[0][0]):.6f}")
print(f"    High: {model.means_[1][0]:.2f} → exp = {np.exp(model.means_[1][0]):.6f}")

print(f"\n  Regime std (log scale):")
print(f"    Low:  {np.sqrt(model.covars_[0][0][0]):.4f}")
print(f"    High: {np.sqrt(model.covars_[1][0][0]):.4f}")

# Calculate actual variance statistics per regime
v_low_actual = variance[variance['Regime'] == 0]['Variance_raw']
v_high_actual = variance[variance['Regime'] == 1]['Variance_raw']
print(f"\n  Actual variance statistics:")
print(f"    Low regime:  mean = {v_low_actual.mean():.6e}, std = {v_low_actual.std():.6e}")
if len(v_high_actual) > 0:
    print(f"    High regime: mean = {v_high_actual.mean():.6e}, std = {v_high_actual.std():.6e}")

# ============================================================================
# 3. Calculate Transition Rates
# ============================================================================
"""
논문 Eq. 1: Generator matrix Q = [[-λ_HL, λ_HL], [λ_LH, -λ_LH]]

From discrete transition probability P(dt) to continuous rate λ:
P_ii(dt) = exp(-λ_ij * dt) approximately for small dt
=> λ_ij = -log(P_ii) / dt

Where:
- λ_LH: rate of transition from Low to High (intensity of regime change L→H)
- λ_HL: rate of transition from High to Low (intensity of regime change H→L)
"""

print("\n" + "-" * 60)
print("3. Transition Rates (논문 Eq. 1)")
print("-" * 60)

trans_mat = model.transmat_
print(f"\n  HMM transition matrix P(dt):")
print(f"    P(L→L) = {trans_mat[0,0]:.6f}, P(L→H) = {trans_mat[0,1]:.6f}")
print(f"    P(H→L) = {trans_mat[1,0]:.6f}, P(H→H) = {trans_mat[1,1]:.6f}")

# Convert to continuous-time rates (per trading day)
# λ_LH: Low → High transition rate
# λ_HL: High → Low transition rate
lambda_LH = -np.log(trans_mat[0, 0]) / dt if trans_mat[0, 0] > 0 else 0
lambda_HL = -np.log(trans_mat[1, 1]) / dt if trans_mat[1, 1] > 0 else 0

# Expected duration in each regime (in trading days)
duration_L = 1 / lambda_LH if lambda_LH > 0 else np.inf
duration_H = 1 / lambda_HL if lambda_HL > 0 else np.inf

print(f"\n  Transition rates (per trading day):")
print(f"    λ_LH (Low→High): {lambda_LH:.4f}")
print(f"    λ_HL (High→Low): {lambda_HL:.4f}")

print(f"\n  Expected regime duration:")
print(f"    Low regime:  {duration_L:.4f} days ({duration_L * TRADING_HOURS * 60:.1f} min)")
print(f"    High regime: {duration_H:.4f} days ({duration_H * TRADING_HOURS * 60:.1f} min)")

# Stationary distribution
pi_L = lambda_HL / (lambda_LH + lambda_HL)
pi_H = lambda_LH / (lambda_LH + lambda_HL)
print(f"\n  Stationary distribution:")
print(f"    π_L: {pi_L:.4f}")
print(f"    π_H: {pi_H:.4f}")

# ============================================================================
# 4. Estimate Heston (CIR) Parameters
# ============================================================================
"""
논문 Eq. 6: dV_t = κ_{X_t}(θ_{X_t} - V_t)dt + ξ√V_t dW^V_t

CIR process parameters:
- κ (kappa): mean-reversion speed
- θ (theta): long-run variance level
- ξ (xi): volatility of variance (vol-of-vol)

Feller condition (Assumption 4.7): 2κθ > ξ² ensures V_t > 0
"""

print("\n" + "-" * 60)
print("4. Heston (CIR) Parameter Estimation (논문 Eq. 6)")
print("-" * 60)

def estimate_cir_params(v_series, regime_name, dt):
    """
    Estimate CIR parameters using quasi-maximum likelihood.
    
    For CIR: dV = κ(θ - V)dt + ξ√V dW
    Discretized: V_{t+1} - V_t ≈ κ(θ - V_t)dt + ξ√(V_t dt) ε
    
    Log-likelihood (Gaussian approximation):
    L = -0.5 Σ [(V_{t+1} - V_t - drift)² / variance + log(variance)]
    """
    v = np.array(v_series)
    n = len(v)
    
    if n < 10:
        print(f"  {regime_name}: Insufficient data ({n} points)")
        return None
    
    # Initial guesses based on data
    v_mean = np.mean(v)
    v_std = np.std(v)
    
    def neg_log_likelihood(params):
        kappa, theta, xi = params
        
        # Feller condition penalty
        if 2 * kappa * theta <= xi**2:
            return 1e10
        
        if kappa <= 0 or theta <= 0 or xi <= 0:
            return 1e10
        
        v_prev = v[:-1]
        v_next = v[1:]
        
        # Expected change: E[dV] = κ(θ - V)dt
        drift = kappa * (theta - v_prev) * dt
        
        # Variance of change: Var[dV] = ξ²V dt
        variance = xi**2 * np.maximum(v_prev, 1e-10) * dt
        
        # Gaussian log-likelihood
        residuals = v_next - v_prev - drift
        ll = -0.5 * np.sum(residuals**2 / variance + np.log(variance))
        
        return -ll  # Negative for minimization
    
    # Bounds based on realistic parameter ranges
    # κ: mean-reversion speed (0.1 to 50 per day seems reasonable)
    # θ: long-run variance (based on data range)
    # ξ: vol-of-vol (typically 0.1 to 5)
    bounds = [
        (0.1, 50),           # kappa
        (v_mean * 0.1, v_mean * 3),  # theta (around observed mean)
        (0.01, 5)            # xi
    ]
    
    # Multiple starting points for robustness
    best_result = None
    best_ll = np.inf
    
    for kappa_init in [1, 5, 10]:
        for xi_init in [0.5, 1, 2]:
            x0 = [kappa_init, v_mean, xi_init * v_std]
            
            try:
                result = minimize(
                    neg_log_likelihood,
                    x0=x0,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000}
                )
                
                if result.fun < best_ll:
                    best_ll = result.fun
                    best_result = result
            except:
                continue
    
    if best_result is None:
        print(f"  {regime_name}: Optimization failed")
        return None
    
    kappa, theta, xi = best_result.x
    
    # Feller condition check
    feller_ratio = 2 * kappa * theta / (xi**2)
    feller_satisfied = feller_ratio > 1
    
    print(f"\n  {regime_name} regime:")
    print(f"    κ (mean-reversion): {kappa:.4f}")
    print(f"    θ (long-run var):   {theta:.6f}")
    print(f"    ξ (vol-of-vol):     {xi:.4f}")
    print(f"    Feller ratio (2κθ/ξ²): {feller_ratio:.4f} {'✓' if feller_satisfied else '✗'}")
    
    return {
        'kappa': kappa,
        'theta': theta,
        'xi': xi,
        'feller_ratio': feller_ratio,
        'feller_satisfied': feller_satisfied
    }

# Estimate for each regime (use raw variance, not log)
v_low = variance[variance['Regime'] == 0]['Variance_raw'].values
v_high = variance[variance['Regime'] == 1]['Variance_raw'].values

params_low = estimate_cir_params(v_low, "Low", dt)
params_high = estimate_cir_params(v_high, "High", dt)

# 논문 Remark 4.9: ξ is regime-independent for tractable filtering
if params_low and params_high:
    xi_common = (params_low['xi'] + params_high['xi']) / 2
    print(f"\n  Common ξ (regime-independent): {xi_common:.4f}")

# ============================================================================
# 5. Save Parameters
# ============================================================================

print("\n" + "-" * 60)
print("5. Saving Parameters")
print("-" * 60)

params_dict = {
    'kappa_L': params_low['kappa'] if params_low else np.nan,
    'theta_L': params_low['theta'] if params_low else np.nan,
    'kappa_H': params_high['kappa'] if params_high else np.nan,
    'theta_H': params_high['theta'] if params_high else np.nan,
    'xi': xi_common if (params_low and params_high) else np.nan,
    'lambda_LH': lambda_LH,
    'lambda_HL': lambda_HL,
    'duration_L_days': duration_L,
    'duration_H_days': duration_H,
    'pi_L': pi_L,
    'pi_H': pi_H,
    'feller_L': params_low['feller_ratio'] if params_low else np.nan,
    'feller_H': params_high['feller_ratio'] if params_high else np.nan
}

params_df = pd.DataFrame([params_dict])
params_df.to_csv(PARAMS_DIR / 'heston_parameters.csv', index=False)

# Save regime results with raw variance
variance['Variance'] = variance['Variance_raw']  # Use raw variance for output
variance[['DateTime', 'Date', 'Variance', 'Regime']].to_csv(
    CSV_DIR / 'regime_results.csv', index=False
)

print(f"  Saved to {PARAMS_DIR}/")

# ============================================================================
# 6. Diagnostic Plots
# ============================================================================

print("\n" + "-" * 60)
print("6. Creating Diagnostic Plots")
print("-" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Variance with regimes
ax1 = axes[0]
trading_days = sorted(variance['Date'].unique())
for i, day in enumerate(trading_days):
    data = variance[variance['Date'] == day]
    
    low = data[data['Regime'] == 0]
    high = data[data['Regime'] == 1]
    
    if len(low) > 0:
        ax1.scatter(low['DateTime'], low['Variance_raw'], c='blue', s=10, alpha=0.5, label='Low' if i == 0 else '')
    if len(high) > 0:
        ax1.scatter(high['DateTime'], high['Variance_raw'], c='red', s=10, alpha=0.5, label='High' if i == 0 else '')
    
    if i < len(trading_days) - 1:
        ax1.axvline(data['DateTime'].max(), color='gray', ls='--', alpha=0.3)

ax1.set_ylabel('Variance')
ax1.set_title('Realized Variance with Regime Labels')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Regime indicator
ax2 = axes[1]
ax2.fill_between(variance['DateTime'], 0, 1, where=(variance['Regime'] == 1), 
                  alpha=0.3, color='red', label='High')
ax2.fill_between(variance['DateTime'], 0, 1, where=(variance['Regime'] == 0), 
                  alpha=0.3, color='blue', label='Low')
ax2.set_ylabel('Regime')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Low', 'High'])
ax2.set_title('Regime Identification')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Variance distribution by regime
ax3 = axes[2]
ax3.hist(v_low, bins=30, alpha=0.7, label=f'Low (n={len(v_low)})', color='blue', density=True)
ax3.hist(v_high, bins=30, alpha=0.7, label=f'High (n={len(v_high)})', color='red', density=True)
ax3.axvline(np.mean(v_low), color='blue', ls='--', lw=2)
ax3.axvline(np.mean(v_high), color='red', ls='--', lw=2)
ax3.set_xlabel('Variance')
ax3.set_ylabel('Density')
ax3.set_title('Variance Distribution by Regime')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'regime.png', dpi=150)
plt.close()

print(f"  Plots saved to {PLOTS_DIR}/")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary: Estimated Parameters")
print("=" * 70)
print(f"""
  Regime Switching (논문 Eq. 1):
    λ_LH = {lambda_LH:.4f} /day (Low → High)
    λ_HL = {lambda_HL:.4f} /day (High → Low)

  Heston Dynamics (논문 Eq. 6):
    Low regime:  κ_L = {params_dict['kappa_L']:.4f}, θ_L = {params_dict['theta_L']:.6f}
    High regime: κ_H = {params_dict['kappa_H']:.4f}, θ_H = {params_dict['theta_H']:.6f}
    Common:      ξ = {params_dict['xi']:.4f}

  Stationary Distribution:
    π_L = {pi_L:.4f}, π_H = {pi_H:.4f}
""")

print("=" * 70)
print("✅ Step 2 Complete!")
print("=" * 70)
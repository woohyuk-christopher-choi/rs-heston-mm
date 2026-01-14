#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 (Paper-grade): CI HJB Solver under Regime-Switching Heston

Solves the full CI reduced HJB for U_i(t,v,q), i in {Low, High}:

0 = ∂_t U_i
  + kappa_i (theta_i - v) ∂_v U_i
  + 0.5 xi^2 v ∂_{vv} U_i
  - rho xi gamma q v ∂_v U_i
  + 0.5 gamma^2 q^2 v U_i
  + sum_{j!=i} lambda_{ij} (U_j - U_i)
  - H_a^*(U_i(q), U_i(q-1))
  - H_b^*(U_i(q), U_i(q+1))

with terminal U_i(T, v, q) = 1.

Uses:
- Backward Euler in time (monotone & stable)
- Upwind for first derivative in v (convection)
- Central difference for second derivative in v (diffusion)
- Policy iteration: given U^k compute optimal deltas & Hamiltonians, then solve linear BE system for U^{k+1}
- Hard inventory bounds with one-sided quoting at boundaries

Inputs (from your pipeline):
- parameters/heston_parameters.csv  (kappa_L, theta_L, kappa_H, theta_H, xi, rho, lambda_LH, lambda_HL)
- parameters/intensity_parameters_side_eta.csv OR intensity_for_hjb_side_eta.csv
  expecting A, eta for (regime, side). Here we treat "buy" as ask-side fills (at ask),
  and "sell" as bid-side fills (at bid), consistent with your Step3 summary.

Outputs:
- parameters/ci_hjb_solution_meta.json
- parameters/ci_U.npy  (shape: Nt+1, 2, Nq, Nv)
- parameters/ci_spreads_t0.csv
- plots/ci_spreads_t0_vs_q.png
- plots/ci_spreads_surface_t0_low.png, plots/ci_spreads_surface_t0_high.png
- parameters/ci_diagnostics.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ----------------------------
# Paths
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["lab", "src"] else SCRIPT_DIR
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PARAMS_DIR = OUTPUT_DIR / "parameters"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Configuration
# ----------------------------
@dataclass(frozen=True)
class CIConfig:
    # Horizon
    T_seconds: float = 6.5 * 3600.0   # full trading day (6.5h)
    Nt: int = 78                      # 5-min steps (6.5h*60/5 = 78)
    # Inventory
    Q_max: int = 10                   # q in [-Q_max, Q_max]
    # Volatility grid
    Nv: int = 220
    v_min: float = 1e-8
    v_max: float = 1e-3               # should cover RV tail; can increase & do truncation test
    v_grid: str = "log"               # "log" or "linear"
    # Risk aversion
    gamma: float = 0.01
    # Policy iteration
    policy_max_iter: int = 100
    policy_tol: float = 1e-7
    # 🔥 NEW: rho sweep
    rho_list: tuple = (-0.7, -0.5, -0.3, 0.0)
    # Safety
    eps: float = 1e-14
    clip_U_max: float = 1e6           # avoid blow-ups
    # Boundary condition in v
    v_bc: str = "neumann"             # "neumann" only implemented (∂_v U = 0 at v_min,v_max)

CFG = CIConfig()


# ----------------------------
# IO helpers
# ----------------------------
def load_heston_params(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()

    def _get(names):
        for n in names:
            if n in row:
                return float(row[n])
        raise KeyError(f"Missing any of {names} in {path.name}")

    out = {
        "kappa_L": _get(["kappa_L", "κ_L"]),
        "theta_L": _get(["theta_L", "θ_L"]),
        "kappa_H": _get(["kappa_H", "κ_H"]),
        "theta_H": _get(["theta_H", "θ_H"]),
        "xi": _get(["xi", "ξ"]),
        "lambda_LH": _get(["lambda_LH", "λ_LH"]),
        "lambda_HL": _get(["lambda_HL", "λ_HL"]),
    }

    # lambda unit normalization
    for k in ["lambda_LH", "lambda_HL"]:
        if out[k] > 0.05:
            out[k] = out[k] / (6.5 * 3600.0)

    return out



def load_intensity_params(path: Path) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Load intensity parameters.

    We expect rows with:
      regime (0=Low,1=High), side in {'buy','sell'} and columns A, eta
    We'll map:
      ask-side intensity uses side='buy'  (trades at ask are buys)
      bid-side intensity uses side='sell' (trades at bid are sells)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def colpick(*cands: str) -> str:
        for c in cands:
            if c in cols:
                return cols[c]
        raise KeyError(f"Missing columns in {path.name}. Need one of {cands}. Found {list(df.columns)}")

    c_reg = colpick("regime", "regime_id")
    c_side = colpick("side")
    c_A = colpick("a", "A".lower())
    c_eta = colpick("eta", "η", "Eta".lower())

    params: Dict[Tuple[int, str], Dict[str, float]] = {}
    for _, r in df.iterrows():
        regime = int(r[c_reg])
        side = str(r[c_side]).strip().lower()
        A = float(r[c_A])
        eta = float(r[c_eta])
        params[(regime, side)] = {"A": A, "eta": eta}

    # Basic checks
    for regime in [0, 1]:
        for side in ["buy", "sell"]:
            if (regime, side) not in params:
                raise KeyError(f"Missing intensity params for (regime={regime}, side={side}) in {path.name}")
    return params


# ----------------------------
# Grids
# ----------------------------
def make_time_grid(T: float, Nt: int) -> np.ndarray:
    return np.linspace(0.0, T, Nt + 1)  # 0..T


def make_v_grid(v_min: float, v_max: float, Nv: int, mode: str) -> np.ndarray:
    if mode == "log":
        return np.exp(np.linspace(np.log(v_min), np.log(v_max), Nv))
    if mode == "linear":
        return np.linspace(v_min, v_max, Nv)
    raise ValueError("v_grid must be 'log' or 'linear'")


def make_q_grid(Q_max: int) -> np.ndarray:
    return np.arange(-Q_max, Q_max + 1, dtype=int)


# ----------------------------
# Optimal control (closed-form)
# ----------------------------
def optimal_delta(gamma: float, eta: float, ratio: float) -> float:
    """
    delta* = [ 1/gamma ln(1+gamma/eta) + 1/gamma ln(ratio) ]^+
    """
    base = (1.0 / gamma) * math.log(1.0 + gamma / eta)
    inv_adj = (1.0 / gamma) * math.log(max(ratio, 1e-300))
    d = base + inv_adj
    DELTA_MAX = 0.05   # 5 cents (논문적으로 매우 보수적)
    return min(max(d, 0.0), DELTA_MAX)


# ----------------------------
# Discretization (v-direction)
# ----------------------------
def upwind_first_derivative_coeffs(v: np.ndarray, drift: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each v_m, build coefficients for ∂_v U using upwind:
      if drift_m >= 0:  (U_m - U_{m-1})/dv_minus
      else:             (U_{m+1} - U_m)/dv_plus

    Returns arrays (coef_m_minus, coef_m, coef_m_plus) s.t.
      ∂_v U ≈ coef_- * U_{m-1} + coef_0 * U_m + coef_+ * U_{m+1}

    Boundary: Neumann handled separately by overriding first derivative at endpoints.
    """
    Nv = len(v)
    coef_m1 = np.zeros(Nv)
    coef_0 = np.zeros(Nv)
    coef_p1 = np.zeros(Nv)

    dv = np.diff(v)
    for m in range(1, Nv - 1):
        dv_minus = dv[m - 1]
        dv_plus = dv[m]
        if drift[m] >= 0:
            coef_m1[m] = -1.0 / dv_minus
            coef_0[m] = 1.0 / dv_minus
        else:
            coef_0[m] = -1.0 / dv_plus
            coef_p1[m] = 1.0 / dv_plus

    return coef_m1, coef_0, coef_p1


def second_derivative_coeffs(v: np.ndarray, diffcoef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Central second derivative for nonuniform grid:
      ∂_{vv} U at v_m approximated by:
        a_m U_{m-1} + b_m U_m + c_m U_{m+1}
    multiplied later by 0.5 * xi^2 * v (already in diffcoef if desired)

    Here we compute coefficients for ∂_{vv} only; caller multiplies by scalar diffusion factor.

    Boundary: Neumann handled separately.
    """
    Nv = len(v)
    a = np.zeros(Nv)
    b = np.zeros(Nv)
    c = np.zeros(Nv)

    dv = np.diff(v)
    for m in range(1, Nv - 1):
        h_m = dv[m]       # v_{m+1} - v_m
        h_mm = dv[m - 1]  # v_m - v_{m-1}
        denom = h_mm * h_m * (h_mm + h_m)
        a[m] = 2.0 * h_m / denom
        c[m] = 2.0 * h_mm / denom
        b[m] = -(a[m] + c[m])
    return a, b, c


def apply_neumann_bc_matrix_rows(A: csr_matrix, rhs: np.ndarray, v: np.ndarray, row_indices: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    """
    Impose Neumann BC ∂_v U = 0 at boundaries:
      at m=0: U_0 - U_1 = 0
      at m=Nv-1: U_{Nv-1} - U_{Nv-2} = 0

    We'll overwrite the corresponding rows in A and rhs for all provided row_indices
    (each row index corresponds to a particular (i,q,m) unknown; here we pass the
     global indices for m=0 and m=Nv-1 nodes).

    This keeps scheme monotone and easy to defend.

    Note: Overwriting sparse rows in CSR is expensive if done repeatedly; we build
    the matrix in COO once per policy iteration and then impose BC by row edits
    using LIL internally. For simplicity & robustness, we rebuild per iter anyway.
    """
    from scipy.sparse import lil_matrix
    A_lil = A.tolil()
    Nv = len(v)

    # Each global row corresponds to either m=0 or m=Nv-1.
    for r in row_indices:
        # Determine if this is m=0 or m=Nv-1 based on modulo Nv
        m = r % Nv
        A_lil.rows[r] = []
        A_lil.data[r] = []
        if m == 0:
            # U_0 - U_1 = 0
            A_lil[r, r] = 1.0
            A_lil[r, r + 1] = -1.0
            rhs[r] = 0.0
        elif m == Nv - 1:
            # U_{Nv-1} - U_{Nv-2} = 0
            A_lil[r, r] = 1.0
            A_lil[r, r - 1] = -1.0
            rhs[r] = 0.0
        else:
            raise RuntimeError("apply_neumann_bc_matrix_rows called with non-boundary row.")
    return A_lil.tocsr(), rhs


# ----------------------------
# Indexing utilities for vectorized linear system
# Unknown ordering: x[(i,q,m)] where i in {0,1}, q index 0..Nq-1, m 0..Nv-1
# global index = ((i*Nq + qidx)*Nv + m)
# ----------------------------
def idx(i: int, qidx: int, m: int, Nq: int, Nv: int) -> int:
    return ((i * Nq + qidx) * Nv + m)


# ----------------------------
# Core solver
# ----------------------------
class CIHJBSolver:
    def __init__(self,
                 cfg: CIConfig,
                 heston: Dict[str, float],
                 intensity: Dict[Tuple[int, str], Dict[str, float]],
                 rho: float):                # 👈 NEW

        self.cfg = cfg
        self.heston = heston
        self.intensity = intensity

        # regime parameters
        self.kappa = {0: heston["kappa_L"], 1: heston["kappa_H"]}
        self.theta = {0: heston["theta_L"], 1: heston["theta_H"]}

        self.xi = heston["xi"]
        self.rho = float(rho)                # ✅ 외부 주입

        self.lam = {
            (0, 1): heston["lambda_LH"],
            (1, 0): heston["lambda_HL"],
        }

        # intensity params 그대로
        self.A_ask = {0: intensity[(0, "buy")]["A"], 1: intensity[(1, "buy")]["A"]}
        self.eta_ask = {0: intensity[(0, "buy")]["eta"], 1: intensity[(1, "buy")]["eta"]}
        self.A_bid = {0: intensity[(0, "sell")]["A"], 1: intensity[(1, "sell")]["A"]}
        self.eta_bid = {0: intensity[(0, "sell")]["eta"], 1: intensity[(1, "sell")]["eta"]}

        self.t_grid = make_time_grid(cfg.T_seconds, cfg.Nt)
        self.dt = self.t_grid[1] - self.t_grid[0]
        self.q_grid = make_q_grid(cfg.Q_max)
        self.v_grid = make_v_grid(cfg.v_min, cfg.v_max, cfg.Nv, cfg.v_grid)

        self.Nq = len(self.q_grid)
        self.Nv = len(self.v_grid)
        self.Nx = 2 * self.Nq * self.Nv  # total unknowns per time-slice

        # Precompute second-derivative coefficients (grid-only)
        self.d2_a, self.d2_b, self.d2_c = second_derivative_coeffs(self.v_grid, diffcoef=np.zeros_like(self.v_grid))

    def _compute_policy(self, U_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Given U_slice[i,qidx,m], compute optimal deltas and Hamiltonian contributions.
        Returns:
          delta_ask[i,qidx,m], delta_bid[i,qidx,m], H_ask[i,qidx,m], H_bid[i,qidx,m]
        Boundary (inventory):
          q = -Q -> no ask (cannot sell more): H_ask=0
          q = +Q -> no bid (cannot buy more): H_bid=0
        """
        cfg = self.cfg
        gamma = cfg.gamma
        eps = cfg.eps

        delta_ask = np.zeros((2, self.Nq, self.Nv), dtype=float)
        delta_bid = np.zeros((2, self.Nq, self.Nv), dtype=float)
        H_ask = np.zeros((2, self.Nq, self.Nv), dtype=float)
        H_bid = np.zeros((2, self.Nq, self.Nv), dtype=float)

        for i in [0, 1]:
            eta_a = self.eta_ask[i]
            A_a = self.A_ask[i]
            eta_b = self.eta_bid[i]
            A_b = self.A_bid[i]

            for qidx, q in enumerate(self.q_grid):
                for m in range(self.Nv):
                    Uq = max(U_slice[i, qidx, m], eps)

                    # Ask side: q -> q-1 after sell. Feasible unless q == -Q_max.
                    if qidx > 0:  # q > -Q
                        Uqm1 = max(U_slice[i, qidx - 1, m], eps)
                        ratio = np.clip(Uqm1 / Uq, eps, 1.0 / eps)
                        d = optimal_delta(gamma, eta_a, ratio)
                        delta_ask[i, qidx, m] = d

                        Lam = A_a * math.exp(-eta_a * d)
                        term = Uq - math.exp(-gamma * d) * Uqm1
                        H_ask[i, qidx, m] = Lam * term
                    else:
                        delta_ask[i, qidx, m] = np.nan
                        H_ask[i, qidx, m] = 0.0

                    # Bid side: q -> q+1 after buy. Feasible unless q == +Q_max.
                    if qidx < self.Nq - 1:  # q < +Q
                        Uqp1 = max(U_slice[i, qidx + 1, m], eps)
                        ratio = np.clip(Uqp1 / Uq, eps, 1.0 / eps)
                        d = optimal_delta(gamma, eta_b, ratio)
                        delta_bid[i, qidx, m] = d

                        Lam = A_b * math.exp(-eta_b * d)
                        term = Uq - math.exp(-gamma * d) * Uqp1
                        H_bid[i, qidx, m] = Lam * term
                    else:
                        delta_bid[i, qidx, m] = np.nan
                        H_bid[i, qidx, m] = 0.0

        return delta_ask, delta_bid, H_ask, H_bid

    def _build_linear_system(self, U_next: np.ndarray, H_ask: np.ndarray, H_bid: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build the BE linear system for U at current time step:
          (U - U_next)/dt + L[U] + regime_switch[U] + reaction*U = H_ask + H_bid

        Unknown vector x corresponds to U_current flattened.
        """
        cfg = self.cfg
        dt = self.dt
        gamma = cfg.gamma
        xi = self.xi
        rho = self.rho

        rows = []
        cols = []
        data = []
        rhs = np.zeros(self.Nx, dtype=float)

        v = self.v_grid
        Nv = self.Nv
        Nq = self.Nq

        # Precompute dv coefficients for second derivative, grid-only
        d2_a, d2_b, d2_c = self.d2_a, self.d2_b, self.d2_c

        # For Neumann BC row indices:
        bc_rows = []

        for i in [0, 1]:
            kappa_i = self.kappa[i]
            theta_i = self.theta[i]
            # regime switch intensity to other state
            j = 1 - i
            lam_ij = self.lam[(i, j)]

            for qidx, q in enumerate(self.q_grid):
                # Reaction term: 0.5 * gamma^2 * q^2 * v
                reaction = 0.5 * (gamma ** 2) * (q ** 2) * v  # array Nv

                # Convection drift in v: kappa_i(theta_i - v) - rho*xi*gamma*q*v
                drift = kappa_i * (theta_i - v) - (rho * xi * gamma * q) * v  # array Nv

                # Upwind first-derivative coefficients
                d1_m1, d1_0, d1_p1 = upwind_first_derivative_coeffs(v, drift)

                for m in range(Nv):
                    g = idx(i, qidx, m, Nq, Nv)
                    # RHS: U_next/dt + H_a + H_b
                    rhs[g] = (U_next[i, qidx, m] / dt) + H_ask[i, qidx, m] + H_bid[i, qidx, m]

                    # If boundary in v, set aside for Neumann overwrite
                    if m == 0 or m == Nv - 1:
                        bc_rows.append(g)
                        continue

                    # Diagonal coefficient starts with 1/dt
                    diag = 1.0 / dt

                    # Add reaction + regime switching outflow ( + lam_ij * U_i )
                    diag += reaction[m] + lam_ij

                    # Diffusion: 0.5 xi^2 v * ∂_{vv}
                    diff_factor = 0.5 * (xi ** 2) * v[m]
                    # contributes to m-1, m, m+1
                    rows += [g, g, g]
                    cols += [g - 1, g, g + 1]
                    data += [diff_factor * d2_a[m], diff_factor * d2_b[m], diff_factor * d2_c[m]]

                    # Convection: drift * ∂_v (upwind)
                    # drift[m] * (d1_m1*U_{m-1} + d1_0*U_m + d1_p1*U_{m+1})
                    rows += [g, g, g]
                    cols += [g - 1, g, g + 1]
                    data += [drift[m] * d1_m1[m], drift[m] * d1_0[m], drift[m] * d1_p1[m]]

                    # Add diagonal term
                    rows.append(g); cols.append(g); data.append(diag)

                    # Regime coupling term: + lam_ij * U_j (moved to LHS as -lam_ij U_j? careful)
                    # Our equation includes + lam_ij (U_j - U_i).
                    # We already put +lam_ij on diag (for -lam_ij*U_i moved to LHS).
                    # Need to move +lam_ij*U_j to LHS with coefficient (-lam_ij)?? No:
                    # The PDE term is + lam_ij*(U_j - U_i).
                    # In BE form, LHS has + lam_ij*U_j  and - lam_ij*U_i.
                    # We included +lam_ij in diag for (-lam_ij*U_i)?? actually diag had +lam_ij: that corresponds to +lam_ij*U_i on LHS.
                    # But PDE contributes -lam_ij*U_i on RHS if we move? Let's do systematically:
                    # We want: (1/dt)U + ... + (-lam_ij)*U_i + (+lam_ij)*U_j ??? Wait:
                    # PDE: ... + lam_ij(U_j - U_i)
                    # Move everything except RHS to LHS: ... + lam_ij U_j - lam_ij U_i.
                    # So coefficient on U_i is (-lam_ij), on U_j is (+lam_ij).
                    # Since we put +lam_ij in diag, that's wrong sign. We should put (-lam_ij) in diag.
                    # Similarly for reaction, it's +reaction*U on LHS (correct).
                    #
                    # Let's fix: diag should include (1/dt) + reaction  + ( - lam_ij ) (because -lam_ij U_i).
                    # And add +lam_ij on the coupling column for U_j.
                    #
                    # Correct below: we'll adjust by subtracting 2*lam_ij? easiest: build correctly from scratch.
                    #
                    # We'll do: diag includes (1/dt) + reaction[m] + ( - lam_ij ).
                    # And add off-diagonal coupling to U_j with (+lam_ij).
                    pass

        # The block above had a sign issue; rebuild properly in one pass:
        # (We keep code robust by building again cleanly.)
        # -------------------------------------------------
        rows, cols, data = [], [], []
        rhs = np.zeros(self.Nx, dtype=float)
        bc_rows = []

        for i in [0, 1]:
            kappa_i = self.kappa[i]
            theta_i = self.theta[i]
            j = 1 - i
            lam_ij = self.lam[(i, j)]

            for qidx, q in enumerate(self.q_grid):
                reaction = 0.5 * (gamma ** 2) * (q ** 2) * v
                drift = kappa_i * (theta_i - v) - (rho * xi * gamma * q) * v
                d1_m1, d1_0, d1_p1 = upwind_first_derivative_coeffs(v, drift)

                for m in range(Nv):
                    g = idx(i, qidx, m, Nq, Nv)
                    rhs[g] = (U_next[i, qidx, m] / dt) + H_ask[i, qidx, m] + H_bid[i, qidx, m]

                    if m == 0 or m == Nv - 1:
                        bc_rows.append(g)
                        continue

                    # Diagonal: (1/dt) + reaction  + ( - lam_ij )  + diffusion/convection central parts (added below)
                    diag = (1.0 / dt) + reaction[m] - lam_ij

                    # Diffusion term
                    diff_factor = 0.5 * (xi ** 2) * v[m]
                    # add to m-1,m,m+1
                    rows += [g, g, g]
                    cols += [g - 1, g, g + 1]
                    data += [diff_factor * d2_a[m], diff_factor * d2_b[m], diff_factor * d2_c[m]]

                    # Convection term
                    rows += [g, g, g]
                    cols += [g - 1, g, g + 1]
                    data += [drift[m] * d1_m1[m], drift[m] * d1_0[m], drift[m] * d1_p1[m]]

                    # Add base diag
                    rows.append(g); cols.append(g); data.append(diag)

                    # Regime coupling to other regime at same (qidx,m): +lam_ij * U_j
                    g_other = idx(j, qidx, m, Nq, Nv)
                    rows.append(g); cols.append(g_other); data.append(lam_ij)

        A = coo_matrix((data, (rows, cols)), shape=(self.Nx, self.Nx)).tocsr()

        if self.cfg.v_bc.lower() == "neumann":
            A, rhs = apply_neumann_bc_matrix_rows(A, rhs, self.v_grid, np.array(bc_rows, dtype=int))
        else:
            raise NotImplementedError("Only Neumann BC is implemented for v-boundaries.")

        return A, rhs

    def solve(self) -> Tuple[np.ndarray, Dict]:
        """
        Main backward solver.
        Returns:
          U_all: array shape (Nt+1, 2, Nq, Nv)
          diagnostics: dict
        """
        cfg = self.cfg
        Nt = cfg.Nt
        eps = cfg.eps

        U_all = np.ones((Nt + 1, 2, self.Nq, self.Nv), dtype=float)  # terminal is 1
        diagnostics = {
            "policy_iters": [],
            "policy_errors": [],
            "max_U": [],
            "min_U": [],
        }

        print("=" * 80)
        print("CI HJB Solver (Paper-grade)")
        print("=" * 80)
        print(f"T={cfg.T_seconds/3600:.2f}h, Nt={cfg.Nt}, dt={self.dt:.2f}s")
        print(f"Inventory q in [{-cfg.Q_max},{cfg.Q_max}] (Nq={self.Nq})")
        print(f"v grid: {cfg.v_grid}, Nv={cfg.Nv}, [{cfg.v_min:.2e},{cfg.v_max:.2e}]")
        print(f"gamma={cfg.gamma}, xi={self.xi}, rho={self.rho}")
        print(f"Regime switch (per sec): lambda_LH={self.lam[(0,1)]:.6g}, lambda_HL={self.lam[(1,0)]:.6g}")
        print("-" * 80)

        for n in range(Nt - 1, -1, -1):
            # warm start from next time
            U_next = U_all[n + 1].copy()
            U_k = U_next.copy()

            # policy iteration
            last_err = None
            for k in range(cfg.policy_max_iter):
                delta_a, delta_b, H_a, H_b = self._compute_policy(U_k)

                A, rhs = self._build_linear_system(U_next, H_a, H_b)

                # solve linear system
                x = spsolve(A, rhs)
                U_new = x.reshape((2, self.Nq, self.Nv))

                # enforce positivity + clip
                U_new = np.clip(U_new, eps, cfg.clip_U_max)

                err = float(np.max(np.abs(U_new - U_k)))
                omega = 0.2   # 0.1 ~ 0.3 권장
                U_k = omega * U_new + (1 - omega) * U_k

                if (k + 1) % 10 == 0 or k == 0:
                    print(f"t-step {n:3d}/{Nt-1} | policy iter {k+1:2d} | err={err:.3e} | U∈[{U_k.min():.3e},{U_k.max():.3e}]")

                if err < cfg.policy_tol:
                    last_err = err
                    break
                last_err = err

            U_all[n] = U_k

            diagnostics["policy_iters"].append(k + 1)
            diagnostics["policy_errors"].append(last_err)
            diagnostics["max_U"].append(float(U_k.max()))
            diagnostics["min_U"].append(float(U_k.min()))

        print("-" * 80)
        print("✅ CI solve completed.")
        return U_all, diagnostics

    # ----------------------------
    # Post-processing
    # ----------------------------
    def spreads_from_U(self, U_slice: np.ndarray, t_index: int) -> pd.DataFrame:
        """
        Compute optimal spreads (ask/bid) in cents at a given time index
        for each regime, inventory q, and v.
        """
        cfg = self.cfg
        gamma = cfg.gamma
        eps = cfg.eps

        records = []
        for i in [0, 1]:
            eta_a = self.eta_ask[i]
            eta_b = self.eta_bid[i]
            name = "Low" if i == 0 else "High"

            for qidx, q in enumerate(self.q_grid):
                for m, vv in enumerate(self.v_grid):
                    Uq = max(U_slice[i, qidx, m], eps)

                    # ask
                    if qidx > 0:
                        Uqm1 = max(U_slice[i, qidx - 1, m], eps)
                        da = optimal_delta(gamma, eta_a, Uqm1 / Uq)
                    else:
                        da = np.nan

                    # bid
                    if qidx < self.Nq - 1:
                        Uqp1 = max(U_slice[i, qidx + 1, m], eps)
                        db = optimal_delta(gamma, eta_b, Uqp1 / Uq)
                    else:
                        db = np.nan

                    records.append({
                        "t_index": t_index,
                        "regime": i,
                        "regime_name": name,
                        "q": int(q),
                        "v": float(vv),
                        "U": float(U_slice[i, qidx, m]),
                        "ask_cents": float(da * 100.0) if not np.isnan(da) else np.nan,
                        "bid_cents": float(db * 100.0) if not np.isnan(db) else np.nan,
                        "total_cents": float((da + db) * 100.0) if (not np.isnan(da) and not np.isnan(db)) else np.nan,
                    })

        return pd.DataFrame.from_records(records)

    def plot_spreads_t0_vs_q(self, df_t0: pd.DataFrame, v_target: Optional[float] = None, fname: str = "ci_spreads_t0_vs_q.png") -> None:
        """
        Plot spreads vs q at t=0 for a chosen v slice.
        If v_target is None, use v median.
        """
        if v_target is None:
            v_target = float(np.median(self.v_grid))

        # find closest v
        v_vals = df_t0["v"].values
        v_unique = np.unique(v_vals)
        v0 = v_unique[np.argmin(np.abs(v_unique - v_target))]

        d = df_t0[df_t0["v"] == v0].copy()

        plt.figure(figsize=(10, 6))
        for i, style in [(0, "-o"), (1, "-s")]:
            dd = d[d["regime"] == i]
            plt.plot(dd["q"], dd["ask_cents"], style, markersize=4, label=("Low ask" if i == 0 else "High ask"))
            plt.plot(dd["q"], dd["bid_cents"], style, markersize=4, alpha=0.7, linestyle="--",
                     label=("Low bid" if i == 0 else "High bid"))

        plt.xlabel("Inventory q")
        plt.ylabel("Half-spread (cents)")
        plt.title(f"CI Optimal Spreads at t=0 (v≈{v0:.2e})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / fname, dpi=150)
        plt.close()

    def plot_spread_surface_t0(self, df_t0: pd.DataFrame, regime: int, fname: str) -> None:
        """
        Heatmap-like surface for total spread as a function of (q, v) at t=0.
        """
        d = df_t0[df_t0["regime"] == regime].copy()
        # pivot
        pivot = d.pivot(index="v", columns="q", values="total_cents")
        # ensure sorted
        pivot = pivot.sort_index()

        plt.figure(figsize=(10, 6))
        plt.imshow(pivot.values, aspect="auto", origin="lower",
                   extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
        plt.xlabel("Inventory q")
        plt.ylabel("Variance v")
        plt.title(f"CI Total Spread (cents) at t=0 | Regime={'Low' if regime==0 else 'High'}")
        plt.colorbar(label="Total spread (cents)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / fname, dpi=150)
        plt.close()


# ----------------------------
# Diagnostics: HJB residual (optional but important)
# ----------------------------
def compute_hjb_residual_timestep(
    solver: CIHJBSolver,
    U_curr: np.ndarray,
    U_next: np.ndarray
) -> Dict[str, float]:
    """
    Compute a coarse residual diagnostic at one time step:
      Res = (U_curr - U_next)/dt + (PDE terms evaluated at U_curr with optimal controls)

    This is not used in solving (we already solve the BE equation), but is valuable for reporting.

    Returns max/mean residual over interior v nodes.
    """
    cfg = solver.cfg
    dt = solver.dt
    gamma = cfg.gamma
    xi = solver.xi
    rho = solver.rho
    v = solver.v_grid
    Nv = solver.Nv
    eps = cfg.eps

    max_abs = 0.0
    sum_abs = 0.0
    count = 0

    delta_a, delta_b, H_a, H_b = solver._compute_policy(U_curr)

    # finite differences for v
    # We'll compute ∂v and ∂vv in a simple way consistent with scheme:
    for i in [0, 1]:
        kappa_i = solver.kappa[i]
        theta_i = solver.theta[i]
        j = 1 - i
        lam_ij = solver.lam[(i, j)]

        for qidx, q in enumerate(solver.q_grid):
            reaction = 0.5 * (gamma ** 2) * (q ** 2) * v
            drift = kappa_i * (theta_i - v) - (rho * xi * gamma * q) * v

            # interior nodes only
            dv = np.diff(v)
            for m in range(1, Nv - 1):
                # upwind first derivative
                if drift[m] >= 0:
                    dUdv = (U_curr[i, qidx, m] - U_curr[i, qidx, m - 1]) / dv[m - 1]
                else:
                    dUdv = (U_curr[i, qidx, m + 1] - U_curr[i, qidx, m]) / dv[m]

                # central second derivative (nonuniform)
                h_m = dv[m]
                h_mm = dv[m - 1]
                denom = h_mm * h_m * (h_mm + h_m)
                d2 = 2.0 * (h_m * U_curr[i, qidx, m - 1] - (h_mm + h_m) * U_curr[i, qidx, m] + h_mm * U_curr[i, qidx, m + 1]) / denom

                # PDE terms
                time_term = (U_curr[i, qidx, m] - U_next[i, qidx, m]) / dt
                drift_term = drift[m] * dUdv
                diff_term = 0.5 * (xi ** 2) * v[m] * d2
                reaction_term = reaction[m] * U_curr[i, qidx, m]
                regime_term = lam_ij * (U_curr[j, qidx, m] - U_curr[i, qidx, m])
                ham_term = -(H_a[i, qidx, m] + H_b[i, qidx, m])

                res = time_term + drift_term + diff_term + reaction_term + regime_term + ham_term
                ares = abs(float(res))
                max_abs = max(max_abs, ares)
                sum_abs += ares
                count += 1

    return {
        "residual_max_abs": float(max_abs),
        "residual_mean_abs": float(sum_abs / max(count, 1)),
        "residual_count": int(count),
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("=" * 80)
    print("Step 4: CI HJB (FULL, paper-grade) — rho sweep")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load inputs (once)
    # ------------------------------------------------------------------
    heston_csv = PARAMS_DIR / "heston_parameters.csv"

    intensity_csv_candidates = [
        PARAMS_DIR / "intensity_for_hjb_side_eta.csv",
        PARAMS_DIR / "intensity_parameters_side_eta.csv",
    ]
    intensity_csv = None
    for p in intensity_csv_candidates:
        if p.exists():
            intensity_csv = p
            break
    if intensity_csv is None:
        raise FileNotFoundError(f"Missing intensity csv. Tried: {intensity_csv_candidates}")

    if not heston_csv.exists():
        raise FileNotFoundError(f"Missing {heston_csv}")

    heston = load_heston_params(heston_csv)
    intensity = load_intensity_params(intensity_csv)

    # ------------------------------------------------------------------
    # rho sweep
    # ------------------------------------------------------------------
    for rho in CFG.rho_list:
        print("\n" + "=" * 80)
        print(f"Running CI-HJB for rho = {rho:.2f}")
        print("=" * 80)

        # ----- namespace per rho (중요) -----
        tag = f"rho_{rho:+.2f}".replace(".", "p").replace("+", "")
        param_dir = PARAMS_DIR / f"ci_{tag}"
        plot_dir = PLOTS_DIR / f"ci_{tag}"

        param_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ----- save meta for this rho -----
        meta = {
            "cfg": CFG.__dict__,
            "rho": rho,
            "heston": heston,
            "intensity_file": str(intensity_csv),
            "intensity_params": {f"{k[0]}_{k[1]}": v for k, v in intensity.items()},
        }
        (param_dir / "ci_hjb_solution_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8"
        )

        # ----- solver with rho injected -----
        solver = CIHJBSolver(
            cfg=CFG,
            heston=heston,
            intensity=intensity,
            rho=rho          # 🔥 핵심
        )

        # ----- solve CI-HJB -----
        U_all, diag = solver.solve()

        # ----- save solution -----
        np.save(param_dir / "ci_U.npy", U_all)

        # ----- postprocess at t=0 -----
        df_t0 = solver.spreads_from_U(U_all[0], t_index=0)
        df_t0.to_csv(param_dir / "ci_spreads_t0.csv", index=False)

        # ----- plots -----
        solver.plot_spreads_t0_vs_q(
            df_t0,
            v_target=None,
            fname=f"ci_spreads_t0_vs_q_{tag}.png"
        )
        solver.plot_spread_surface_t0(
            df_t0, regime=0,
            fname=f"ci_spreads_surface_t0_low_{tag}.png"
        )
        solver.plot_spread_surface_t0(
            df_t0, regime=1,
            fname=f"ci_spreads_surface_t0_high_{tag}.png"
        )

        # ----- diagnostics -----
        res0 = compute_hjb_residual_timestep(solver, U_all[0], U_all[1])
        diag_out = {
            "rho": rho,
            "policy_iters_mean": float(np.mean(diag["policy_iters"])) if diag["policy_iters"] else None,
            "policy_iters_max": int(np.max(diag["policy_iters"])) if diag["policy_iters"] else None,
            "policy_final_error_mean": float(np.mean(diag["policy_errors"])) if diag["policy_errors"] else None,
            "U_min_overall": float(np.min(U_all)),
            "U_max_overall": float(np.max(U_all)),
            "residual_t0": res0,
        }
        (param_dir / "ci_diagnostics.json").write_text(
            json.dumps(diag_out, indent=2),
            encoding="utf-8"
        )

        print(f"✅ Finished CI-HJB for rho = {rho:.2f}")

    print("\n" + "=" * 80)
    print("ALL RHO EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

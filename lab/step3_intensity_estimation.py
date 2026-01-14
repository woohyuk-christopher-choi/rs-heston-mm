#!/usr/bin/env python3
"""
Step 3 (V5): Intensity Estimation with SIDE-only η (buy/sell) + Regime-specific A (Low/High)

Goal
-----
We estimate a structurally disciplined intensity model:

    Λ_i^k(δ) = A_i^k * exp(-η^k * δ)

- Regime i ∈ {Low(0), High(1)}
- Side  k ∈ {buy (ask-hit), sell (bid-hit)}
- δ = Half-spread in dollars

Key design choices
------------------
1) η is estimated ONLY by side (pooled across regimes), to stabilize slope identification.
2) δ support is data-driven (quotes-based quantiles), no hard-coded ranges.
3) A is regime-specific, computed from overall rate within each regime & mean δ of fills.
4) We add a regime-wise bin-based shape check to verify that exp(-ηδ) is not grossly violated.

Outputs
-------
- output/parameters/intensity_parameters_side_eta.csv
    (η^buy, η^sell + A_i^k + diagnostics)
- output/parameters/intensity_for_hjb_side_eta.csv
    (minimal HJB input: A_i^k, η^k)
- output/parameters/intensity_binned_rates_side_eta.csv
    (bin audit: exposure/fills/rate per regime/side)
- output/plots/intensity_side_eta_fit.png
- output/plots/intensity_side_eta_fit_log.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CSV_DIR = OUTPUT_DIR / "csv"
PLOTS_DIR = OUTPUT_DIR / "plots"
PARAMS_DIR = OUTPUT_DIR / "parameters"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# Strict quote-hit tolerance (dollars)
PRICE_TOL = 0.0001

# Quote duration filtering
MAX_GAP_SEC = 10.0  # remove big gaps between quote updates

# δ support quantiles (quotes-based)
DELTA_Q_LOW = 0.01
DELTA_Q_HIGH = 0.99

# Binning for η estimation (pooled)
TARGET_BINS_ETA = 12
MIN_EXPOSURE_SEC_ETA = 10.0
MIN_FILLS_ETA = 5

# Binning for regime-wise shape check (per regime/side)
TARGET_BINS_SHAPE = 10
MIN_EXPOSURE_SEC_SHAPE = 10.0
MIN_FILLS_SHAPE = 5


# ============================================================
# Helpers
# ============================================================

def wls_loglinear_fit(delta: np.ndarray, rate: np.ndarray, weight: np.ndarray):
    """
    Weighted LS fit on log scale:
        log(rate) = logA - eta * delta

    Returns:
        A, eta, r2_log, se_logA, se_eta, ok
    """
    mask = (rate > 0) & np.isfinite(rate) & np.isfinite(delta) & (delta >= 0)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    x = delta[mask]
    y = np.log(rate[mask])
    w = weight[mask]
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    X = np.column_stack([np.ones_like(x), x])

    try:
        XtW = (X.T * w)
        XtWX = XtW @ X
        XtWy = XtW @ y
        beta = np.linalg.inv(XtWX) @ XtWy
        b0, b1 = beta
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    y_hat = X @ beta
    y_bar = np.average(y, weights=w)
    ss_tot = np.sum(w * (y - y_bar) ** 2)
    ss_res = np.sum(w * (y - y_hat) ** 2)
    r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # covariance (classic WLS)
    n = len(y)
    p = 2
    dof = max(n - p, 1)
    sigma2 = ss_res / dof
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
        se_b0 = float(np.sqrt(cov[0, 0]))
        se_b1 = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se_b0, se_b1 = np.nan, np.nan

    A = float(np.exp(b0))
    eta = float(-b1)
    return A, eta, float(r2_log), se_b0, se_b1, True


def make_bins_from_quotes(q_delta: pd.Series, low: float, high: float, target_bins: int):
    """
    Quantile-based bins from QUOTES (exposure distribution), restricted to [low, high].
    """
    s = q_delta[(q_delta >= low) & (q_delta <= high)].dropna()
    if len(s) < 2000:
        s = q_delta.dropna()

    probs = np.linspace(0, 1, target_bins + 1)
    edges = np.quantile(s.values, probs)
    edges = np.unique(edges)

    if len(edges) < 4:
        edges = np.linspace(float(np.nanmin(s.values)), float(np.nanmax(s.values)), max(4, target_bins + 1))
        edges = np.unique(edges)

    edges[0] = min(edges[0], low)
    edges[-1] = max(edges[-1], high)
    edges = np.unique(edges)
    return np.sort(edges)


def bin_exposure_and_fills(q: pd.DataFrame, t: pd.DataFrame, bin_edges: np.ndarray):
    """
    Compute per-bin exposure and fills.
    q: HalfSpread, Duration_sec
    t: Quote_Delta
    """
    n_bins = len(bin_edges) - 1
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    qbin = pd.cut(q["HalfSpread"], bins=bin_edges, labels=False, include_lowest=True)
    tbin = pd.cut(t["Quote_Delta"], bins=bin_edges, labels=False, include_lowest=True)

    exposure = q.groupby(qbin)["Duration_sec"].sum().reindex(range(n_bins), fill_value=0.0).values.astype(float)
    fills = t.groupby(tbin).size().reindex(range(n_bins), fill_value=0).values.astype(int)

    rate = fills / np.clip(exposure, 1e-12, None)
    return centers, exposure, fills, rate


def shape_check(centers, exposure, fills, rate, min_exp, min_fills):
    """
    Diagnostics:
    - corr(delta, log(rate)) negative indicates decreasing log-rate with δ.
    - monotone violations count: increases > 5% considered violations.
    """
    mask = (exposure >= min_exp) & (fills >= min_fills) & (rate > 0)
    if mask.sum() < 3:
        return {"shape_nbins": int(mask.sum()), "shape_corr": np.nan, "shape_monotone_viol": np.nan}

    x = centers[mask]
    y = np.log(rate[mask])
    corr = float(np.corrcoef(x, y)[0, 1])

    order = np.argsort(x)
    r_sorted = rate[mask][order]
    viol = int(np.sum((r_sorted[1:] - r_sorted[:-1]) > 0.05 * np.maximum(r_sorted[:-1], 1e-12)))

    return {"shape_nbins": int(mask.sum()), "shape_corr": corr, "shape_monotone_viol": viol}


def estimate_eta_pooled_by_side(quotes: pd.DataFrame, trades_strict: pd.DataFrame, side: str):
    """
    Estimate η^side by pooling ALL regimes, using quote-based bins and exposure/fills rates.
    """
    q = quotes.copy()
    t = trades_strict[trades_strict["ExecSide"] == side].copy()

    # data-driven δ support from quotes
    low = float(q["HalfSpread"].quantile(DELTA_Q_LOW))
    high = float(q["HalfSpread"].quantile(DELTA_Q_HIGH))
    low = max(low, 1e-6)
    high = max(high, low * 1.10)

    # bins from quotes (NOT trades)
    edges = make_bins_from_quotes(q["HalfSpread"], low, high, TARGET_BINS_ETA)
    centers, exposure, fills, rate = bin_exposure_and_fills(q, t, edges)

    valid = (exposure >= MIN_EXPOSURE_SEC_ETA) & (fills >= MIN_FILLS_ETA) & (rate > 0)
    if valid.sum() < 5:
        # adaptive fallback: fewer bins + slightly relaxed thresholds
        for bins, min_exp, min_fill in [(8, 7.0, 4), (6, 5.0, 3), (5, 5.0, 3)]:
            edges = make_bins_from_quotes(q["HalfSpread"], low, high, bins)
            centers, exposure, fills, rate = bin_exposure_and_fills(q, t, edges)
            valid = (exposure >= min_exp) & (fills >= min_fill) & (rate > 0)
            if valid.sum() >= 3:
                break

    if valid.sum() < 3:
        return {
            "side": side,
            "eta": np.nan,
            "A_pooled": np.nan,
            "r2_log": np.nan,
            "delta_support_low": low,
            "delta_support_high": high,
            "n_valid_bins": int(valid.sum()),
            "bins_used": int(len(edges) - 1),
            "bin_centers": centers,
            "bin_exposure": exposure,
            "bin_fills": fills,
            "bin_rates": rate,
            "bin_valid": valid,
        }

    x = centers[valid]
    y = rate[valid]
    w = np.sqrt(np.maximum(fills[valid], 1))

    A, eta, r2_log, se_logA, se_eta, ok = wls_loglinear_fit(x, y, w)
    if (not ok) or (not np.isfinite(eta)) or (eta <= 0) or (not np.isfinite(A)) or (A <= 0):
        A, eta, r2_log = np.nan, np.nan, np.nan

    return {
        "side": side,
        "eta": float(eta),
        "A_pooled": float(A),
        "r2_log": float(r2_log),
        "delta_support_low": low,
        "delta_support_high": high,
        "n_valid_bins": int(valid.sum()),
        "bins_used": int(len(edges) - 1),
        "bin_centers": centers,
        "bin_exposure": exposure,
        "bin_fills": fills,
        "bin_rates": rate,
        "bin_valid": valid,
    }


def estimate_A_regime_side_given_eta(quotes: pd.DataFrame, trades_strict: pd.DataFrame, regime: int, side: str, eta: float):
    """
    Compute regime-specific A_i^side using:
        overall_rate_i^side = fills / exposure
        A_i^side = overall_rate_i^side * exp(eta * mean_delta_fills)
    """
    q = quotes[quotes["Regime"] == regime].copy()
    t = trades_strict[(trades_strict["Regime"] == regime) & (trades_strict["ExecSide"] == side)].copy()

    total_exposure = float(q["Duration_sec"].sum())
    fills = int(len(t))
    overall_rate = float(fills / max(total_exposure, 1e-12))

    mean_delta = float(t["Quote_Delta"].mean()) if fills > 0 else float(q["HalfSpread"].median())

    A = overall_rate * float(np.exp(eta * mean_delta)) if np.isfinite(eta) else np.nan
    return {
        "regime": regime,
        "regime_name": "Low" if regime == 0 else "High",
        "side": side,
        "A": float(A) if np.isfinite(A) else np.nan,
        "overall_rate": overall_rate,
        "mean_delta": mean_delta,
        "total_exposure_sec": total_exposure,
        "total_fills": fills,
    }


def regime_shape_audit(quotes: pd.DataFrame, trades_strict: pd.DataFrame, regime: int, side: str, eta_side: float):
    """
    Regime-wise bin audit + shape check:
    - compute binned empirical rate by regime/side
    - compute correlation & monotonicity diagnostics
    - compute fitted curve using A_i^k (not here) OR using implied A from overall_rate at mean_delta
      For plotting, we'll compute fitted curve later using final A_i^k and eta_side.
    """
    q = quotes[quotes["Regime"] == regime].copy()
    t = trades_strict[(trades_strict["Regime"] == regime) & (trades_strict["ExecSide"] == side)].copy()

    low = float(q["HalfSpread"].quantile(DELTA_Q_LOW))
    high = float(q["HalfSpread"].quantile(DELTA_Q_HIGH))
    low = max(low, 1e-6)
    high = max(high, low * 1.10)

    edges = make_bins_from_quotes(q["HalfSpread"], low, high, TARGET_BINS_SHAPE)
    centers, exposure, fills, rate = bin_exposure_and_fills(q, t, edges)

    diag = shape_check(centers, exposure, fills, rate, MIN_EXPOSURE_SEC_SHAPE, MIN_FILLS_SHAPE)
    return {
        "delta_support_low": low,
        "delta_support_high": high,
        "bins_used": int(len(edges) - 1),
        "bin_centers": centers,
        "bin_exposure": exposure,
        "bin_fills": fills,
        "bin_rates": rate,
        "shape_nbins": diag["shape_nbins"],
        "shape_corr": diag["shape_corr"],
        "shape_monotone_viol": diag["shape_monotone_viol"],
    }


def plot_panels(results, pooled_eta, out_png, log_scale=False):
    """
    Plot per regime/side binned rates with fitted curves using final A_i^k and eta^k.
    results: list of dict with keys:
        regime, regime_name, side, A, eta, plus bin_centers/bin_rates and validity masks if desired.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pos = {(0, "buy"): (0, 0), (0, "sell"): (0, 1), (1, "buy"): (1, 0), (1, "sell"): (1, 1)}

    for r in results:
        ax = axes[pos[(r["regime"], r["side"])]]
        x = r["bin_centers"]
        y = r["bin_rates"]
        exp = r["bin_exposure"]
        fl = r["bin_fills"]

        # validity for display
        valid = (exp >= MIN_EXPOSURE_SEC_SHAPE) & (fl >= MIN_FILLS_SHAPE) & (y > 0)

        if log_scale:
            yy = np.where(y > 0, np.log(y), np.nan)
            ax.scatter(x[valid] * 100, yy[valid], s=70, alpha=0.85, label="log empirical (valid)", zorder=3)
            ax.scatter(x[~valid] * 100, yy[~valid], s=40, alpha=0.25, label="log empirical (other)", zorder=2)

            if np.isfinite(r["A"]) and np.isfinite(r["eta"]) and r["A"] > 0 and r["eta"] > 0:
                x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                y_fit = np.log(r["A"]) - r["eta"] * x_fit
                ax.plot(x_fit * 100, y_fit, lw=2, label=f"fit: logΛ=logA-ηδ (η={r['eta']:.1f})")
            ax.set_ylabel("log(Fill rate)")
        else:
            ax.scatter(x[valid] * 100, y[valid], s=70, alpha=0.85, label="empirical (valid)", zorder=3)
            ax.scatter(x[~valid] * 100, y[~valid], s=40, alpha=0.25, label="empirical (other)", zorder=2)

            if np.isfinite(r["A"]) and np.isfinite(r["eta"]) and r["A"] > 0 and r["eta"] > 0:
                x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                y_fit = r["A"] * np.exp(-r["eta"] * x_fit)
                ax.plot(x_fit * 100, y_fit, lw=2,
                        label=f"fit: A exp(-ηδ)\nA={r['A']:.2f}, η={r['eta']:.1f}")
            ax.set_ylabel("Fill rate (fills/sec)")

        ax.set_xlabel("Half-spread δ (cents)")
        ax.grid(True, alpha=0.3)

        title = f"{r['regime_name']} - {r['side'].capitalize()}"
        ds = f"δ∈[{r['delta_support_low']*100:.2f}¢,{r['delta_support_high']*100:.2f}¢]"
        diag = f"corr={r['shape_corr']:.2f}, viol={r['shape_monotone_viol']}"
        ax.set_title(f"{title}\n{ds}\n{diag}")

        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_appendix_hexbin_with_bins(
    quotes,
    trades_strict,
    final_results,   # V5에서 만든 final_results
    figsize=(14, 10),
    gridsize=35,
    mincnt=1,
    save_path=None
):
    """
    Appendix plot:
    - hexbin of raw trade density
    - overlay all-bin averages (faint)
    - overlay valid-bin averages (bold)
    - overlay exp fit using A_i^k, eta^k (fit used only valid bins)
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    pos = {(0, "buy"): (0, 0), (0, "sell"): (0, 1),
           (1, "buy"): (1, 0), (1, "sell"): (1, 1)}

    for r in final_results:
        ax = axes[pos[(r["regime"], r["side"])]]

        # -------------------------
        # 1) Raw trade density
        # -------------------------
        t = trades_strict[
            (trades_strict["Regime"] == r["regime"]) &
            (trades_strict["ExecSide"] == r["side"])
        ]

        hb = ax.hexbin(
            t["Quote_Delta"] * 100,      # cents
            np.zeros(len(t)),            # dummy y (density only)
            gridsize=gridsize,
            cmap="Greys",
            mincnt=mincnt,
            bins="log",
            alpha=0.5
        )

        # -------------------------
        # 2) Bin averages
        # -------------------------
        x = r["bin_centers"] * 100
        y = r["bin_rates"]
        exp = r["bin_exposure"]
        fl = r["bin_fills"]

        valid = (exp >= 10) & (fl >= 5) & (y > 0)

        # all bins (faint)
        ax.scatter(
            x,
            np.log(y),
            s=40,
            alpha=0.25,
            color="tab:blue",
            label="bin avg (all)"
        )

        # valid bins (bold)
        ax.scatter(
            x[valid],
            np.log(y[valid]),
            s=80,
            alpha=0.9,
            color="tab:blue",
            edgecolor="black",
            label="bin avg (valid)"
        )

        # -------------------------
        # 3) Exp fit (valid bins only used in estimation)
        # -------------------------
        if np.isfinite(r["A"]) and np.isfinite(r["eta"]):
            x_fit = np.linspace(x.min()/100, x.max()/100, 200)
            y_fit = np.log(r["A"]) - r["eta"] * x_fit
            ax.plot(
                x_fit * 100,
                y_fit,
                lw=2,
                color="red",
                label=fr"fit: $\log\Lambda=\log A-\eta\delta$"
            )

        # -------------------------
        # Labels
        # -------------------------
        title = f"{r['regime_name']} – {r['side'].capitalize()}"
        ds = f"δ∈[{r['delta_support_low']*100:.2f}¢,{r['delta_support_high']*100:.2f}¢]"
        diag = f"corr={r['shape_corr']:.2f}, viol={r['shape_monotone_viol']}"

        ax.set_title(f"{title}\n{ds}, {diag}")
        ax.set_xlabel("Half-spread δ (cents)")
        ax.set_ylabel("log(Fill rate)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved appendix figure: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Step 3 (V5): Final Intensity Estimation (SIDE-only η + Regime-specific A)")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1) Load
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("1. Loading Data")
    print("-" * 60)

    quotes = pd.read_csv(CSV_DIR / "quotes_cleaned.csv", parse_dates=["DateTime"])
    trades = pd.read_csv(CSV_DIR / "trades_classified.csv", parse_dates=["DateTime"])
    regimes = pd.read_csv(CSV_DIR / "regime_results.csv", parse_dates=["DateTime"])

    print(f"  Quotes: {len(quotes):,}")
    print(f"  Trades: {len(trades):,}")
    print(f"  Regime obs: {len(regimes):,}")

    # ------------------------------------------------------------
    # 2) Add regime labels (5min) + Quote duration + HalfSpread
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("2. Preparing Data (Regime merge + Quote durations)")
    print("-" * 60)

    quotes["DateTime_5min"] = quotes["DateTime"].dt.floor("5min")
    trades["DateTime_5min"] = trades["DateTime"].dt.floor("5min")
    regimes["DateTime_5min"] = regimes["DateTime"]

    regmap = regimes[["DateTime_5min", "Regime"]].drop_duplicates()

    quotes = quotes.merge(regmap, on="DateTime_5min", how="left")
    trades = trades.merge(regmap, on="DateTime_5min", how="left")

    quotes["Regime"] = quotes["Regime"].ffill().bfill()
    trades["Regime"] = trades["Regime"].ffill().bfill()

    quotes = quotes.dropna(subset=["Regime"]).copy()
    trades = trades.dropna(subset=["Regime"]).copy()
    quotes["Regime"] = quotes["Regime"].astype(int)
    trades["Regime"] = trades["Regime"].astype(int)

    # Quote durations per day
    quotes = quotes.sort_values(["Date", "DateTime"]).reset_index(drop=True)
    quotes["Duration"] = quotes.groupby("Date")["DateTime"].diff().shift(-1)
    quotes["Duration_sec"] = quotes["Duration"].dt.total_seconds()

    quotes = quotes[(quotes["Duration_sec"] > 0) & (quotes["Duration_sec"] < MAX_GAP_SEC)].copy()
    quotes["HalfSpread"] = quotes["Spread"] / 2.0

    print(f"  Valid quotes: {len(quotes):,}")
    print(f"  Total exposure: {quotes['Duration_sec'].sum()/3600:.2f} hours")
    print(f"  Mean duration: {quotes['Duration_sec'].mean()*1000:.2f} ms | Median: {quotes['Duration_sec'].median()*1000:.2f} ms")

    # ------------------------------------------------------------
    # 3) Match trades to active quotes + strict quote-hit classification
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("3. Matching Trades to Active Quotes (Strict quote-hit)")
    print("-" * 60)

    trades = trades.sort_values("DateTime").reset_index(drop=True)

    quotes_for_merge = quotes[["DateTime", "Bid", "Ask", "HalfSpread"]].rename(
        columns={"Bid": "Quote_Bid", "Ask": "Quote_Ask", "HalfSpread": "Quote_Delta"}
    )
    trades = pd.merge_asof(trades, quotes_for_merge, on="DateTime", direction="backward")

    trades["AtAsk"] = trades["Price"] >= (trades["Quote_Ask"] - PRICE_TOL)
    trades["AtBid"] = trades["Price"] <= (trades["Quote_Bid"] + PRICE_TOL)
    trades["ExecSide"] = np.where(trades["AtAsk"], "buy", np.where(trades["AtBid"], "sell", "unknown"))

    n_buy = int((trades["ExecSide"] == "buy").sum())
    n_sell = int((trades["ExecSide"] == "sell").sum())
    n_unk = int((trades["ExecSide"] == "unknown").sum())

    print(f"  At Ask (buy): {n_buy:,}")
    print(f"  At Bid (sell): {n_sell:,}")
    print(f"  Unknown (excluded): {n_unk:,}")

    trades_strict = trades[trades["ExecSide"] != "unknown"].copy()
    print(f"  Trades for estimation: {len(trades_strict):,}")

    # ------------------------------------------------------------
    # 4) Estimate η by side (pooled across regimes)
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("4. Estimating SIDE-only η (pooled across regimes)")
    print("-" * 60)

    pooled_eta = {}
    pooled_details = {}

    for side in ["buy", "sell"]:
        res = estimate_eta_pooled_by_side(quotes, trades_strict, side)
        pooled_eta[side] = res["eta"]
        pooled_details[side] = res

        print(f"\n  {side}:")
        print(f"    δ support ({DELTA_Q_LOW:.0%}-{DELTA_Q_HIGH:.0%}, quotes-based): "
              f"[{res['delta_support_low']*100:.3f}, {res['delta_support_high']*100:.3f}] cents")
        print(f"    bins_used={res['bins_used']}, valid_bins={res['n_valid_bins']}")
        if np.isfinite(res["eta"]):
            print(f"    η^{side} = {res['eta']:.2f}  | pooled A = {res['A_pooled']:.4f}  | R²(log)={res['r2_log']:.4f}")
        else:
            print("    η estimation FAILED (insufficient slope identification).")

    if not (np.isfinite(pooled_eta["buy"]) and np.isfinite(pooled_eta["sell"])):
        print("\nERROR: Could not estimate η for both sides. Exiting.")
        return

    # ------------------------------------------------------------
    # 5) Estimate regime-specific A given η^side
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("5. Estimating Regime-specific A (given η^side)")
    print("-" * 60)

    A_results = []
    for regime in [0, 1]:
        for side in ["buy", "sell"]:
            eta = pooled_eta[side]
            ares = estimate_A_regime_side_given_eta(quotes, trades_strict, regime, side, eta)
            A_results.append(ares)

            print(f"\n  {ares['regime_name']} {side}:")
            print(f"    Exposure: {ares['total_exposure_sec']/3600:.2f} hours | Fills: {ares['total_fills']:,}")
            print(f"    Mean δ (fills): {ares['mean_delta']*100:.3f} cents | Overall rate: {ares['overall_rate']:.4f}/s")
            print(f"    A_{ares['regime_name']}^{side} = {ares['A']:.4f} (at δ=0)")

    # ------------------------------------------------------------
    # 6) Regime-wise shape audit (bin-based)
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("6. Regime-wise Shape Check (bin-based, exp plausibility)")
    print("-" * 60)

    final_results = []
    binned_rows = []

    # Turn A_results into lookup
    A_lookup = {(r["regime"], r["side"]): r for r in A_results}

    for regime in [0, 1]:
        for side in ["buy", "sell"]:
            eta = pooled_eta[side]
            ainfo = A_lookup[(regime, side)]

            audit = regime_shape_audit(quotes, trades_strict, regime, side, eta)

            # Merge into final structure for plots/saving
            final = {
                "regime": regime,
                "regime_name": "Low" if regime == 0 else "High",
                "side": side,
                "A": ainfo["A"],
                "eta": eta,
                "overall_rate": ainfo["overall_rate"],
                "mean_delta": ainfo["mean_delta"],
                "delta_support_low": audit["delta_support_low"],
                "delta_support_high": audit["delta_support_high"],
                "bins_used": audit["bins_used"],
                "shape_nbins": audit["shape_nbins"],
                "shape_corr": audit["shape_corr"],
                "shape_monotone_viol": audit["shape_monotone_viol"],
                "bin_centers": audit["bin_centers"],
                "bin_exposure": audit["bin_exposure"],
                "bin_fills": audit["bin_fills"],
                "bin_rates": audit["bin_rates"],
            }
            final_results.append(final)

            print(f"\n  {final['regime_name']} {side}:")
            print(f"    δ support ({DELTA_Q_LOW:.0%}-{DELTA_Q_HIGH:.0%}, quotes-based): "
                  f"[{final['delta_support_low']*100:.3f}, {final['delta_support_high']*100:.3f}] cents")
            print(f"    Shape check: corr(log rate, δ)={final['shape_corr']}, monotone violations={final['shape_monotone_viol']} "
                  f"(nbins used={final['bins_used']}, nbins valid={final['shape_nbins']})")

            # bin audit table
            for c, exp, fl, rt in zip(final["bin_centers"], final["bin_exposure"], final["bin_fills"], final["bin_rates"]):
                is_valid = (exp >= MIN_EXPOSURE_SEC_SHAPE) and (fl >= MIN_FILLS_SHAPE) and (rt > 0)
                binned_rows.append({
                    "Regime": final["regime_name"],
                    "Side": side,
                    "delta": float(c),
                    "delta_cents": float(c * 100),
                    "exposure_sec": float(exp),
                    "fills": int(fl),
                    "rate_per_sec": float(rt),
                    "is_valid": bool(is_valid),
                })

    # ------------------------------------------------------------
    # 7) Save tables
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("7. Saving Parameters")
    print("-" * 60)

    # Main parameter table
    rows = []
    for r in final_results:
        rows.append({
            "Regime": r["regime"],
            "Regime_Name": r["regime_name"],
            "Side": r["side"],
            "A": r["A"],
            "eta": r["eta"],
            "overall_rate_per_sec": r["overall_rate"],
            "mean_delta_fills": r["mean_delta"],
            "delta_support_low": r["delta_support_low"],
            "delta_support_high": r["delta_support_high"],
            "shape_corr": r["shape_corr"],
            "shape_monotone_viol": r["shape_monotone_viol"],
            "bins_used_shape": r["bins_used"],
            "shape_nbins_valid": r["shape_nbins"],
            # add pooled η fit quality
            "eta_pooled_r2_log": pooled_details[r["side"]]["r2_log"],
            "eta_pooled_bins_used": pooled_details[r["side"]]["bins_used"],
            "eta_pooled_valid_bins": pooled_details[r["side"]]["n_valid_bins"],
        })

    params_df = pd.DataFrame(rows)
    out_params = PARAMS_DIR / "intensity_parameters_side_eta.csv"
    params_df.to_csv(out_params, index=False)
    print(f"  Saved: {out_params}")

    # Minimal HJB input
    hjb_df = params_df[["Regime", "Regime_Name", "Side", "A", "eta"]].copy()
    out_hjb = PARAMS_DIR / "intensity_for_hjb_side_eta.csv"
    hjb_df.to_csv(out_hjb, index=False)
    print(f"  Saved: {out_hjb}")

    # Bin audit
    binned_df = pd.DataFrame(binned_rows)
    out_bins = PARAMS_DIR / "intensity_binned_rates_side_eta.csv"
    binned_df.to_csv(out_bins, index=False)
    print(f"  Saved: {out_bins}")

    # ------------------------------------------------------------
    # 8) Plots
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("8. Creating Plots")
    print("-" * 60)

    out_plot = PLOTS_DIR / "intensity_side_eta_fit.png"
    out_plot_log = PLOTS_DIR / "intensity_side_eta_fit_log.png"

    plot_panels(final_results, pooled_eta, out_plot, log_scale=False)
    plot_panels(final_results, pooled_eta, out_plot_log, log_scale=True)

    print(f"  Saved: {out_plot}")
    print(f"  Saved: {out_plot_log}")

    appendix_path = PLOTS_DIR / "appendix_hexbin_bins_overlay.png"
    plot_appendix_hexbin_with_bins(
        quotes,
        trades_strict,
        final_results,
        save_path=appendix_path
    )


    

    # ------------------------------------------------------------
    # 9) Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL INTENSITY PARAMETERS (SIDE-only η + Regime-specific A)")
    print("=" * 70)

    print("\n  Pooled η by side:")
    for side in ["buy", "sell"]:
        print(f"    η^{side}: {pooled_eta[side]:.2f}  (R²_log={pooled_details[side]['r2_log']:.3f}, valid_bins={pooled_details[side]['n_valid_bins']})")

    view = params_df[["Regime_Name", "Side", "A", "eta", "shape_corr", "shape_monotone_viol", "overall_rate_per_sec"]].copy()
    view["A"] = view["A"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "NA")
    view["eta"] = view["eta"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "NA")
    view["shape_corr"] = view["shape_corr"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "NA")
    view["overall_rate_per_sec"] = view["overall_rate_per_sec"].map(lambda v: f"{v:.4f}")
    print("\n" + view.to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ Step 3 (V5) Complete! Ready for Step 4 (CI HJB Solver)")
    print("=" * 70)


if __name__ == "__main__":
    main()

# hh.py
# Hodgkin-Huxley parameter estimation (g_L & E_L) — command-line/batch runner.
# - Parametric-KL via MLE (scipy .fit)
# - Monte-Carlo objective evaluation (RMSE + parametric KL)
# - Saves results and plots to ./hh_outputs
#
# Run: python hh.py
#
# All the model math lives in hh_core.py — this file just wires up
# file loading, plotting, and console output around it.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import hh_core as core

warnings.filterwarnings("ignore")

# ----------------------
# USER CONFIG (edit these)
# ----------------------
FILE_PATH = "Voltage Data.xlsx"   # path to experimental data (Excel/CSV) — same folder as this script
COLUMN_NUMBER = 3
DT = 0.01                # time-step for data & estimation (s)
WINDOW_SIZE = 0.1        # window size for local estimation (s)
STEP_SIZE = 0.01         # step between windows (s)
TOTAL_TIME = 5000.0      # total time for synthetic self-test dataset (s)
USE_SYNTHETIC_TEST = False   # If True, runs a synthetic self-test instead of reading a file
POISSON_RATE = 5.0       # spikes per second for the synthetic injected current

MC_N_SAMPLES = 500        # Monte-Carlo sample count (increase for final runs)
MC_SEED = 2025

OUTPUT_DIR = "./hh_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------


def print_kl_note():
    print("\nKL-Divergence note:")
    print("KL(P || Q) = \u222b p(x) log( p(x) / q(x) ) dx")
    print("We fit parametric PDFs to exp & sim via MLE; numerically integrate to compute KL.")
    print("Small epsilon added to pdf values to avoid log(0); integration bounds set from fitted P.\n")


def run_analysis(file_path, column_number, dt, window_size, step_size, use_synthetic=False):
    print("=" * 70)
    print("HODGKIN-HUXLEY PARAMETER ESTIMATION (g_L, E_L) - WITH PARAMETRIC KL & MONTE-CARLO")
    print("=" * 70)
    print(f"Time-step: {dt} s | Window: {window_size} s | Step: {step_size} s\n")

    if use_synthetic:
        print("Generating synthetic self-test dataset (time-varying g_L and E_L)...")
        t = np.arange(0, TOTAL_TIME, dt)
        n_pts = len(t)
        gL_series, EL_series = core.monte_carlo_time_varying_params(t, seed=42)
        I_ext = core.generate_poisson_spikes(n_pts, dt, rate=POISSON_RATE)
        X = np.zeros((n_pts, 4))
        X[0, :] = [-65.0, 0.0529, 0.5961, 0.3177]
        for i in range(n_pts - 1):
            t_span = [t[i], t[i + 1]]
            sol = core.odeint(core.dALLdt, X[i, :], t_span,
                               args=(lambda tt: float(np.interp(tt, t, I_ext)), gL_series[i], EL_series[i]))
            X[i + 1, :] = sol[-1, :]
        V_exp = X[:, 0]
        t_exp_sec = t
        print("Synthetic dataset generated.")
    else:
        try:
            t_exp_sec, V_exp = core.load_voltage_data(file_path, column_number, dt)
            I_ext = core.generate_poisson_spikes(len(t_exp_sec), dt, rate=POISSON_RATE)
            print(f"Loaded file {file_path} ({len(V_exp)} points).")
        except Exception as e:
            print("Error loading file:", e)
            return

    I_inj_func = interp1d(t_exp_sec, I_ext, fill_value="extrapolate")

    # 1) Dynamic parameter estimation (sliding-window)
    print("\n[1/3] Running dynamic (windowed) parameter estimation for g_L and E_L...")
    est_t, est_gL, est_EL = core.estimate_gL_EL_windowed(t_exp_sec, V_exp, I_inj_func, window_size, step_size)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if est_t:
        ax1.plot(est_t, est_gL, '.-', label='Estimated g_L')
        ax2.plot(est_t, est_EL, '.-', label='Estimated E_L')
        print(f"Found {len(est_t)} estimates.")
    else:
        print("No valid estimates found in the window setup - try increasing window size.")
    ax1.set_ylabel('g_L (mS/cm^2)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('E_L (mV)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    fn_est = os.path.join(OUTPUT_DIR, 'estimates_gL_EL.png')
    plt.savefig(fn_est, dpi=300, bbox_inches='tight')
    plt.show()
    if est_t:
        df_est = pd.DataFrame({'time': est_t, 'gL': est_gL, 'EL': est_EL})
        df_est.to_csv(os.path.join(OUTPUT_DIR, 'dynamic_estimates_gL_EL.csv'), index=False)
        print("Saved dynamic estimates CSV.")

    # 2) Distribution analysis
    print("\n[2/3] Analyzing voltage distribution (experiment) and performing parametric MLE fits...")
    V_no_outliers = V_exp[np.abs(V_exp - np.mean(V_exp)) < 3 * np.std(V_exp)]
    fn_vd = None
    if V_no_outliers.size > 0:
        x_range = np.linspace(np.min(V_no_outliers), np.max(V_no_outliers), 500)
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(V_no_outliers, bins=100, density=True, histtype='stepfilled', alpha=0.7)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(V_no_outliers)
            axes[1].plot(x_range, kde(x_range))
        except Exception:
            axes[1].text(0.5, 0.5, "KDE unavailable", ha='center')
        mu, std = core.norm.fit(V_no_outliers)
        axes[2].plot(x_range, core.norm.pdf(x_range, mu, std), '--')
        axes[0].set_title('Histogram (visual)')
        axes[1].set_title('KDE (visual)')
        axes[2].set_title('Gaussian fit (visual)')
        plt.tight_layout()
        fn_vd = os.path.join(OUTPUT_DIR, 'voltage_dists.png')
        plt.savefig(fn_vd, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No valid voltage data for distribution analysis.")

    print_kl_note()

    # 3) Monte-Carlo objective (parametric-KL + RMSE)
    print("\n[3/3] Monte-Carlo sampling of objective (RMSE & parametric-KL)...")
    t_analysis_end = min(1.0, t_exp_sec[-1]) if len(t_exp_sec) > 1 else t_exp_sec[-1]
    idx_end = np.searchsorted(t_exp_sec, t_analysis_end)
    if idx_end <= 1:
        idx_end = len(t_exp_sec)
    t_short = t_exp_sec[:idx_end]
    V_short = V_exp[:idx_end]
    I_inj_short = interp1d(t_short, I_ext[:idx_end], fill_value="extrapolate")
    X0 = [-65.0, 0.05, 0.6, 0.32]

    df_mc = core.monte_carlo_objective_sampling(
        t_short, V_short, I_inj_short, X0,
        gL_bounds=core.MC_gL_bounds, EL_bounds=core.MC_EL_bounds,
        n_samples=MC_N_SAMPLES, rng_seed=MC_SEED, fit_parametric_for_KL=True
    )
    fn_mc = os.path.join(OUTPUT_DIR, 'monte_carlo_objective_results.csv')
    df_mc.to_csv(fn_mc, index=False)
    print(f"Monte-Carlo results saved to {fn_mc} (n={len(df_mc)})")

    best_rmse_row = df_mc.loc[df_mc['rmse'].idxmin()]
    best_kl_row = df_mc.loc[df_mc['kl'].idxmin()]
    print(f"Best by RMSE: g_L={best_rmse_row['gL']:.4f}, E_L={best_rmse_row['EL']:.2f} (RMSE={best_rmse_row['rmse']:.4f})")
    print(f"Best by KL:   g_L={best_kl_row['gL']:.4f}, E_L={best_kl_row['EL']:.2f} (KL={best_kl_row['kl']:.4f})")

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(df_mc['EL'], df_mc['gL'], c=df_mc['rmse'], cmap='viridis', s=28, edgecolor='k', linewidth=0.2)
    plt.colorbar(sc, label='RMSE')
    plt.xlabel('E_L (mV)')
    plt.ylabel('g_L (mS/cm^2)')
    plt.title('Monte-Carlo RMSE samples')
    fn_mc_rmse = os.path.join(OUTPUT_DIR, 'montecarlo_rmse_scatter.png')
    plt.savefig(fn_mc_rmse, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 5))
    sc2 = plt.scatter(df_mc['EL'], df_mc['gL'], c=df_mc['kl'], cmap='magma', s=28, edgecolor='k', linewidth=0.2)
    plt.colorbar(sc2, label='KL (parametric)')
    plt.xlabel('E_L (mV)')
    plt.ylabel('g_L (mS/cm^2)')
    plt.title('Monte-Carlo KL samples')
    fn_mc_kl = os.path.join(OUTPUT_DIR, 'montecarlo_kl_scatter.png')
    plt.savefig(fn_mc_kl, dpi=300, bbox_inches='tight')
    plt.show()

    print("\nAnalysis complete. Saved outputs in:", OUTPUT_DIR)
    print(" -", fn_est)
    print(" -", fn_vd if fn_vd else "(no voltage dist plot)")
    print(" -", fn_mc)
    print(" -", fn_mc_rmse)
    print(" -", fn_mc_kl)
    print("=" * 70)


if __name__ == "__main__":
    run_analysis(FILE_PATH, COLUMN_NUMBER, DT, WINDOW_SIZE, STEP_SIZE, use_synthetic=USE_SYNTHETIC_TEST)
import streamlit as st  
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import hh_core as core

# ===================================================================
# === 1. Theme Setup (Creme / Vintage Paper Styling)
# ===================================================================

def apply_creme_theme():
    creme_css = """
    <style>
        .stApp {
            background-color: #F9F7F1;
            color: #433E3F;
        }
        [data-testid="stSidebar"] {
            background-color: #F0Ece2;
            border-right: 1px solid #D3CFC0;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            color: #433E3F;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2C2420 !important;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        p {
            color: #433E3F;
        }
        hr {
            border-color: #B0A89E;
        }
        [data-testid="stFileUploader"] button {
            color: #7D8E79 !important;
            border-color: #7D8E79 !important;
            background-color: white !important;
        }
        [data-testid="stFileUploader"] button:hover {
            color: #002500 !important;
            border-color: #002500 !important;
            background-color: #F0F5F0 !important;
        }
        [data-testid="stFileUploader"] div[role="listitem"] div {
            color: #7D8E79 !important;
            font-weight: bold;
        }
        [data-testid="stFileUploader"] div[role="listitem"] svg {
            fill: #7D8E79 !important;
        }
        button[kind="primary"] {
            background-color: #002500 !important;
            border-color: #002500 !important;
            color: white !important;
        }
        button[kind="primary"]:hover {
            background-color: #003500 !important;
            border-color: #003500 !important;
        }
        button[kind="primary"]:focus {
            background-color: #002500 !important;
            border-color: #002500 !important;
            box-shadow: none !important;
        }
    </style>
    """
    st.markdown(creme_css, unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#F9F7F1',
        'axes.facecolor': '#F9F7F1',
        'savefig.facecolor': '#F9F7F1',
        'text.color': '#433E3F',
        'axes.labelcolor': '#433E3F',
        'xtick.color': '#433E3F',
        'ytick.color': '#433E3F',
        'grid.color': '#D3CFC0',
        'grid.linestyle': '--',
        'axes.edgecolor': '#433E3F',
        'lines.linewidth': 1.5
    })


# ===================================================================
# === 2. Cached wrappers around the hh_core engine
# ===================================================================

@st.cache_data
def cached_estimate_gL_EL(t_sec, V, I_ext, window_size_sec, step_size_sec):
    I_func = interp1d(t_sec, I_ext, fill_value="extrapolate")
    return core.estimate_gL_EL_windowed(t_sec, V, I_func, window_size_sec, step_size_sec)


@st.cache_data
def cached_monte_carlo(t_short, V_short, I_ext_short, n_samples, seed):
    I_inj_short = interp1d(t_short, I_ext_short, fill_value="extrapolate")
    X0 = [-65.0, 0.05, 0.6, 0.32]
    return core.monte_carlo_objective_sampling(
        t_short, V_short, I_inj_short, X0,
        gL_bounds=core.MC_gL_bounds, EL_bounds=core.MC_EL_bounds,
        n_samples=n_samples, rng_seed=seed, fit_parametric_for_KL=True
    )


# ===================================================================
# === 3. Streamlit GUI Application
# ===================================================================

st.set_page_config(layout="wide", page_title="HH Parameter Estimation")
apply_creme_theme()

st.title("\U0001F52C Hodgkin-Huxley Parameter Estimation Framework")
st.caption("Estimates leak conductance $g_L$ and leak reversal potential $E_L$, with $g_K$ and $E_K$ fixed at standard squid-axon values.")

st.header("Analysis Controls")
uploaded_file = st.file_uploader("Upload Voltage Data (Excel/CSV)", type=["xlsx", "csv"])

col_params1, col_params2 = st.columns(2)

with col_params1:
    st.subheader("Data & Model Parameters")
    dt = st.number_input("Time Step (dt) (s)", value=0.01, format="%.3f", step=0.001)
    column_number = st.number_input("Column Number", value=3, min_value=1, max_value=10, step=1)
    poisson_rate = st.number_input("Synthetic spike rate (Hz)", value=float(core.POISSON_RATE), min_value=0.0, step=0.5,
                                    help="Used to generate a synthetic injected-current trace, since the data file has no recorded current column.")

with col_params2:
    st.subheader("Dynamic Regression Parameters")
    win_size = st.number_input("Window Size (s)", value=0.1, format="%.2f", step=0.01)
    step_size = st.number_input("Step Size (s)", value=0.01, format="%.2f", step=0.01)
    mc_samples = st.number_input("Monte-Carlo samples", value=core.MC_N_SAMPLES, min_value=50, max_value=5000, step=50)

st.divider()
run_button = st.button("Run Analysis", type="primary")

# --- Main Page (Results) ---
if not uploaded_file:
    st.info("Please upload a voltage data file (Excel or CSV) above to begin.")
else:
    try:
        t_exp_sec, V_exp = core.load_voltage_data(uploaded_file, column_number, dt)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    if run_button:
        n_points = len(V_exp)
        I_ext = core.generate_poisson_spikes(n_points, dt, rate=poisson_rate)

        # --- Section 1: Introduction ---
        st.header("Introduction")
        st.write("""
        The Hodgkin-Huxley (HH) model is a Nobel Prize-winning mathematical model that describes how action potentials in neurons are initiated and propagated. It models the neuron's membrane as a circuit with a capacitor (the membrane) and variable resistors (the ion channels).

        This application uses an experimental voltage trace to perform a "dynamic regression" to estimate the time-varying leak-channel parameters ($g_L$ and $E_L$), while $g_K$ and $E_K$ are held at their standard values — testing the hypothesis that the leak parameters are not constant.
        """)
        st.divider()

        # --- Section 2: Dynamic Parameter Estimation ---
        st.header("Dynamic Parameter Estimation Results")
        with st.spinner("Running dynamic parameter estimation..."):
            est_t, est_gL, est_EL = cached_estimate_gL_EL(t_exp_sec, V_exp, I_ext, win_size, step_size)

            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            if est_t:
                ax1.plot(est_t, est_gL, '.-', color='#8B4513', label=f'Estimated $g_L$ ({win_size*1000:.0f}ms window)')
                ax2.plot(est_t, est_EL, '.-', color='#008080', label=f'Estimated $E_L$ ({step_size*1000:.0f}ms step)')
                st.caption(f"Found {len(est_t)} estimates.")
            else:
                st.warning("No valid parameter estimates found for the dynamic regression. Try a larger window size.")

            ax1.set_ylabel('$g_L$ (mS/cm\u00b2)')
            ax1.legend(frameon=False)
            ax1.set_title('Dynamic Parameter Estimation Results')

            ax2.set_xlabel('Time (sec)')
            ax2.set_ylabel('$E_L$ (mV)')
            ax2.legend(frameon=False)

            plt.tight_layout()
            st.pyplot(fig1)
        st.divider()

        # --- Section 3: Voltage Distribution Plots ---
        st.header("Visualization of Experimental Voltage Distribution")

        V_exp_no_outliers = V_exp[np.abs(V_exp - np.mean(V_exp)) < 3 * np.std(V_exp)]

        if V_exp_no_outliers.size == 0:
            st.warning("Could not generate distribution plots. Input data might be empty.")
        else:
            x_range = np.linspace(np.min(V_exp_no_outliers), np.max(V_exp_no_outliers), 500)

            col1, col2, col3 = st.columns(3)

            hist_color = '#BDB76B'
            kde_color = '#2F4F4F'
            norm_color = '#A52A2A'

            with col1:
                fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
                ax_hist.hist(V_exp_no_outliers, bins=100, density=True, histtype='stepfilled',
                             alpha=0.8, color=hist_color, label='Discrete (Histogram)')
                ax_hist.set_title('1. Discrete Distribution')
                ax_hist.set_xlabel('Voltage (mV)')
                ax_hist.set_ylabel('Prob. Density')
                ax_hist.legend(frameon=False, fontsize='small')
                plt.tight_layout()
                st.pyplot(fig_hist)

            with col2:
                fig_kde, ax_kde = plt.subplots(figsize=(5, 3))
                kde = gaussian_kde(V_exp_no_outliers)
                p_kde = kde(x_range)
                ax_kde.plot(x_range, p_kde, color=kde_color, linewidth=2, label='Continuous (KDE)')
                ax_kde.set_title('2. Continuous Distribution')
                ax_kde.set_xlabel('Voltage (mV)')
                ax_kde.set_ylabel('Prob. Density')
                ax_kde.legend(frameon=False, fontsize='small')
                plt.tight_layout()
                st.pyplot(fig_kde)

            with col3:
                fig_norm, ax_norm = plt.subplots(figsize=(5, 3))
                mu, std = core.norm.fit(V_exp_no_outliers)
                p_norm = core.norm.pdf(x_range, mu, std)
                ax_norm.plot(x_range, p_norm, '--', color=norm_color, linewidth=2, label='Gaussian Fit')
                ax_norm.set_title('3. Fitted Normal Distribution')
                ax_norm.set_xlabel('Voltage (mV)')
                ax_norm.set_ylabel('Prob. Density')
                ax_norm.legend(frameon=False, fontsize='small')
                plt.tight_layout()
                st.pyplot(fig_norm)
        st.divider()

        # --- Section 4: Theory ---
        st.header("Theory: Parametric KL Divergence")
        st.write("""
        The Kullback-Leibler (KL) divergence between two continuous probability distributions, $P(x)$ (experimental) and $Q(x)$ (simulated), is defined as:
        """)
        st.latex(r"D_{KL}(P || Q) = \int_{-\infty}^{\infty} P(x) \log\left(\frac{P(x)}{Q(x)}\right) dx")
        st.write("""
        Rather than using raw histograms, this app fits the best-matching parametric distribution (normal, gamma, lognormal, exponential, or laplace — whichever gives the highest likelihood) to both the experimental and simulated voltage traces via MLE, then numerically integrates the KL divergence between the two fitted curves. A small epsilon is added to avoid log(0), and integration bounds are set from the fitted experimental distribution's mean and standard deviation.
        """)
        st.divider()

        # --- Section 5: Monte-Carlo Objective Sampling ---
        st.header("Monte-Carlo Objective Sampling")
        with st.spinner(f"Sampling {mc_samples} candidate (g_L, E_L) pairs... this may take a minute."):
            t_analysis_end = min(1.0, t_exp_sec[-1]) if len(t_exp_sec) > 1 else t_exp_sec[-1]
            idx_end = np.searchsorted(t_exp_sec, t_analysis_end)
            if idx_end <= 1:
                idx_end = len(t_exp_sec)

            t_short = t_exp_sec[:idx_end]
            V_short = V_exp[:idx_end]
            I_ext_short = I_ext[:idx_end]

            df_mc = cached_monte_carlo(t_short, V_short, I_ext_short, int(mc_samples), core.MC_SEED)

            best_rmse_row = df_mc.loc[df_mc['rmse'].idxmin()]
            best_kl_row = df_mc.loc[df_mc['kl'].idxmin()]

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Best g_L (by RMSE)", f"{best_rmse_row['gL']:.4f} mS/cm\u00b2")
                st.metric("Best E_L (by RMSE)", f"{best_rmse_row['EL']:.2f} mV")
            with m2:
                st.metric("Best g_L (by KL)", f"{best_kl_row['gL']:.4f} mS/cm\u00b2")
                st.metric("Best E_L (by KL)", f"{best_kl_row['EL']:.2f} mV")

            fig_surf, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            sc1 = ax1.scatter(df_mc['EL'], df_mc['gL'], c=df_mc['rmse'], cmap='cividis', s=24, edgecolor='k', linewidth=0.2)
            fig_surf.colorbar(sc1, ax=ax1, label='RMSE')
            ax1.set_xlabel('$E_L$ (mV)')
            ax1.set_ylabel('$g_L$ (mS/cm\u00b2)')
            ax1.set_title('Monte-Carlo RMSE samples')

            sc2 = ax2.scatter(df_mc['EL'], df_mc['gL'], c=np.log(df_mc['kl'] + 1e-9), cmap='cividis', s=24, edgecolor='k', linewidth=0.2)
            fig_surf.colorbar(sc2, ax=ax2, label='log(KL Divergence)')
            ax2.set_xlabel('$E_L$ (mV)')
            ax2.set_ylabel('$g_L$ (mS/cm\u00b2)')
            ax2.set_title('Monte-Carlo KL samples')

            plt.tight_layout()
            st.pyplot(fig_surf)

            st.download_button(
                "Download Monte-Carlo results (CSV)",
                df_mc.to_csv(index=False).encode("utf-8"),
                file_name="monte_carlo_objective_results.csv",
                mime="text/csv",
            )

        st.success("Analysis Complete!")
# hh_core.py
# Hodgkin-Huxley parameter-estimation engine — shared backend.
#
# Method: g_K and E_K are FIXED at their standard squid-axon values.
# g_L (leak conductance) and E_L (leak reversal potential) are the unknowns,
# recovered from a voltage trace via:
#   1) Sliding-window linear regression (isolates the leak current after
#      subtracting the well-characterized Na+/K+ currents)
#   2) Monte-Carlo sampling of the (g_L, E_L) space, scored by RMSE and by
#      a parametric-KL divergence between fitted experimental/simulated
#      voltage distributions
#
# This module has NO plotting, printing, or file I/O side effects (besides
# reading the voltage data file itself)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint
from scipy.stats import norm, gamma, lognorm, expon, laplace

# ----------------------
# Fixed model constants (standard Hodgkin-Huxley squid-axon values)
# ----------------------
Cm = 1.0
g_Na = 120.0
g_K = 36.0
E_Na = 55.0
E_K = -72.0

# Defaults / starting points for the leak parameters being estimated
g_L_default = 0.3
E_L_default = -49.0

# Candidate parametric distributions for MLE fitting (used in parametric-KL)
CANDIDATE_DISTS = {
    'normal': norm,
    'gamma': gamma,
    'lognormal': lognorm,
    'exponential': expon,
    'laplace': laplace,
}

# Poisson synthetic-current defaults (used when the data file has no
# recorded injected-current column)
POISSON_RATE = 5.0       # spikes per second
SPIKE_MAG = 2.0           # uA/cm^2 spike magnitude
SPIKE_WIDTH = 0.001       # spike width in seconds (1 ms)

# Monte-Carlo defaults
MC_N_SAMPLES = 500
MC_SEED = 2025
MC_gL_bounds = (0.01, 1.5)
MC_EL_bounds = (E_L_default - 20, E_L_default + 20)


# ----------------------
# Gating kinetics (standard Hodgkin-Huxley forms)
# ----------------------
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10) + 1e-12)
def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80)
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10) + 1e-12)
def beta_m(V):  return 4 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V):  return 1 / (1 + np.exp(-(V + 35) / 10))


# ----------------------
# Ionic currents
# ----------------------
def I_Na(V, m, h, gNa=g_Na, ENa=E_Na):
    return gNa * (m ** 3) * h * (V - ENa)

def I_K(V, n, gK=g_K, EK=E_K):
    return gK * (n ** 4) * (V - EK)

def I_L(V, gL, EL):
    return gL * (V - EL)


# ----------------------
# Full ODE system (V, m, h, n) for simulating with a candidate (gL, EL)
# ----------------------
def dALLdt(X, t, I_inj_func, gL, EL):
    V, m, h, n = X
    I_inj = I_inj_func(t)
    dVdt = (I_inj - I_Na(V, m, h) - I_K(V, n) - I_L(V, gL, EL)) / Cm
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]


# ----------------------
# Evolve gating variables given a recorded/simulated V trace
# ----------------------
def solve_gating_vars(V_trace, t_trace):
    def dGatingsdt(X, t):
        m, h, n = X
        V = np.interp(t, t_trace, V_trace)
        dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
        dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
        dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
        return [dmdt, dhdt, dndt]
    X0 = [0.05, 0.6, 0.32]
    X = odeint(dGatingsdt, X0, t_trace)
    return X[:, 0], X[:, 1], X[:, 2]


# ----------------------
# Synthetic injected-current generator (Poisson spike train)
# Used when the data file has no recorded injected-current column.
# ----------------------
def generate_poisson_spikes(n_points, dt, rate=POISSON_RATE, spike_mag=SPIKE_MAG, spike_width=SPIKE_WIDTH):
    p = rate * dt
    events = np.random.rand(n_points) < p
    I_ext = np.zeros(n_points)
    spike_samples = max(1, int(np.round(spike_width / dt)))
    for i, ev in enumerate(events):
        if ev:
            end = min(n_points, i + spike_samples)
            I_ext[i:end] += spike_mag
    return I_ext


# ----------------------
# Synthetic time-varying (gL, EL) generator — for self-test / demo mode
# ----------------------
def monte_carlo_time_varying_params(t, base_gL=g_L_default, base_EL=E_L_default, seed=None):
    rng = np.random.RandomState(seed)
    n = len(t)
    gL = np.ones(n) * base_gL
    EL = np.ones(n) * base_EL
    step_std_g = 0.03 * base_gL
    step_std_EL = 0.5
    for i in range(1, n):
        gL[i] = gL[i - 1] + rng.normal(scale=step_std_g)
        EL[i] = EL[i - 1] + rng.normal(scale=step_std_EL)
    gL += 0.05 * base_gL * np.sin(2 * np.pi * 0.001 * t)
    EL += 1.0 * np.sin(2 * np.pi * 0.002 * t)
    gL = np.clip(gL, 0.01, 5.0)
    EL = np.clip(EL, -100.0, 20.0)
    return gL, EL


# ----------------------
# Data loading — shared by CLI (file path) and Streamlit (uploaded file)
# ----------------------
def load_voltage_data(file, column_number, dt):
    """
    file: a path string OR a file-like object (e.g. Streamlit's UploadedFile).
    column_number: 1-indexed column selector; actual column = 2*column_number - 2
                    (matches the paired time/voltage layout of the source data).
    Returns (t_exp_sec, V_exp).
    """
    name = getattr(file, "name", None)
    if name is None:
        name = file if isinstance(file, str) else ""
    if str(name).lower().endswith(".csv"):
        data = pd.read_csv(file, header=None)
    else:
        data = pd.read_excel(file, header=None, skiprows=3)

    col_idx = 2 * int(column_number) - 2
    if col_idx >= len(data.columns):
        max_col = (len(data.columns) + 2) // 2
        raise ValueError(f"Column {column_number} does not exist in the data (max column: {max_col}).")

    V_exp = data.iloc[:, col_idx].astype(float).values
    n_points = len(V_exp)
    t_exp_sec = np.arange(0, n_points * dt, dt)
    return t_exp_sec, V_exp


# ----------------------
# Windowed estimation of g_L and E_L via linear regression
# ----------------------
def estimate_gL_EL_windowed(t_sec, V, I_func, window_size_sec, step_size_sec):
    estimated_times = []
    estimated_gL = []
    estimated_EL = []

    if len(t_sec) < 2:
        return [], [], []

    dt = t_sec[1] - t_sec[0]
    if dt <= 0:
        return [], [], []

    dVdt = np.gradient(V, dt)
    m, h, n = solve_gating_vars(V, t_sec)
    I_inj_trace = I_func(t_sec)
    I_Na_trace = I_Na(V, m, h)
    I_K_trace = I_K(V, n)
    Y_target = (Cm * dVdt) - I_inj_trace + I_Na_trace + I_K_trace  # should equal I_L

    X_features = np.vstack([V, np.ones_like(V)]).T

    n_points = len(t_sec)
    window_pts = int(round(window_size_sec / dt))
    step_pts = int(round(step_size_sec / dt))
    if window_pts <= 0:
        window_pts = 1
    if step_pts <= 0:
        step_pts = 1
    if window_pts > n_points:
        window_pts = n_points

    model = LinearRegression(fit_intercept=False)

    for i in range(0, n_points - window_pts + 1, step_pts):
        win_start = i
        win_end = i + window_pts
        t_window_mid = t_sec[win_start + window_pts // 2]
        Y_win = Y_target[win_start:win_end]
        X_win = X_features[win_start:win_end, :]
        if np.allclose(X_win[:, 0], X_win[0, 0]):
            continue
        try:
            model.fit(X_win, Y_win)
            coef_V = model.coef_[0]
            intercept = model.coef_[1]
            if abs(coef_V) < 1e-8:
                continue
            gL_est = coef_V
            EL_est = -intercept / gL_est
            if 0.0 < gL_est < 50.0 and -200.0 < EL_est < 100.0:
                estimated_times.append(t_window_mid)
                estimated_gL.append(gL_est)
                estimated_EL.append(EL_est)
        except Exception:
            pass

    return estimated_times, estimated_gL, estimated_EL


# ----------------------
# Simulation driver for a single (constant) parameter set
# ----------------------
def run_simulation_const_params(params, t_short, I_inj_short, X0):
    gL, EL = params
    try:
        X = odeint(dALLdt, X0, t_short, args=(I_inj_short, gL, EL))
        V_sim = X[:, 0]
        if not np.all(np.isfinite(V_sim)):
            return None
        return V_sim
    except Exception:
        return None


def objective_rmse(V_short, V_sim):
    if V_sim is None:
        return 1e9
    return np.sqrt(np.mean((V_short - V_sim) ** 2))


# ----------------------
# Parametric distribution fitting + parametric KL divergence
# ----------------------
def fit_best_parametric_distribution(data, candidates=CANDIDATE_DISTS):
    best_name = None
    best_dist = None
    best_params = None
    best_ll = -np.inf
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None, None, None, -np.inf
    for name, dist in candidates.items():
        try:
            if name == 'gamma':
                if np.any(data <= 0):
                    shift = np.min(data) - 1e-3
                    params = dist.fit(data - shift)
                    params = (shift,) + params
                else:
                    params = dist.fit(data)
            else:
                params = dist.fit(data)
            try:
                if name == 'gamma' and params is not None and isinstance(params, tuple) and len(params) > 0 and isinstance(params[0], (float, np.floating)):
                    shift = params[0]
                    gamma_params = params[1:]
                    ll = np.sum(dist.logpdf(data - shift, *gamma_params))
                else:
                    ll = np.sum(dist.logpdf(data, *params))
            except Exception:
                ll = -np.inf
            if ll > best_ll:
                best_ll = ll
                best_name = name
                best_dist = dist
                best_params = params
        except Exception:
            continue
    return best_name, best_dist, best_params, best_ll


def parametric_pdf(dist, params, x):
    if dist == gamma and isinstance(params, tuple) and len(params) > 0 and params[0] is not None and (params[0] < -1e-6 or params[0] > -1e-6):
        shift = params[0]
        gamma_params = params[1:]
        return dist.pdf(x - shift, *gamma_params)
    else:
        return dist.pdf(x, *params)


def kl_between_parametric(distP, paramsP, distQ, paramsQ, x_min=None, x_max=None, npoints=2000, eps=1e-12):
    try:
        mu = distP.mean(*paramsP) if hasattr(distP, 'mean') else np.mean
        sd = np.sqrt(distP.var(*paramsP)) if hasattr(distP, 'var') else None
        if sd is None or not np.isfinite(sd):
            x_min, x_max = -120, 80
        else:
            x_min = mu - 6 * sd
            x_max = mu + 6 * sd
    except Exception:
        x_min, x_max = -120, 80
    if x_min is None or x_max is None:
        x_min, x_max = -120, 80
    x = np.linspace(x_min, x_max, npoints)
    p = parametric_pdf(distP, paramsP, x) + eps
    q = parametric_pdf(distQ, paramsQ, x) + eps
    p = p / (np.trapezoid(p, x))
    q = q / (np.trapezoid(q, x))
    integrand = p * np.log(p / q)
    kl = np.trapezoid(integrand, x)
    return float(kl)


# ----------------------
# Monte-Carlo sampling of the objective surface (RMSE + parametric-KL)
# ----------------------
def monte_carlo_objective_sampling(t_short, V_short, I_inj_short, X0,
                                    gL_bounds=MC_gL_bounds, EL_bounds=MC_EL_bounds,
                                    n_samples=MC_N_SAMPLES, rng_seed=MC_SEED, fit_parametric_for_KL=True):
    rng = np.random.RandomState(rng_seed)
    samples = rng.rand(n_samples, 2)
    gL_samples = gL_bounds[0] + samples[:, 0] * (gL_bounds[1] - gL_bounds[0])
    EL_samples = EL_bounds[0] + samples[:, 1] * (EL_bounds[1] - EL_bounds[0])
    results = []

    if fit_parametric_for_KL:
        _, best_dist_exp, best_params_exp, _ = fit_best_parametric_distribution(V_short)
    else:
        best_dist_exp = None
        best_params_exp = None

    for gL, EL in zip(gL_samples, EL_samples):
        V_sim = run_simulation_const_params((gL, EL), t_short, I_inj_short, X0)
        rmse = objective_rmse(V_short, V_sim)
        if best_dist_exp is None or V_sim is None:
            kl_val = 1e9
        else:
            _, best_dist_sim, best_params_sim, _ = fit_best_parametric_distribution(V_sim)
            if best_dist_sim is None:
                kl_val = 1e9
            else:
                kl_val = kl_between_parametric(best_dist_exp, best_params_exp, best_dist_sim, best_params_sim)
        results.append((gL, EL, rmse, kl_val))

    return pd.DataFrame(results, columns=['gL', 'EL', 'rmse', 'kl'])
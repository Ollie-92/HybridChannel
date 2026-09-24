# Source: YuChein/lsp_calculator.py; SHA-256 in docs/provenance.json.
# Modified for this local review on 2026-09-23: packaging and documentation.
# Original code rights: research team (confirmed by project owner); LICENSE pending.
# B10-B11, B19: legacy Dense Urban LOS tables, sampling, and statistics. Only the NTN registry is retained. Parameters are not certified as exact TR 38.811 conformance.

import numpy as np


import warnings


eps = 1e-30


C_PHI_TABLE = {
    4: 0.779,  5: 0.860,  8: 1.018, 10: 1.090, 11: 1.123,
    12: 1.146, 14: 1.190, 15: 1.211, 16: 1.226, 19: 1.273,
    20: 1.289, 25: 1.358,
}


C_THETA_TABLE = {
    8: 0.889,  10: 0.957, 11: 1.031, 12: 1.104,
    15: 1.1088, 19: 1.184, 20: 1.178, 25: 1.282,
}


def _lookup_nearest(table, N):
    """Look up the nearest key in a table (no interpolation)."""
    keys = np.array(sorted(table.keys()))
    idx = int(np.argmin(np.abs(keys - N)))
    return table[keys[idx]]


def lookup_C_phi(N, K_dB=None, is_LOS=False):
    """
    Look up C_phi from Table 7.5-2, with LOS correction.

    Parameters
    ----------
    N : int
        Total number of clusters (RT + RC after merge).
    K_dB : float or None
        K-factor in dB (only used if is_LOS).
    is_LOS : bool

    Returns
    -------
    float
    """
    C_nlos = _lookup_nearest(C_PHI_TABLE, N)
    if is_LOS and K_dB is not None:
        K = K_dB
        # TR 38.901 Table 7.5-2 footnote
        C_los = C_nlos * (1.1035 - 0.028 * K - 0.002 * K**2
                          + 0.0001 * K**3)
        return C_los
    return C_nlos


def lookup_C_theta(N, K_dB=None, is_LOS=False):
    """
    Look up C_theta from Table 7.5-4, with LOS correction.

    Parameters
    ----------
    N : int
        Total number of clusters (RT + RC after merge).
    K_dB : float or None
    is_LOS : bool

    Returns
    -------
    float
    """
    C_nlos = _lookup_nearest(C_THETA_TABLE, N)
    if is_LOS and K_dB is not None:
        K = K_dB
        # TR 38.901 Table 7.5-4 footnote
        C_los = C_nlos * (1.3086 + 0.0339 * K - 0.0077 * K**2
                          + 0.0002 * K**3)
        return C_los
    return C_nlos


GPP_TABLE = {
    "elevation": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    "DS":  {"mean": [-7.43, -7.62, -7.76, -8.02, -8.13, -8.30, -8.34, -8.39, -8.45],
            "std":  [ 0.90,  0.78,  0.80,  0.72,  0.61,  0.47,  0.39,  0.26,  0.01]},
    "ASD": {"mean": [-3.43, -3.06, -2.91, -2.81, -2.74, -2.72, -2.46, -2.30, -1.11],
            "std":  [ 0.54,  0.41,  0.42,  0.34,  0.34,  0.70,  0.40,  0.78,  0.51]},
    "ASA": {"mean": [ 0.65,  0.53,  0.60,  0.43,  0.36,  0.16,  0.18,  0.24,  0.36],
            "std":  [ 0.82,  0.78,  0.83,  0.78,  0.77,  0.84,  0.64,  0.81,  0.65]},
    "ZSA": {"mean": [ 0.82,  0.47,  0.80,  1.23,  1.42,  1.56,  1.65,  1.73,  1.79],
            "std":  [ 0.05,  0.11,  0.05,  0.04,  0.10,  0.06,  0.07,  0.02,  0.01]},
    "ZSD": {"mean": [-2.75, -2.64, -2.49, -2.51, -2.54, -2.71, -2.85, -3.01, -3.08],
            "std":  [ 0.55,  0.64,  0.69,  0.57,  0.50,  0.37,  0.31,  0.45,  0.27]},
    "SF":  {"mean": [  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
            "std":  [  2.9,   2.4,   2.7,   2.4,   2.4,   2.7,   2.6,   2.8,   0.6]},
    "K":   {"mean": [  6.1,  13.7,  12.9,  10.3,   9.2,   8.4,   8.0,   7.4,   7.6],
            "std":  [  2.6,   6.8,   6.0,   3.3,   2.2,   1.9,   1.5,   1.6,   1.3]},
}


def _ntn_dur_los(elev_deg):
    """NTN Dense Urban LOS — elevation-dependent (from GPP_TABLE)."""
    elevs = GPP_TABLE["elevation"]
    idx = int(np.argmin(np.abs(np.array(elevs) - elev_deg)))
    return {
        'scenario': f'NTN-DenseUrban-LOS-{elevs[idx]}deg',
        'mu_lgDS':  GPP_TABLE["DS"]["mean"][idx],
        'sig_lgDS': GPP_TABLE["DS"]["std"][idx],
        'mu_K':     GPP_TABLE["K"]["mean"][idx],
        'sig_K':    GPP_TABLE["K"]["std"][idx],
        'mu_lgASD': GPP_TABLE["ASD"]["mean"][idx],
        'sig_lgASD': GPP_TABLE["ASD"]["std"][idx],
        'mu_lgASA': GPP_TABLE["ASA"]["mean"][idx],
        'sig_lgASA': GPP_TABLE["ASA"]["std"][idx],
        'mu_lgZSD': GPP_TABLE["ZSD"]["mean"][idx],
        'sig_lgZSD': GPP_TABLE["ZSD"]["std"][idx],
        'mu_lgZSA': GPP_TABLE["ZSA"]["mean"][idx],
        'sig_lgZSA': GPP_TABLE["ZSA"]["std"][idx],
        'mu_offset_ZOD': 0.0,
        # NTN cross-correlation: use approximate values
        # (3GPP TR 38.811 Table 6.7.2-4 for Dense Urban LOS)
        'C': np.array([
            [ 1.0, -0.4, -0.2, -0.3,  0.0,  0.0],
            [-0.4,  1.0,  0.5,  0.8, -0.1,  0.2],
            [-0.2,  0.5,  1.0,  0.4,  0.5,  0.0],
            [-0.3,  0.8,  0.4,  1.0,  0.0,  0.1],
            [ 0.0, -0.1,  0.5,  0.0,  1.0,  0.0],
            [ 0.0,  0.2,  0.0,  0.1,  0.0,  1.0],
        ]),
        'r_tau':     2.3,
        'N_cluster': 8,
        'zeta_dB':   3.0,
        'c_DS':      5e-9,
        'mu_XPR':    9.0,   # NTN Dense Urban LOS (approx UMi LOS)
        'sigma_XPR': 3.0,
    }


SCENARIO_TABLE = {'NTN-DenseUrban-LOS': _ntn_dur_los}


def get_scenario_params(scenario_name, fc_GHz=None, elev_deg=None):
    """
    Look up scenario parameters from 3GPP TR 38.901 Table 7.5-6.

    Parameters
    ----------
    scenario_name : str
        Only 'NTN-DenseUrban-LOS' is retained in this extraction.
    fc_GHz : float
        Carrier frequency in GHz (required for UMi/UMa/InH)
    elev_deg : float
        Elevation angle in degrees (required for NTN scenarios)

    Returns
    -------
    dict with all scenario parameters needed for sample_lsps()
    """
    if scenario_name not in SCENARIO_TABLE:
        raise ValueError(f"Unknown scenario '{scenario_name}'. "
                         f"Available: {list(SCENARIO_TABLE.keys())}")

    factory = SCENARIO_TABLE[scenario_name]

    if scenario_name.startswith('NTN'):
        if elev_deg is None:
            raise ValueError("elev_deg required for NTN scenarios")
        return factory(elev_deg)
    else:
        if fc_GHz is None:
            raise ValueError("fc_GHz required for terrestrial scenarios")
        return factory(fc_GHz)


def sample_lsps(scenario_params, rng):
    """
    Sample a correlated 6-dimensional LSP vector using Cholesky
    decomposition of the cross-correlation matrix.

    LSP vector order: [K, DS, ASD, ASA, ZSD, ZSA]

    Parameters
    ----------
    scenario_params : dict
        Output of get_scenario_params(), containing mu/sigma for
        each LSP and the 6x6 cross-correlation matrix 'C'.
    rng : np.random.Generator
        Random number generator instance.

    Returns
    -------
    dict with keys:
        'DS':     float, delay spread in seconds (linear, not log10)
        'K_R_dB': float, K-factor in dB (None if NLOS)
        'ASA':    float, azimuth spread of arrival in degrees
        'ASD':    float, azimuth spread of departure in degrees
        'ZSA':    float, zenith spread of arrival in degrees
        'ZSD':    float, zenith spread of departure in degrees
        'r_tau':  float, delay distribution proportionality factor
        'zeta_dB': float, per-cluster shadowing std (dB)
        'N_cluster': int, number of clusters
        'is_LOS': bool
        'c_DS':   float, intra-cluster delay spread (seconds)
        'mu_offset_ZOD': float
    """
    sp = scenario_params

    # ── Mean and std vectors (order: K, DS, ASD, ASA, ZSD, ZSA) ──
    mu = np.array([
        sp['mu_K'],
        sp['mu_lgDS'],
        sp['mu_lgASD'],
        sp['mu_lgASA'],
        sp['mu_lgZSD'],
        sp['mu_lgZSA'],
    ])
    sigma = np.array([
        sp['sig_K'],
        sp['sig_lgDS'],
        sp['sig_lgASD'],
        sp['sig_lgASA'],
        sp['sig_lgZSD'],
        sp['sig_lgZSA'],
    ])

    # ── Cholesky of cross-correlation matrix ──
    C = np.array(sp['C'], dtype=np.float64)
    # Ensure positive-definite (numerical safety)
    eigvals = np.linalg.eigvalsh(C)
    if np.min(eigvals) < 1e-10:
        C += np.eye(6) * (1e-10 - np.min(eigvals))
    L = np.linalg.cholesky(C)

    # ── Draw correlated standard normal samples ──
    z = rng.standard_normal(6)
    s_corr = L @ z  # correlated standard normals

    # ── Transform to physical LSP values ──
    s = mu + sigma * s_corr

    # s[0] = K (dB), s[1..5] = log10 values
    K_dB = s[0]
    lgDS  = s[1]
    lgASD = s[2]
    lgASA = s[3]
    lgZSD = s[4]
    lgZSA = s[5]

    # ── Convert from log10 to linear ──
    DS_s  = 10.0**lgDS       # seconds
    ASD_s = 10.0**lgASD      # degrees
    ASA_s = 10.0**lgASA      # degrees
    ZSD_s = 10.0**lgZSD      # degrees
    ZSA_s = 10.0**lgZSA      # degrees

    # ── Apply angle upper limits (TR 38.901 §7.5 Step 4 Note) ──
    ASA_s = min(ASA_s, 104.0)
    ASD_s = min(ASD_s, 104.0)
    ZSA_s = min(ZSA_s, 52.0)
    ZSD_s = min(ZSD_s, 52.0)

    # ── Determine LOS ──
    is_LOS = sp.get('sig_K', 0) > 0  # NLOS has sig_K = 0

    return {
        'DS':       float(DS_s),
        'K_R_dB':   float(K_dB) if is_LOS else None,
        'ASA':      float(ASA_s),
        'ASD':      float(ASD_s),
        'ZSA':      float(ZSA_s),
        'ZSD':      float(ZSD_s),
        'r_tau':    sp['r_tau'],
        'zeta_dB':  sp['zeta_dB'],
        'N_cluster': sp['N_cluster'],
        'is_LOS':   is_LOS,
        'c_DS':     sp.get('c_DS', 0.0),
        'mu_offset_ZOD': sp.get('mu_offset_ZOD', 0.0),
        'mu_XPR':   sp.get('mu_XPR', 9.0),
        'sigma_XPR': sp.get('sigma_XPR', 3.0),
    }


def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def circ_rms_spread_deg(theta_rad, w):
    """Return circular RMS angular spread in degrees from radian angles and weights."""
    mu = np.angle(np.sum(w * np.exp(1j*theta_rad)))
    d  = wrap_pi(theta_rad - mu)
    return np.degrees(np.sqrt(np.sum(w*d*d) / (np.sum(w)+eps)))


def lin_rms_spread_deg(x_rad, w):
    """Return linear RMS angular spread in degrees from radian angles and weights."""
    mu = np.sum(w*x_rad) / (np.sum(w)+eps)
    return np.degrees(np.sqrt(np.sum(w*(x_rad-mu)**2) / (np.sum(w)+eps)))


def rms_delay_spread(tau_excess, p):
    """Return power-weighted RMS delay spread in seconds for linear path powers."""
    ps = np.sum(p) + eps
    mu = np.sum(p*tau_excess) / ps
    return np.sqrt(np.sum(p*(tau_excess-mu)**2) / ps)


def has_los(paths_los, rx_idx=0, tx_idx=0):
    """Return whether the scalar VV LOS probe has a valid delay and amplitude > 1e-15."""
    a_los, tau_los = paths_los.cir(normalize_delays=False, out_type="numpy")
    a0   = a_los[rx_idx, 0, tx_idx, 0, :, 0]
    tau0 = tau_los[rx_idx, tx_idx, :]
    v    = (tau0 >= 0) & (np.abs(a0) > 1e-15)
    return np.any(v)

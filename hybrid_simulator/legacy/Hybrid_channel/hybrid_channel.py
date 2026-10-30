# Source history and numerical checks: archive/research-validation branch.

"""
Map-based Hybrid Channel Model: PDP fusion
Based on 3GPP TR 38.901 V19.2.0 Clause 8 (§8.4, Step 3–10)

RT supplies the backbone; random clusters augment its paths.
"""

import numpy as np
import warnings


# ============================================================
# Step A: Compute mu_tau_RC  (TR 38.901 §8.4, Step 5, eq. below 8.4-1)
# ============================================================
def compute_mu_tau_RC(tau_rt_norm, L_RT, scenario_params):
    """
    Parameters
    ----------
    tau_rt_norm : array, shape (L_RT,)
        RT excess delay = tau_rt - min(tau_rt), in seconds
    L_RT : int
        Number of RT paths
    scenario_params : dict
        Requires 'DS', 'r_tau', 'N_cluster'

    Returns
    -------
    mu_tau_RC : float
    L_RC_prime : int
    """
    DS = scenario_params['DS']
    r_tau = scenario_params['r_tau']
    L_RC_prime = scenario_params['N_cluster']

    tau_rt_mean = np.mean(tau_rt_norm)

    # TR 38.901 §8.4, text below eq. 8.4-1
    mu_tau = (r_tau * DS
              + (L_RT / (L_RC_prime + 1))
              * (r_tau * DS - tau_rt_mean))
    mu_tau_RC = max(mu_tau, tau_rt_mean)

    return mu_tau_RC, L_RC_prime


# ============================================================
# Step B: Generate RC delays  (TR 38.901 eq. 8.4-1, 8.4-2, 8.4-3)
# ============================================================
def generate_rc_delays(mu_tau_RC, L_RC_prime, scenario_params, rng):
    """
    Returns
    -------
    tau_RC_scaled : array  — Output delays, multiplied by C_tau
    tau_RC_for_power : array  — Unscaled delays for the power calculation
    """
    K_R_dB = scenario_params.get('K_R_dB', None)
    is_LOS = scenario_params['is_LOS']

    # eq. 8.4-1: tau'_n = -mu_tau_RC * ln(X_n)
    X = rng.uniform(0, 1, size=L_RC_prime)
    tau_prime = -mu_tau_RC * np.log(X)

    # eq. 8.4-2: normalize and sort
    tau_norm = np.sort(tau_prime - np.min(tau_prime))

    # eq. 8.4-3: C_tau
    if is_LOS and K_R_dB is not None:
        K = K_R_dB
        C_tau = (0.7705 - 0.0433 * K + 0.0002 * K**2
                 + 0.000017 * K**3)
    else:
        C_tau = 1.0

    # Power calculation uses unscaled delays (C_tau = 1).
    tau_RC_for_power = tau_norm.copy()      # C_tau = 1
    tau_RC_scaled = tau_norm * C_tau         # Used for final output

    return tau_RC_scaled, tau_RC_for_power


# ============================================================
# Step C: Remove RC clusters  (TR 38.901 §8.4, second part of Step 5)
# ============================================================
def remove_rc_clusters(tau_RC_scaled, tau_RC_for_power,
                       tau_rt_norm, mu_tau_RC, config):
    """
    Removal rules:
    1. n == 0 (earliest candidate)
    2. Any RT path satisfies |tau_RC_n - tau_rt_l| < tau_th

    Returns
    -------
    tau_RC_scaled_kept, tau_RC_for_power_kept
    """
    p0 = config.get('p0', 0.2)

    # tau_th = mu_tau_RC * ln(1 / (1 - p0))
    tau_th = mu_tau_RC * np.log(1.0 / (1.0 - p0))

    keep = []
    for n in range(len(tau_RC_scaled)):
        # Rule 1: remove the earliest candidate (n == 0)
        if n == 0:
            continue
        # Rule 2: candidate is too close to an RT path
        # Compare unscaled candidate delays with RT excess delays.
        if np.any(np.abs(tau_RC_for_power[n] - tau_rt_norm) < tau_th):
            continue
        keep.append(n)

    keep = np.array(keep, dtype=int)
    if len(keep) == 0:
        return np.array([]), np.array([])

    return tau_RC_scaled[keep], tau_RC_for_power[keep]


# ============================================================
# Step D: Compute virtual powers  (TR 38.901 eq. 8.4-4, 8.4-5)
# ============================================================
def compute_virtual_powers(tau_RC_for_power, tau_rt_norm,
                           scenario_params, rng,
                           Z_RC=None, Z_RT=None,
                           return_shadowing=False):
    """
    Returns
    -------
    V_RC : array (L_RC,)
    V_RT : array (L_RT,)
    """
    DS = scenario_params['DS']
    r_tau = scenario_params['r_tau']
    zeta_dB = scenario_params['zeta_dB']

    L_RC = len(tau_RC_for_power)
    L_RT = len(tau_rt_norm)

    # eq. 8.4-4: V_RC_i
    if Z_RC is None:
        Z_RC = rng.normal(0, zeta_dB, size=L_RC)
    else:
        Z_RC = np.asarray(Z_RC, dtype=np.float64)
    V_RC = (np.exp(-tau_RC_for_power * (r_tau - 1) / (r_tau * DS))
            * 10.0**(-Z_RC / 10.0))

    # eq. 8.4-5: V_RT_j
    if Z_RT is None:
        Z_RT = rng.normal(0, zeta_dB, size=L_RT)
    else:
        Z_RT = np.asarray(Z_RT, dtype=np.float64)
    V_RT = (np.exp(-tau_rt_norm * (r_tau - 1) / (r_tau * DS))
            * 10.0**(-Z_RT / 10.0))

    if return_shadowing:
        return V_RC, V_RT, Z_RC, Z_RT
    return V_RC, V_RT


# ============================================================
# Step E: Normalize virtual powers  (TR 38.901 eq. 8.4-6, 8.4-7)
# ============================================================
def normalize_virtual_powers(V_RC, V_RT, scenario_params):
    """
    Returns
    -------
    P_RC_virtual : array (L_RC,)
    P_RT_virtual : array (L_RT,)
    """
    is_LOS = scenario_params['is_LOS']
    K_R_dB = scenario_params.get('K_R_dB', None)

    # eq. 8.4-6
    if is_LOS and K_R_dB is not None:
        A = 10.0**(K_R_dB / 10.0)
    else:
        A = 0.0

    V_sum = np.sum(V_RC) + np.sum(V_RT)

    # eq. 8.4-7
    P_RC_virtual = (1.0 / (A + 1.0)) * V_RC / V_sum
    P_RT_virtual = (1.0 / (A + 1.0)) * V_RT / V_sum

    # Add the LOS component to the first RT path.
    if len(P_RT_virtual) > 0:
        P_RT_virtual[0] += A / (A + 1.0)

    return P_RC_virtual, P_RT_virtual


# ============================================================
# Step F: Anchor RC powers to RT powers  (TR 38.901 eq. 8.4-8) ⭐
# ============================================================
def anchor_rc_power(P_RC_virtual, P_RT_virtual, power_rt):
    """
    scale = sum(P_RT_real) / sum(P_RT_virtual)
    P_RC_real[i] = scale * P_RC_virtual[i]

    Returns
    -------
    P_RC_real : array (L_RC,)
    """
    sum_P_RT_real = np.sum(power_rt)
    sum_P_RT_virtual = np.sum(P_RT_virtual)

    if sum_P_RT_virtual < 1e-30:
        return np.zeros_like(P_RC_virtual)

    # eq. 8.4-8
    scale = sum_P_RT_real / sum_P_RT_virtual
    P_RC_real = scale * P_RC_virtual

    return P_RC_real


# ============================================================
# Step 7: Generate RC cluster angles (TR 38.901 eq. 8.4-10 ~ 8.4-18)
# ============================================================
def generate_rc_angles(rc_delays, P_RC_real, rt_paths, power_rt,
                       scenario_params, rng):
    """
    Generate 4 angles for each RC cluster.

    Parameters
    ----------
    rc_delays : array (L_RC,)
        RC cluster delays (seconds, normalized).
    P_RC_real : array (L_RC,)
        RC cluster powers (linear).
    rt_paths : list of dict
        RT paths with 'aoa','aod','zoa','zod' (radians).
    power_rt : array (L_RT,)
        RT path powers (linear).
    scenario_params : dict
        Sampled LSPs: ASA, ASD, ZSA, ZSD (degrees), K_R_dB, is_LOS,
        mu_offset_ZOD.
    rng : np.random.Generator

    Returns
    -------
    rc_angles : dict with keys 'aoa','aod','zoa','zod',
                each array (L_RC,) in radians.
    """
    from .lsp_calculator import lookup_C_phi, lookup_C_theta

    L_RC = len(rc_delays)
    if L_RC == 0:
        return {'aoa': np.array([]), 'aod': np.array([]),
                'zoa': np.array([]), 'zod': np.array([])}

    ASA = scenario_params['ASA']   # degrees
    ASD = scenario_params['ASD']   # degrees
    ZSA = scenario_params['ZSA']   # degrees
    ZSD = scenario_params['ZSD']   # degrees
    K_dB = scenario_params.get('K_R_dB', None)
    is_LOS = scenario_params['is_LOS']
    mu_offset_ZOD = scenario_params.get('mu_offset_ZOD', 0.0)  # degrees

    # RT angles (radians)
    phi_aoa_rt = np.array([p['aoa'] for p in rt_paths])   # radians
    phi_aod_rt = np.array([p['aod'] for p in rt_paths])
    theta_zoa_rt = np.array([p['zoa'] for p in rt_paths])
    theta_zod_rt = np.array([p['zod'] for p in rt_paths])

    # Total cluster count for C_phi/C_theta lookup
    L_RT = len(rt_paths)
    N_total = L_RT + L_RC

    C_phi = lookup_C_phi(N_total, K_dB, is_LOS)
    C_theta = lookup_C_theta(N_total, K_dB, is_LOS)

    # Reference power: max over all clusters (eq. 8.4-10)
    P_ref = max(np.max(P_RC_real), np.max(power_rt))

    # ── Center angles from RT (power-weighted circular mean, radians) ──
    # eq. 8.4-13
    w_rt = power_rt / (np.sum(power_rt) + 1e-30)
    phi_center_aoa = np.angle(np.sum(w_rt * np.exp(1j * phi_aoa_rt)))
    phi_center_aod = np.angle(np.sum(w_rt * np.exp(1j * phi_aod_rt)))
    theta_center_zoa = np.angle(np.sum(w_rt * np.exp(1j * theta_zoa_rt)))
    theta_center_zod = np.angle(np.sum(w_rt * np.exp(1j * theta_zod_rt)))

    # ── Generate angles for each RC cluster ──
    out_aoa = np.zeros(L_RC)
    out_aod = np.zeros(L_RC)
    out_zoa = np.zeros(L_RC)
    out_zod = np.zeros(L_RC)

    for n in range(L_RC):
        P_n = P_RC_real[n]
        ratio = max(P_n / P_ref, 1e-30)

        # ── Azimuth: wrapped Gaussian PAS (eq. 8.4-10) ──
        # phi_prime in degrees
        phi_prime = 2.0 * (ASA / 1.4) * np.sqrt(-np.log(ratio)) / C_phi

        X_aoa = rng.choice([-1, 1])
        Y_aoa = rng.normal(0, ASA / 7.0)  # jitter, degrees
        # Convert to radians and add center
        out_aoa[n] = np.deg2rad(X_aoa * phi_prime + Y_aoa) + phi_center_aoa

        # AOD: same formula with ASD
        phi_prime_d = 2.0 * (ASD / 1.4) * np.sqrt(-np.log(ratio)) / C_phi
        X_aod = rng.choice([-1, 1])
        Y_aod = rng.normal(0, ASD / 7.0)
        out_aod[n] = np.deg2rad(X_aod * phi_prime_d + Y_aod) + phi_center_aod

        # ── Zenith: Laplacian PAS (eq. 8.4-14) ──
        # theta_prime in degrees (no sqrt, unlike azimuth)
        theta_prime_a = -ZSA * np.log(ratio) / C_theta

        X_zoa = rng.choice([-1, 1])
        Y_zoa = rng.normal(0, ZSA / 7.0)
        out_zoa[n] = np.deg2rad(X_zoa * theta_prime_a + Y_zoa) + theta_center_zoa

        # ZOD: same + mu_offset_ZOD (eq. 8.4-18)
        theta_prime_d = -ZSD * np.log(ratio) / C_theta
        X_zod = rng.choice([-1, 1])
        Y_zod = rng.normal(0, ZSD / 7.0)
        out_zod[n] = (np.deg2rad(X_zod * theta_prime_d + Y_zod)
                      + theta_center_zod
                      + np.deg2rad(mu_offset_ZOD))  # offset in degrees → rad

    # ── Angle wrapping ──
    # Azimuth: wrap to [-pi, pi]
    out_aoa = np.mod(out_aoa + np.pi, 2 * np.pi) - np.pi
    out_aod = np.mod(out_aod + np.pi, 2 * np.pi) - np.pi

    # Zenith: wrap to [0, pi] (eq. 8.4-16)
    out_zoa = np.mod(out_zoa, 2 * np.pi)
    mask = out_zoa > np.pi
    out_zoa[mask] = 2 * np.pi - out_zoa[mask]

    out_zod = np.mod(out_zod, 2 * np.pi)
    mask = out_zod > np.pi
    out_zod[mask] = 2 * np.pi - out_zod[mask]

    return {'aoa': out_aoa, 'aod': out_aod,
            'zoa': out_zoa, 'zod': out_zod}


# ============================================================
# Step G: Merge and apply the power threshold  (TR 38.901 §8.4, Step 8)
# ============================================================
def merge_and_truncate(tau_rt_norm, power_rt, rt_paths,
                       tau_RC_scaled, P_RC_real, rc_angles, config,
                       Z_RC=None):
    """
    Merge RT and RC clusters into a single list.

    Parameters
    ----------
    tau_rt_norm : array
    power_rt : array
    rt_paths : list of dict
        RT paths with angle fields (radians).
    tau_RC_scaled : array
    P_RC_real : array
    rc_angles : dict
        Output of generate_rc_angles(): aoa/aod/zoa/zod arrays (radians).
    config : dict

    Returns
    -------
    clusters : list of dict
        Each dict has: delay, power, source, aoa, aod, zoa, zod (radians)
    """
    power_threshold_dB = config.get('power_threshold_dB', -25.0)
    if Z_RC is None:
        Z_RC = np.zeros(len(tau_RC_scaled), dtype=np.float64)
    else:
        Z_RC = np.asarray(Z_RC, dtype=np.float64)

    clusters = []

    # RT clusters (angles from rt_paths, radians)
    for j in range(len(tau_rt_norm)):
        clusters.append({
            'delay': float(tau_rt_norm[j]),
            'power': float(power_rt[j]),
            'source': 'RT',
            'aoa': float(rt_paths[j]['aoa']),  # radians
            'aod': float(rt_paths[j]['aod']),
            'zoa': float(rt_paths[j]['zoa']),
            'zod': float(rt_paths[j]['zod']),
        })

    # RC clusters (angles from generate_rc_angles, radians)
    for i in range(len(tau_RC_scaled)):
        clusters.append({
            'delay': float(tau_RC_scaled[i]),
            'power': float(P_RC_real[i]),
            'source': 'RC',
            'rc_index': int(i),
            'shadowing_db': float(Z_RC[i]) if i < len(Z_RC) else 0.0,
            'aoa': float(rc_angles['aoa'][i]),
            'aod': float(rc_angles['aod'][i]),
            'zoa': float(rc_angles['zoa'][i]),
            'zod': float(rc_angles['zod'][i]),
        })

    if not clusters:
        return clusters

    # Apply power threshold
    P_max = max(c['power'] for c in clusters)
    threshold = P_max * 10.0**(power_threshold_dB / 10.0)
    clusters = [c for c in clusters if c['power'] >= threshold]

    # Sort by increasing delay
    clusters.sort(key=lambda c: c['delay'])

    return clusters


# ============================================================
# Step H: Discretize the PDP
# ============================================================
def discretize_pdp(clusters, pdp_bin_size=1e-9):
    """
    Returns
    -------
    delay_axis : array (N_bins,)  — bin centers (seconds)
    power_array : array (N_bins,) — sum of powers in each bin
    """
    if not clusters:
        return np.array([]), np.array([])

    max_delay = max(c['delay'] for c in clusters)
    n_bins = int(np.ceil(max_delay / pdp_bin_size)) + 1
    power_array = np.zeros(n_bins)

    for c in clusters:
        idx = int(np.round(c['delay'] / pdp_bin_size))
        idx = min(idx, n_bins - 1)
        power_array[idx] += c['power']

    delay_axis = np.arange(n_bins) * pdp_bin_size
    return delay_axis, power_array


# ============================================================
# Compute metadata
# ============================================================
def compute_metadata(clusters, scenario_params):
    """Compute statistics of the merged cluster set."""
    L_RT = sum(1 for c in clusters if c['source'] == 'RT')
    L_RC = sum(1 for c in clusters if c['source'] == 'RC')

    delays = np.array([c['delay'] for c in clusters])
    powers = np.array([c['power'] for c in clusters])

    # RMS delay spread
    if len(delays) > 1 and np.sum(powers) > 0:
        w = powers / np.sum(powers)
        mean_d = np.sum(w * delays)
        DS_actual = np.sqrt(np.sum(w * (delays - mean_d)**2))
    else:
        DS_actual = 0.0

    return {
        'L_RT_final': L_RT,
        'L_RC_final': L_RC,
        'is_LOS': scenario_params['is_LOS'],
        'DS_target': scenario_params['DS'],
        'DS_actual': DS_actual,
    }


# ============================================================
# Main generation function
# ============================================================
def generate_hybrid_pdp(rt_paths, scenario_params=None, config=None,
                        seed=None, rng=None,
                        scenario_name=None, fc_GHz=None, elev_deg=None):
    """
    Fuse RT and RC paths into a map-based hybrid PDP.

    Parameters (new interface — preferred)
    ----------
    rt_paths : list of dict
        Each dictionary: {'tau_rt', 'power_rt', 'aoa', 'aod', 'zoa', 'zod'}
    scenario_name : str
        3GPP scenario name (e.g. 'UMi-LOS', 'NTN-DenseUrban-LOS')
    fc_GHz : float
        Carrier frequency in GHz (terrestrial scenarios)
    elev_deg : float
        Elevation angle (NTN scenarios)
    rng : np.random.Generator, optional
        Shared RNG (preferred over seed)
    seed : int, optional
        Fallback if rng not provided

    Parameters (legacy interface — deprecated)
    ----------
    scenario_params : dict
        Directly provide {'DS', 'K_R_dB', 'r_tau', 'zeta_dB',
        'N_cluster', 'is_LOS'}

    Returns
    -------
    dict with keys:
        'clusters': list of cluster dicts
        'pdp_array': (delay_axis, power_array)
        'metadata': dict
        'scenario_params': dict (the LSP used for this call)
    """
    if config is None:
        config = {}
    config.setdefault('p0', 0.2)
    config.setdefault('power_threshold_dB', -25.0)
    config.setdefault('pdp_bin_size', 1e-9)

    # ── RNG: prefer passed-in rng, fallback to seed ──
    if rng is None:
        rng = np.random.default_rng(seed)

    # ── Resolve scenario_params ──
    if scenario_params is not None and scenario_name is not None:
        raise ValueError("Provide scenario_params OR scenario_name, not both")

    if scenario_name is not None:
        # New interface: sample LSPs from table
        from .lsp_calculator import get_scenario_params, sample_lsps
        sp_table = get_scenario_params(scenario_name, fc_GHz=fc_GHz,
                                       elev_deg=elev_deg)
        scenario_params = sample_lsps(sp_table, rng)
    elif scenario_params is not None:
        # Legacy interface
        warnings.warn(
            "Passing scenario_params directly is deprecated. "
            "Use scenario_name='...' with fc_GHz or elev_deg instead.",
            DeprecationWarning, stacklevel=2)
    else:
        raise ValueError("Must provide scenario_params or scenario_name")

    # Extract RT inputs
    tau_rt = np.array([p['tau_rt'] for p in rt_paths])
    power_rt = np.array([p['power_rt'] for p in rt_paths])

    # Normalize delays
    tau_rt_norm = tau_rt - np.min(tau_rt)
    L_RT = len(tau_rt)

    # Step A
    mu_tau_RC, L_RC_prime = compute_mu_tau_RC(
        tau_rt_norm, L_RT, scenario_params)

    # Step B
    tau_RC_scaled, tau_RC_for_power = generate_rc_delays(
        mu_tau_RC, L_RC_prime, scenario_params, rng)

    # Step C
    tau_RC_scaled, tau_RC_for_power = remove_rc_clusters(
        tau_RC_scaled, tau_RC_for_power,
        tau_rt_norm, mu_tau_RC, config)

    L_RC = len(tau_RC_scaled)

    # Step D
    V_RC, V_RT, Z_RC, Z_RT = compute_virtual_powers(
        tau_RC_for_power, tau_rt_norm, scenario_params, rng,
        return_shadowing=True)

    # Step E
    P_RC_virtual, P_RT_virtual = normalize_virtual_powers(
        V_RC, V_RT, scenario_params)

    # Step F  ⭐
    P_RC_real = anchor_rc_power(P_RC_virtual, P_RT_virtual, power_rt)

    # Step 7: RC cluster angles (eq. 8.4-10 ~ 8.4-18)
    rc_angles = generate_rc_angles(
        tau_RC_scaled, P_RC_real, rt_paths, power_rt,
        scenario_params, rng)

    # Step G
    clusters = merge_and_truncate(
        tau_rt_norm, power_rt, rt_paths,
        tau_RC_scaled, P_RC_real, rc_angles, config,
        Z_RC=Z_RC)

    # Step H
    delay_axis, power_array = discretize_pdp(
        clusters, config['pdp_bin_size'])

    metadata = compute_metadata(clusters, scenario_params)

    return {
        'clusters': clusters,
        'pdp_array': (delay_axis, power_array),
        'metadata': metadata,
        'scenario_params': scenario_params,
        'rc_shadowing_db': Z_RC,
        'rt_shadowing_db': Z_RT,
    }


# ============================================================
# Step 9+10: Expand clusters into rays
# (TR 38.901 §8.4, Step 9: eq. 8.4-19~21, Step 10: K_B=1)
# ============================================================

# alpha_m for M=3: ±sqrt(3/2) quantiles + center
ALPHA_M_3 = np.array([0.0, 1.2247, -1.2247])


def expand_clusters_to_rays(merged_clusters, scenario_params,
                            config=None, seed=None, rng=None):
    """
    Expand each cluster into M rays with full (τ, P, AOA, AOD, ZOA, ZOD).

    LOS first cluster: M=1 (specular, no expansion).
    All other clusters: M=3, K_B=1 (equal power, angle offsets).

    Parameters
    ----------
    merged_clusters : list of dict
        From generate_hybrid_pdp(). Each dict has:
        delay, power, source, aoa, aod, zoa, zod (radians).
    scenario_params : dict
        Needs: c_DS, ASA, ASD, ZSA, ZSD (degrees), is_LOS.
    config : dict, optional
        'M' (default 3), 'is_LOS_first_cluster' (default False),
        'pdp_bin_size' (default 1e-9)
    rng : np.random.Generator, optional
    seed : int, optional

    Returns
    -------
    dict with keys:
        'rays': list of ray dicts (each has τ, P, AOA, AOD, ZOA, ZOD)
        'ray_pdp_array': (delay_axis, power_array)
        'metadata': dict
    """
    if config is None:
        config = {}
    config.setdefault('M', 3)
    config.setdefault('is_LOS_first_cluster', False)
    config.setdefault('pdp_bin_size', 1e-9)

    if rng is None:
        rng = np.random.default_rng(seed)

    c_DS = scenario_params.get('c_DS', 0.0)
    M = config['M']
    is_LOS = config['is_LOS_first_cluster']

    # Cluster angular spreads (degrees → used with alpha_m in degrees)
    c_ASA = scenario_params.get('ASA', 5.0)   # degrees
    c_ASD = scenario_params.get('ASD', 5.0)   # degrees
    c_ZSA = scenario_params.get('ZSA', 3.0)   # degrees
    # ZOD uses special formula: c_ZSD_eff = (3/8) * 10^(mu_lgZSD)
    ZSD_deg = scenario_params.get('ZSD', 3.0)  # degrees
    mu_lgZSD = np.log10(max(ZSD_deg, 1e-6))
    c_ZSD_eff = (3.0 / 8.0) * 10.0**mu_lgZSD   # degrees (eq. 8.4-21)

    # Select alpha_m offsets
    if M == 3:
        alpha_m = ALPHA_M_3
    else:
        # Fallback: generate quantile-based offsets for arbitrary M
        if M == 1:
            alpha_m = np.array([0.0])
        else:
            from scipy.stats import norm
            probs = (np.arange(M) + 0.5) / M
            alpha_m = norm.ppf(probs)

    sorted_clusters = sorted(merged_clusters, key=lambda c: c['delay'])

    rays = []
    for idx, cluster in enumerate(sorted_clusters):
        is_los_cluster = is_LOS and idx == 0

        # Check if cluster has angle fields
        has_angles = all(k in cluster for k in ('aoa', 'aod', 'zoa', 'zod'))

        if is_los_cluster:
            # LOS specular: M=1, no expansion
            ray = {
                'delay': cluster['delay'],
                'power': cluster['power'],
                'source': cluster['source'],
                'cluster_id': idx,
                'ray_id': 0,
                'is_los_ray': True,
            }
            if has_angles:
                ray['aoa'] = cluster['aoa']
                ray['aod'] = cluster['aod']
                ray['zoa'] = cluster['zoa']
                ray['zod'] = cluster['zod']
            rays.append(ray)
        else:
            # K_B = 1: equal power split, same delay, angle offsets
            # TODO: K_B > 1 angle decay (eq 8.4-24)
            P_ray = cluster['power'] / M   # equal split

            for m in range(M):
                ray = {
                    'delay': cluster['delay'],   # K_B=1: all rays same delay
                    'power': P_ray,
                    'source': cluster['source'],
                    'cluster_id': idx,
                    'ray_id': m,
                    'is_los_ray': False,
                }

                if has_angles:
                    a = alpha_m[m % len(alpha_m)]

                    # eq. 8.4-19: azimuth offsets (degrees → radians)
                    ray['aoa'] = cluster['aoa'] + np.deg2rad(c_ASA * a)
                    ray['aod'] = cluster['aod'] + np.deg2rad(c_ASD * a)

                    # eq. 8.4-20: ZOA offset
                    ray['zoa'] = cluster['zoa'] + np.deg2rad(c_ZSA * a)

                    # eq. 8.4-21: ZOD offset (uses c_ZSD_eff)
                    ray['zod'] = cluster['zod'] + np.deg2rad(c_ZSD_eff * a)

                    # ── Angle wrapping ──
                    # Azimuth → [-pi, pi]
                    ray['aoa'] = float(np.mod(ray['aoa'] + np.pi, 2*np.pi) - np.pi)
                    ray['aod'] = float(np.mod(ray['aod'] + np.pi, 2*np.pi) - np.pi)

                    # Zenith → [0, pi]
                    zoa = np.mod(ray['zoa'], 2*np.pi)
                    if zoa > np.pi:
                        zoa = 2*np.pi - zoa
                    ray['zoa'] = float(zoa)

                    zod = np.mod(ray['zod'], 2*np.pi)
                    if zod > np.pi:
                        zod = 2*np.pi - zod
                    ray['zod'] = float(zod)

                rays.append(ray)

    # Discretize the PDP
    ray_pdp = _discretize_pdp_from_rays(rays, config['pdp_bin_size'])

    total_power_clusters = sum(c['power'] for c in merged_clusters)
    total_power_rays = sum(r['power'] for r in rays)

    return {
        'rays': rays,
        'ray_pdp_array': ray_pdp,
        'metadata': {
            'total_rays': len(rays),
            'M': M,
            'c_DS': c_DS,
            'total_power_clusters': total_power_clusters,
            'total_power_rays': total_power_rays,
        },
    }


def _discretize_pdp_from_rays(rays, pdp_bin_size=1e-9):
    """Accumulate ray powers on a discrete delay grid."""
    if not rays:
        return np.array([]), np.array([])

    max_delay = max(r['delay'] for r in rays)
    n_bins = int(np.ceil(max_delay / pdp_bin_size)) + 1
    power_array = np.zeros(n_bins)

    for r in rays:
        idx = int(np.round(r['delay'] / pdp_bin_size))
        idx = min(idx, n_bins - 1)
        power_array[idx] += r['power']

    delay_axis = np.arange(n_bins) * pdp_bin_size
    return delay_axis, power_array


# ============================================================
# Step 11: XPR sampling (TR 38.901 eq. 8.4-26)
# ============================================================
def generate_xpr(rays, scenario_params, rng):
    """
    Sample per-ray XPR from log-normal distribution.

    Parameters
    ----------
    rays : list of ray dicts
    scenario_params : dict
        Needs 'mu_XPR' (dB), 'sigma_XPR' (dB).
    rng : np.random.Generator

    Returns
    -------
    rays : same list, each ray now has 'xpr' field (linear scale)
    """
    mu = scenario_params.get('mu_XPR', 9.0)
    sigma = scenario_params.get('sigma_XPR', 3.0)

    for r in rays:
        # eq. 8.4-26: X ~ N(mu, sigma^2) in dB
        X_dB = rng.normal(mu, sigma)
        r['xpr'] = 10.0**(X_dB / 10.0)  # linear

    return rays


# ============================================================
# Step 12: Initial phase sampling
# ============================================================
def generate_initial_phases(rays, scenario_params, rng,
                            tx_rx_distance_3D=None):
    """
    Sample 4 random initial phases per ray (θθ, θφ, φθ, φφ).
    LOS specular ray: θθ and φφ set to geometric phase.

    Parameters
    ----------
    rays : list of ray dicts
    scenario_params : dict
        Needs 'is_LOS'. If LOS, needs fc_GHz or wavelength info.
    rng : np.random.Generator
    tx_rx_distance_3D : float or None
        3D distance in meters. Required for LOS phase computation.
        If None and LOS, uses tau_LOS * c to estimate.

    Returns
    -------
    rays : same list, each ray now has 'phase' dict
    """
    is_LOS = scenario_params.get('is_LOS', False)
    fc_GHz = scenario_params.get('fc_GHz', None)

    # Wavelength
    if fc_GHz is not None:
        wavelength = 0.3 / fc_GHz  # meters
    else:
        wavelength = 0.3 / 28.0  # fallback 28 GHz

    # LOS geometric phase
    Phi_LOS = 0.0
    if is_LOS and tx_rx_distance_3D is not None:
        Phi_LOS = -2.0 * np.pi * tx_rx_distance_3D / wavelength
    elif is_LOS:
        # Estimate from LOS ray delay
        los_rays = [r for r in rays if r.get('is_los_ray', False)]
        if los_rays:
            d_est = los_rays[0]['delay'] * 3e8  # tau * c, approximate
            Phi_LOS = -2.0 * np.pi * d_est / wavelength
            warnings.warn("tx_rx_distance_3D not provided; estimating from LOS delay")

    for r in rays:
        if r.get('is_los_ray', False):
            # LOS specular: θθ and φφ = geometric phase
            r['phase'] = {
                'theta_theta': float(Phi_LOS),
                'theta_phi':   float(rng.uniform(-np.pi, np.pi)),
                'phi_theta':   float(rng.uniform(-np.pi, np.pi)),
                'phi_phi':     float(Phi_LOS),
            }
        else:
            r['phase'] = {
                'theta_theta': float(rng.uniform(-np.pi, np.pi)),
                'theta_phi':   float(rng.uniform(-np.pi, np.pi)),
                'phi_theta':   float(rng.uniform(-np.pi, np.pi)),
                'phi_phi':     float(rng.uniform(-np.pi, np.pi)),
            }

    return rays


# ============================================================
# Step 13: H(t) channel coefficient assembly (Eq. 8.4-27)
# Simplified: isotropic single-pol (θ-pol only)
# ============================================================
def compute_channel_coefficients(
    rays,
    rx_array_geom,           # shape (U, 3), meters
    tx_array_geom,           # shape (S, 3), meters
    fc_GHz,
    velocity_vec=None,       # shape (3,), m/s
    t=0.0,                   # seconds
):
    """
    Compute MIMO channel coefficients H_{u,s,n,m}(t).

    Isotropic single-pol (θ-pol): F_θ=1, F_φ=0.
    Under this assumption, only the θθ element of the polarization
    matrix survives.

    Parameters
    ----------
    rays : list of ray dicts
        Each must have: delay, power, aoa, aod, zoa, zod,
        phase['theta_theta'].
    rx_array_geom : ndarray (U, 3)
    tx_array_geom : ndarray (S, 3)
    fc_GHz : float
    velocity_vec : ndarray (3,) or None
    t : float

    Returns
    -------
    dict with 'H' (U,S,N,M) complex, 'tau' (N,M) float, metadata
    """
    wavelength = 0.3 / fc_GHz  # meters
    k = 2.0 * np.pi / wavelength

    if velocity_vec is None:
        velocity_vec = np.zeros(3)
    velocity_vec = np.asarray(velocity_vec, dtype=np.float64)

    rx_pos = np.asarray(rx_array_geom, dtype=np.float64)  # (U, 3)
    tx_pos = np.asarray(tx_array_geom, dtype=np.float64)  # (S, 3)
    U = rx_pos.shape[0]
    S = tx_pos.shape[0]

    # ── Organize rays into (N, M) structure ──
    cluster_ids = sorted(set(r['cluster_id'] for r in rays))
    N = len(cluster_ids)
    cid_map = {cid: n for n, cid in enumerate(cluster_ids)}

    # Find max M
    M_max = max(
        sum(1 for r in rays if r['cluster_id'] == cid)
        for cid in cluster_ids
    )

    # Build arrays: power, phase, angles, delays — shape (N, M_max)
    P = np.zeros((N, M_max))
    phi_tt = np.zeros((N, M_max))  # θθ phase
    aoa = np.zeros((N, M_max))     # radians
    aod = np.zeros((N, M_max))
    zoa = np.zeros((N, M_max))
    zod = np.zeros((N, M_max))
    tau = np.zeros((N, M_max))
    valid = np.zeros((N, M_max), dtype=bool)

    for r in rays:
        n = cid_map[r['cluster_id']]
        m = r['ray_id']
        if m >= M_max:
            continue
        P[n, m] = r['power']
        phi_tt[n, m] = r['phase']['theta_theta']
        aoa[n, m] = r['aoa']
        aod[n, m] = r['aod']
        zoa[n, m] = r['zoa']
        zod[n, m] = r['zod']
        tau[n, m] = r['delay']
        valid[n, m] = True

    # ── Direction vectors (spherical → cartesian) ──
    # r_rx_hat: arrival direction, shape (N, M, 3)
    r_rx = np.stack([
        np.sin(zoa) * np.cos(aoa),
        np.sin(zoa) * np.sin(aoa),
        np.cos(zoa),
    ], axis=-1)  # (N, M, 3)

    r_tx = np.stack([
        np.sin(zod) * np.cos(aod),
        np.sin(zod) * np.sin(aod),
        np.cos(zod),
    ], axis=-1)  # (N, M, 3)

    # ── Array phase: Ψ_{u,s,n,m} ──
    # rx phase: k * r_rx · d_u → (N, M, U)
    psi_rx = k * np.einsum('nmi,ui->nmu', r_rx, rx_pos)  # (N, M, U)
    # tx phase: k * r_tx · d_s → (N, M, S)
    psi_tx = k * np.einsum('nmi,si->nms', r_tx, tx_pos)  # (N, M, S)

    # ── Doppler phase ──
    # k * (r_rx · v) * t → (N, M)
    doppler = k * np.einsum('nmi,i->nm', r_rx, velocity_vec) * t  # (N, M)

    # ── Assemble H_{u,s,n,m} ──
    # H = sqrt(P) * exp(j*phi_tt) * exp(j*psi_rx) * exp(j*psi_tx) * exp(j*doppler)
    # Isotropic θ-pol: only θθ phase survives

    sqrt_P = np.sqrt(P)  # (N, M)
    phase_base = phi_tt + doppler  # (N, M)

    # Broadcast: (U, S, N, M)
    # psi_rx: (N, M, U) → transpose to (U, N, M) → expand S
    # psi_tx: (N, M, S) → transpose to (S, N, M) → expand U
    H = (sqrt_P[np.newaxis, np.newaxis, :, :]          # (1,1,N,M)
         * np.exp(1j * phase_base[np.newaxis, np.newaxis, :, :])  # (1,1,N,M)
         * np.exp(1j * psi_rx.transpose(2, 0, 1)[np.newaxis, :, :, :].transpose(1, 0, 2, 3))  # need (U,1,N,M)
         * np.exp(1j * psi_tx.transpose(2, 0, 1)[np.newaxis, :, :, :].transpose(0, 1, 2, 3))  # need (1,S,N,M)
         )

    # Cleaner broadcast:
    # psi_rx shape (N,M,U) → (U,N,M) via transpose(2,0,1) → (U,1,N,M)
    psi_rx_usable = psi_rx.transpose(2, 0, 1)[:, np.newaxis, :, :]  # (U,1,N,M)
    # psi_tx shape (N,M,S) → (S,N,M) via transpose(2,0,1) → (1,S,N,M)
    psi_tx_usable = psi_tx.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1,S,N,M)

    H = (sqrt_P[np.newaxis, np.newaxis, :, :]
         * np.exp(1j * phase_base[np.newaxis, np.newaxis, :, :])
         * np.exp(1j * psi_rx_usable)
         * np.exp(1j * psi_tx_usable)
         ).astype(np.complex64)  # (U, S, N, M)

    # Zero out invalid entries
    H[:, :, ~valid] = 0.0

    return {
        'H': H,
        'tau': tau,
        'fc_GHz': fc_GHz,
        'wavelength': wavelength,
        't': t,
        'metadata': {
            'num_rx_ant': U,
            'num_tx_ant': S,
            'num_clusters': N,
            'M_max': M_max,
            'total_rays': int(np.sum(valid)),
        },
    }


# ============================================================
# Top-level: generate_full_channel()
# ============================================================
def generate_full_channel(
    rt_paths,
    scenario_name,
    rx_array_geom,
    tx_array_geom,
    elev_deg=None,
    fc_GHz=None,
    velocity_vec=None,
    t=0.0,
    rng=None,
    config=None,
    tx_rx_distance_3D=None,
):
    """
    Full pipeline: RT → hybrid PDP → angles → rays → XPR → phase → H(t).

    Parameters
    ----------
    rt_paths : list of dict
        Each: {tau_rt, power_rt, aoa, aod, zoa, zod}
    scenario_name : str
    rx_array_geom : ndarray (U, 3), meters
    tx_array_geom : ndarray (S, 3), meters
    elev_deg : float (NTN)
    fc_GHz : float (terrestrial; if None, inferred from scenario)
    velocity_vec : ndarray (3,) or None
    t : float, seconds
    rng : np.random.Generator or None
    config : dict or None
    tx_rx_distance_3D : float or None, meters

    Returns
    -------
    dict with all intermediate and final results
    """
    if rng is None:
        rng = np.random.default_rng()

    if config is None:
        config = {}
    hybrid_config = {
        'p0': config.get('p0', 0.2),
        'power_threshold_dB': config.get('power_threshold_dB', -25.0),
        'pdp_bin_size': config.get('pdp_bin_size', 1e-9),
    }
    ray_config = {
        'M': config.get('M', 3),
        'is_LOS_first_cluster': config.get('is_LOS_first_cluster', True),
        'pdp_bin_size': config.get('pdp_bin_size', 1e-9),
    }

    # Step 3-8: Hybrid PDP with sampled LSPs + RC angles
    hybrid = generate_hybrid_pdp(
        rt_paths, config=hybrid_config, rng=rng,
        scenario_name=scenario_name, fc_GHz=fc_GHz, elev_deg=elev_deg)

    clusters = hybrid['clusters']
    sp = hybrid['scenario_params']

    # Infer fc_GHz if not given
    if fc_GHz is None:
        fc_GHz = sp.get('fc_GHz', 28.0)
    # Store fc_GHz in sp for downstream
    sp['fc_GHz'] = fc_GHz

    # Estimate d_3D from LOS delay if not provided
    if tx_rx_distance_3D is None:
        los_tau = [p['tau_rt'] for p in rt_paths]
        if los_tau:
            tx_rx_distance_3D = min(los_tau) * 3e8

    # Step 9-10: Ray expansion (delay + power + angles)
    ray_result = expand_clusters_to_rays(clusters, sp, ray_config, rng=rng)
    rays = ray_result['rays']

    # Step 11: XPR
    rays = generate_xpr(rays, sp, rng)

    # Step 12: Initial phases
    rays = generate_initial_phases(rays, sp, rng,
                                   tx_rx_distance_3D=tx_rx_distance_3D)

    # Step 13: H(t)
    channel = compute_channel_coefficients(
        rays, rx_array_geom, tx_array_geom, fc_GHz,
        velocity_vec=velocity_vec, t=t)

    return {
        'hybrid': hybrid,
        'ray_result': ray_result,
        'rays': rays,
        'channel': channel,
        'scenario_params': sp,
        'config': {'hybrid': hybrid_config, 'ray': ray_config},
    }

"""Hybrid generation steps, starting at generate_full_channel. Preserve standard formulas, filtering and RNG order. RT-derived clusters use statistical polarization reconstruction; native RT phase is not preserved."""

from copy import deepcopy
import numpy as np
from scipy.constants import c
from scipy.special import j1
from .lsp_calculator import get_scenario_params, sample_lsps
from .ionosphere import faraday_state, angle_at_frequency, rotation, faraday_inputs
from .topology import elevation_deg, endpoint_names

ALPHA_M_3 = np.array([0.0, 1.2247, -1.2247])
MODEL_REVISION = "tr38901_8_4_kb1_20261010"


def generate_full_channel(
    rt_paths,
    scenario_name,
    rx_array_geom,
    tx_array_geom,
    *,
    elev_deg,
    fc_GHz,
    rng,
    config,
    tx_rx_distance_3D,
    propagation_context=None,
):
    """Fixed-endpoint KB=1 generation; RT inputs must first pass the -25 dB threshold."""
    if (
        config["M"] != 20
        or config["ray_delay_mode"] != "kb1"
        or config["table_profile"] != "tr38811_corrected"
    ):
        raise ValueError("Standard route requires corrected NTN tables, M=20, KB=1")
    if config["power_threshold_dB"] != -25:
        raise ValueError("Standard route uses the Clause-8 -25 dB threshold")
    paths = deepcopy(rt_paths)

    # Step 4: correlated LOS/NLOS LSPs; do not apply global SF or FSPL twice.
    table = get_scenario_params(
        scenario_name,
        fc_GHz=fc_GHz,
        elev_deg=elev_deg,
        table_profile="tr38811_corrected",
    )
    sp = sample_lsps(table, rng)
    sp.update(
        is_LOS=scenario_name.endswith("-LOS"), fc_GHz=fc_GHz, source=table["source"]
    )
    if not sp["is_LOS"]:
        sp["K_R_dB"] = None
    tau_rt = np.array([p["tau_rt"] for p in paths])
    tau_rt -= tau_rt.min()
    power_rt = np.array([p["power_rt"] for p in paths])

    # Step 5: corrected sample count, C_tau and cluster-removal delays.
    mu, count = compute_mu_tau_RC(tau_rt, len(paths), sp)
    delays, power_delays = generate_rc_delays(mu, count, sp, rng)
    delays, power_delays = remove_rc_clusters(
        delays,
        power_delays,
        tau_rt,
        mu,
        config["p0"],
    )

    # Step 6: retain the verified virtual powers, K peak and RT anchoring formulas.
    vrc, vrt, zrc, zrt = compute_virtual_powers(
        power_delays,
        tau_rt,
        sp,
        rng,
        return_shadowing=True,
    )
    prc, prt = normalize_virtual_powers(vrc, vrt, sp)
    real_rc = anchor_rc_power(prc, prt, power_rt)

    # Steps 7-8: RC angles and RT/RC merging; apply the same -25 dB threshold to both.
    angles = generate_rc_angles(delays, real_rc, paths, power_rt, sp, rng)
    clusters = merge_and_truncate(
        tau_rt,
        power_rt,
        paths,
        delays,
        real_rc,
        angles,
        config,
        Z_RC=zrc,
    )
    if sp["is_LOS"] and (
        not clusters
        or clusters[0]["source"] != "RT"
        or clusters[0].get("rt_index") != 0
    ):
        raise NotImplementedError(
            "LOS cluster was truncated by merged power threshold; explicit LOS handling is required"
        )
    # Steps 9-10: M=20 offsets, equal delays/powers and random angle coupling.
    ray_result = expand_clusters_to_rays(
        clusters,
        sp,
        dict(config, is_LOS_first_cluster=sp["is_LOS"]),
        rng=rng,
    )
    rays = couple_ray_angles(ray_result["rays"], rng)

    # Step 11: RC table mean/std; the RT-derived mean uses the probe power ratio.
    rays = generate_xpr(rays, sp, rng)
    prepare_hybrid_polarization(rays, paths, sp, {"rt_xpr_from_rt": True})

    # Step 12: compute the LOS carrier phase using the SI speed of light.
    rays = draw_initial_phases(rays, fc_GHz * 1e9, tx_rx_distance_3D, rng)
    pol = dict(config["polarization"])
    fr = config["faraday"]
    if pol["direction"] == "uplink":
        # Reverse arrival/departure and cross-polar phases for the same realization.
        rays = reverse_paths(rays)

    # Step 13: DL uses P @ F; UL uses F @ P_UL, followed by antenna projection.
    channel = assemble_coefficients(
        rays,
        rx_array_geom,
        tx_array_geom,
        fc_GHz * 1e9,
        pol,
        fr,
        propagation_context,
    )
    return {
        "rays": rays,
        "channel": channel,
        "scenario_params": sp,
        "hybrid": {"clusters": clusters, "scenario_params": sp},
        "config": dict(config, model_revision=MODEL_REVISION),
        # Expose sampled state without resampling or changing RNG order; reuse for temporal evolution.
        "sampling_state": {
            "rc_delays_s": delays,
            "rc_power_delays_s": power_delays,
            "rc_angles": angles,
            "zrc_db": zrc,
            "zrt_db": zrt,
        },
    }


def generate_rc_delays(mu_tau_rc, count, scenario_params, rng):
    """Step 5 (8.4-1..3): n=0..L_RC_prime; divide output delays by C_tau, retaining unscaled power delays."""
    x = rng.uniform(0, 1, size=count + 1)
    raw = -mu_tau_rc * np.log(np.maximum(x, np.finfo(float).tiny))
    unscaled = np.sort(raw - raw.min())
    k = scenario_params.get("K_R_dB")
    ct = (
        0.7705 - 0.0433 * k + 0.0002 * k * k + 0.000017 * k**3
        if scenario_params["is_LOS"]
        else 1.0
    )
    if not np.isfinite(ct) or ct <= 0:
        raise ValueError("Sampled K gives invalid C_tau; no silent clipping")
    return unscaled / ct, unscaled


def remove_rc_clusters(scaled, unscaled, tau_rt, mu_tau_rc, p0):
    """Step 5: remove n=0 and RCs whose output delays are too close to RT delays."""
    if not 0 <= p0 < 1:
        raise ValueError("p0 must lie in [0,1)")
    threshold = -mu_tau_rc * np.log1p(-p0)
    keep = np.arange(len(scaled)) != 0
    keep &= ~np.any(np.abs(scaled[:, None] - tau_rt[None, :]) < threshold, axis=1)
    return scaled[keep], unscaled[keep]


def couple_ray_angles(rays, rng):
    """Step 10 KB=1: retain marginal angles and independently permute AOD, ZOA and ZOD against fixed AOA order. This implements random azimuth/zenith coupling; the single LOS ray is unchanged."""
    rays = deepcopy(rays)
    for cid in sorted({r["cluster_id"] for r in rays}):
        group = [r for r in rays if r["cluster_id"] == cid]
        if len(group) == 1:
            continue
        for key in ("aod", "zoa", "zod"):
            shuffled = rng.permutation([r[key] for r in group])
            for ray, value in zip(group, shuffled):
                ray[key] = float(value)
    return rays


def draw_initial_phases(rays, frequency_hz, distance_m, rng):
    """Step 12: four independent NLOS phases; LOS geometric phase uses the SI vacuum speed of light."""
    for ray in rays:
        ray["phase"] = dict(
            zip(
                ("theta_theta", "theta_phi", "phi_theta", "phi_phi"),
                rng.uniform(-np.pi, np.pi, size=4).tolist(),
            )
        )
        if ray.get("is_los_ray", False):
            phase = -2 * np.pi * frequency_hz * distance_m / c
            ray["phase"]["theta_theta"] = float(phase)
            ray["phase"]["phi_phi"] = float(phase)
    return rays


def assemble_coefficients(
    rays, rx_array_m, tx_array_m, frequency_hz, polarization, faraday, context=None
):
    """Step 13: Jones matrices, antenna projection and array phase; no extra Doppler for static snapshots."""
    rx = np.asarray(rx_array_m, float)
    tx = np.asarray(tx_array_m, float)
    state = faraday_state(faraday, context, direction=polarization["direction"])
    gains = polarized_path_gains(rays, polarization, state, frequency_hz)
    ids = {cid: i for i, cid in enumerate(sorted({r["cluster_id"] for r in rays}))}
    nr, nt = len(port_vectors(polarization["rx"])), len(
        port_vectors(polarization["tx"])
    )
    h = np.zeros(
        (len(rx) * nr, len(tx) * nt, len(ids), max(r["ray_id"] for r in rays) + 1),
        complex,
    )
    tau = np.zeros(h.shape[-2:])

    def direction(zenith, azimuth):
        return np.array(
            [
                np.sin(zenith) * np.cos(azimuth),
                np.sin(zenith) * np.sin(azimuth),
                np.cos(zenith),
            ]
        )

    for ray, gain in zip(rays, gains):
        u_rx, u_tx = direction(ray["zoa"], ray["aoa"]), direction(
            ray["zod"], ray["aod"]
        )
        phase = (
            2 * np.pi * frequency_hz / c * ((rx @ u_rx)[:, None] + (tx @ u_tx)[None, :])
        )
        field = np.exp(1j * phase)[:, None, :, None] * gain[None, :, None, :]
        i, j = ids[ray["cluster_id"]], ray["ray_id"]
        h[:, :, i, j] = field.reshape(h.shape[:2])
        tau[i, j] = ray["delay"]
    return {
        "H": h,
        "tau": tau,
        "fc_GHz": frequency_hz / 1e9,
        "wavelength": c / frequency_hz,
        "metadata": {
            "faraday": dict(
                state,
                psi_rad=float(angle_at_frequency(state, frequency_hz)),
                propagation_psi_rad=state["propagation_coefficient_rad_hz2"]
                / frequency_hz**2,
                implementation_revision="ul_dl_satellite_basis_20261016",
            )
        },
    }


def compute_mu_tau_RC(tau_rt_norm, L_RT, scenario_params):
    """Estimate RC delay scale and candidate count from RT excess delays in seconds; return seconds and an integer count."""
    DS = scenario_params["DS"]
    r_tau = scenario_params["r_tau"]
    L_RC_prime = scenario_params["N_cluster"]

    tau_rt_mean = np.mean(tau_rt_norm)

    # TR 38.901 §8.4, text below eq. 8.4-1
    mu_tau = r_tau * DS + (L_RT / (L_RC_prime + 1)) * (r_tau * DS - tau_rt_mean)
    mu_tau_RC = max(mu_tau, tau_rt_mean)

    return mu_tau_RC, L_RC_prime


def compute_virtual_powers(
    tau_RC_for_power,
    tau_rt_norm,
    scenario_params,
    rng,
    Z_RC=None,
    Z_RT=None,
    return_shadowing=False,
):
    """Compute virtual RC/RT powers from delays in seconds and shadowing Z in dB. Supplied Z avoids resampling. Outputs do not yet include actual RT gain."""
    DS = scenario_params["DS"]
    r_tau = scenario_params["r_tau"]
    zeta_dB = scenario_params["zeta_dB"]

    L_RC = len(tau_RC_for_power)
    L_RT = len(tau_rt_norm)

    # eq. 8.4-4: V_RC_i
    if Z_RC is None:
        Z_RC = rng.normal(0, zeta_dB, size=L_RC)
    else:
        Z_RC = np.asarray(Z_RC, dtype=np.float64)
    V_RC = np.exp(-tau_RC_for_power * (r_tau - 1) / (r_tau * DS)) * 10.0 ** (
        -Z_RC / 10.0
    )

    # eq. 8.4-5: V_RT_j
    if Z_RT is None:
        Z_RT = rng.normal(0, zeta_dB, size=L_RT)
    else:
        Z_RT = np.asarray(Z_RT, dtype=np.float64)
    V_RT = np.exp(-tau_rt_norm * (r_tau - 1) / (r_tau * DS)) * 10.0 ** (-Z_RT / 10.0)

    if return_shadowing:
        return V_RC, V_RT, Z_RC, Z_RT
    return V_RC, V_RT


def normalize_virtual_powers(V_RC, V_RT, scenario_params):
    """Allocate virtual power using LOS state and K_R_dB; NLOS has no LOS peak allocation."""
    is_LOS = scenario_params["is_LOS"]
    K_R_dB = scenario_params.get("K_R_dB", None)

    # eq. 8.4-6
    if is_LOS and K_R_dB is not None:
        A = 10.0 ** (K_R_dB / 10.0)
    else:
        A = 0.0

    V_sum = np.sum(V_RC) + np.sum(V_RT)

    # eq. 8.4-7
    P_RC_virtual = (1.0 / (A + 1.0)) * V_RC / V_sum
    P_RT_virtual = (1.0 / (A + 1.0)) * V_RT / V_sum

    # Add the LOS peak to the first RT path.
    if len(P_RT_virtual) > 0:
        P_RT_virtual[0] += A / (A + 1.0)

    return P_RC_virtual, P_RT_virtual


def anchor_rc_power(P_RC_virtual, P_RT_virtual, power_rt):
    """Scale RC using actual linear RT power without renormalizing RT or applying free-space loss again."""
    sum_P_RT_real = np.sum(power_rt)
    sum_P_RT_virtual = np.sum(P_RT_virtual)

    if sum_P_RT_virtual < 1e-30:
        return np.zeros_like(P_RC_virtual)

    # eq. 8.4-8
    scale = sum_P_RT_real / sum_P_RT_virtual
    P_RC_real = scale * P_RC_virtual

    return P_RC_real


def generate_rc_angles(rc_delays, P_RC_real, rt_paths, power_rt, scenario_params, rng):
    """Generate four RC angles around RT directions. LSP spreads are in degrees; output angles are radians. Preserve RNG order."""
    from .lsp_calculator import lookup_C_phi, lookup_C_theta

    L_RC = len(rc_delays)
    if L_RC == 0:
        return {
            "aoa": np.array([]),
            "aod": np.array([]),
            "zoa": np.array([]),
            "zod": np.array([]),
        }

    ASA = scenario_params["ASA"]  # degrees
    ASD = scenario_params["ASD"]  # degrees
    ZSA = scenario_params["ZSA"]  # degrees
    ZSD = scenario_params["ZSD"]  # degrees
    K_dB = scenario_params.get("K_R_dB", None)
    is_LOS = scenario_params["is_LOS"]
    mu_offset_ZOD = scenario_params.get("mu_offset_ZOD", 0.0)  # degrees

    # RT angles (radians)
    phi_aoa_rt = np.array([p["aoa"] for p in rt_paths])  # radians
    phi_aod_rt = np.array([p["aod"] for p in rt_paths])
    theta_zoa_rt = np.array([p["zoa"] for p in rt_paths])
    theta_zod_rt = np.array([p["zod"] for p in rt_paths])

    # Total cluster count for C_phi/C_theta lookup
    L_RT = len(rt_paths)
    N_total = L_RT + L_RC

    C_phi = lookup_C_phi(
        N_total,
        K_dB,
        is_LOS,
        table_profile=scenario_params.get("table_profile", "legacy"),
    )
    C_theta = lookup_C_theta(
        N_total,
        K_dB,
        is_LOS,
        table_profile=scenario_params.get("table_profile", "legacy"),
    )

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
        out_zod[n] = (
            np.deg2rad(X_zod * theta_prime_d + Y_zod)
            + theta_center_zod
            + np.deg2rad(mu_offset_ZOD)
        )  # offset in degrees → rad

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

    return {"aoa": out_aoa, "aod": out_aod, "zoa": out_zoa, "zod": out_zod}


def merge_and_truncate(
    tau_rt_norm,
    power_rt,
    rt_paths,
    tau_RC_scaled,
    P_RC_real,
    rc_angles,
    config,
    Z_RC=None,
):
    """Merge RT/RC clusters and threshold relative to their combined peak. This can remove RT clusters, preserving the existing model."""
    power_threshold_dB = config.get("power_threshold_dB", -25.0)
    if Z_RC is None:
        Z_RC = np.zeros(len(tau_RC_scaled), dtype=np.float64)
    else:
        Z_RC = np.asarray(Z_RC, dtype=np.float64)

    clusters = []

    # RT clusters (angles from rt_paths, radians)
    for j in range(len(tau_rt_norm)):
        clusters.append(
            {
                "delay": float(tau_rt_norm[j]),
                "power": float(power_rt[j]),
                "source": "RT",
                # Keep the originating deterministic path identity. Propagation
                # extensions can then map each expanded ray back to Sionna's
                # actual path geometry (for example for path-specific STEC).
                "rt_index": int(j),
                "aoa": float(rt_paths[j]["aoa"]),  # radians
                "aod": float(rt_paths[j]["aod"]),
                "zoa": float(rt_paths[j]["zoa"]),
                "zod": float(rt_paths[j]["zod"]),
            }
        )

    # RC clusters (angles from generate_rc_angles, radians)
    for i in range(len(tau_RC_scaled)):
        clusters.append(
            {
                "delay": float(tau_RC_scaled[i]),
                "power": float(P_RC_real[i]),
                "source": "RC",
                "rc_index": int(i),
                "shadowing_db": float(Z_RC[i]) if i < len(Z_RC) else 0.0,
                "aoa": float(rc_angles["aoa"][i]),
                "aod": float(rc_angles["aod"][i]),
                "zoa": float(rc_angles["zoa"][i]),
                "zod": float(rc_angles["zod"][i]),
            }
        )

    if not clusters:
        return clusters

    # Power truncation.
    P_max = max(c["power"] for c in clusters)
    threshold = P_max * 10.0 ** (power_threshold_dB / 10.0)
    clusters = [c for c in clusters if c["power"] >= threshold]

    # Sort by increasing delay.
    clusters.sort(key=lambda c: c["delay"])

    return clusters


def expand_clusters_to_rays(
    merged_clusters, scenario_params, config=None, seed=None, rng=None
):
    """Expand clusters with M=20, KB=1 in the main flow. The first LOS cluster has one ray; other rays share cluster delay and equal power. Return rays and existing statistics."""
    if config is None:
        config = {}
    config.setdefault("M", 3)
    config.setdefault("is_LOS_first_cluster", False)
    config.setdefault("pdp_bin_size", 1e-9)

    if rng is None:
        rng = np.random.default_rng(seed)

    delay_mode = config.get("ray_delay_mode", "legacy_delay")
    if delay_mode not in ("legacy_delay", "kb1"):
        raise ValueError("ray_delay_mode must be legacy_delay or kb1")
    c_DS = scenario_params.get("c_DS", 0.0)
    if delay_mode == "legacy_delay" and (not np.isfinite(c_DS) or c_DS <= 0):
        raise ValueError("Historical delay expansion requires finite c_DS > 0 seconds")
    M = config["M"]
    is_LOS = config["is_LOS_first_cluster"]

    # Cluster angular spreads (degrees → used with alpha_m in degrees)
    c_ASA = scenario_params.get("ASA", 5.0)  # degrees
    c_ASD = scenario_params.get("ASD", 5.0)  # degrees
    c_ZSA = scenario_params.get("ZSA", 3.0)  # degrees
    # ZOD uses special formula: c_ZSD_eff = (3/8) * 10^(mu_lgZSD)
    ZSD_deg = scenario_params.get("ZSD", 3.0)  # degrees
    mu_lgZSD = np.log10(max(ZSD_deg, 1e-6))
    c_ZSD_eff = (3.0 / 8.0) * 10.0**mu_lgZSD  # degrees (eq. 8.4-21)

    if scenario_params.get("table_profile") == "tr38811_corrected":
        c_ASA = scenario_params["c_ASA"]
        c_ASD = scenario_params["c_ASD"]
        c_ZSA = scenario_params["c_ZSA"]
        c_ZSD_eff = scenario_params["c_ZSD"]

    # Select alpha_m offsets
    if M == 20 and scenario_params.get("table_profile") == "tr38811_corrected":
        from .lsp_calculator import ALPHA

        alpha_m = ALPHA
    elif M == 3:
        alpha_m = ALPHA_M_3
    else:
        # Fallback: generate quantile-based offsets for arbitrary M
        if M == 1:
            alpha_m = np.array([0.0])
        else:
            from scipy.stats import norm

            probs = (np.arange(M) + 0.5) / M
            alpha_m = norm.ppf(probs)

    sorted_clusters = sorted(merged_clusters, key=lambda c: c["delay"])

    rays = []
    for idx, cluster in enumerate(sorted_clusters):
        is_los_cluster = is_LOS and idx == 0

        # Check if cluster has angle fields
        has_angles = all(k in cluster for k in ("aoa", "aod", "zoa", "zod"))

        if is_los_cluster:
            # LOS specular: M=1, no expansion
            ray = {
                "delay": cluster["delay"],
                "power": cluster["power"],
                "source": cluster["source"],
                "rt_index": cluster.get("rt_index"),
                "cluster_id": idx,
                "ray_id": 0,
                "is_los_ray": True,
            }
            if has_angles:
                ray["aoa"] = cluster["aoa"]
                ray["aod"] = cluster["aod"]
                ray["zoa"] = cluster["zoa"]
                ray["zod"] = cluster["zod"]
            rays.append(ray)
        else:
            # Restored verbatim historical v4 delay and power operations.
            # This is NOT the full 38.901 K_B>1 branch: angle weights/bin fields absent.
            if delay_mode == "legacy_delay":
                tau_double_prime = rng.uniform(0, 2 * c_DS, size=M)
                tau_rel = np.sort(tau_double_prime - tau_double_prime.min())
                weights = np.exp(-tau_rel / c_DS)
                weights /= weights.sum()
            else:
                tau_rel = np.zeros(M)
                weights = np.full(M, 1.0 / M)
            P_ray = cluster["power"] / M

            for m in range(M):
                ray = {
                    "delay": cluster["delay"] + tau_rel[m],
                    "power": (
                        cluster["power"] * weights[m]
                        if delay_mode == "legacy_delay"
                        else P_ray
                    ),
                    "source": cluster["source"],
                    "rt_index": cluster.get("rt_index"),
                    "cluster_id": idx,
                    "ray_id": m,
                    "is_los_ray": False,
                }

                if has_angles:
                    a = alpha_m[m % len(alpha_m)]

                    # eq. 8.4-19: azimuth offsets (degrees → radians)
                    ray["aoa"] = cluster["aoa"] + np.deg2rad(c_ASA * a)
                    ray["aod"] = cluster["aod"] + np.deg2rad(c_ASD * a)

                    # eq. 8.4-20: ZOA offset
                    ray["zoa"] = cluster["zoa"] + np.deg2rad(c_ZSA * a)

                    # eq. 8.4-21: ZOD offset (uses c_ZSD_eff)
                    ray["zod"] = cluster["zod"] + np.deg2rad(c_ZSD_eff * a)

                    # ── Angle wrapping ──
                    # Azimuth → [-pi, pi]
                    ray["aoa"] = float(np.mod(ray["aoa"] + np.pi, 2 * np.pi) - np.pi)
                    ray["aod"] = float(np.mod(ray["aod"] + np.pi, 2 * np.pi) - np.pi)

                    # Zenith → [0, pi]
                    zoa = np.mod(ray["zoa"], 2 * np.pi)
                    if zoa > np.pi:
                        zoa = 2 * np.pi - zoa
                    ray["zoa"] = float(zoa)

                    zod = np.mod(ray["zod"], 2 * np.pi)
                    if zod > np.pi:
                        zod = 2 * np.pi - zod
                    ray["zod"] = float(zod)

                rays.append(ray)

    # Discretize the PDP.
    ray_pdp = _discretize_pdp_from_rays(rays, config["pdp_bin_size"])

    total_power_clusters = sum(c["power"] for c in merged_clusters)
    total_power_rays = sum(r["power"] for r in rays)

    return {
        "rays": rays,
        "ray_pdp_array": ray_pdp,
        "metadata": {
            "total_rays": len(rays),
            "M": M,
            "c_DS": c_DS,
            "total_power_clusters": total_power_clusters,
            "total_power_rays": total_power_rays,
        },
    }


def generate_xpr(rays, scenario_params, rng):
    """Sample per-ray XPR using table mean/std in dB; store linear power ratios in ray xpr. Modify and return rays."""
    mu = scenario_params.get("mu_XPR", 9.0)
    sigma = scenario_params.get("sigma_XPR", 3.0)

    for r in rays:
        # eq. 8.4-26: X ~ N(mu, sigma^2) in dB
        X_dB = rng.normal(mu, sigma)
        r["xpr"] = 10.0 ** (X_dB / 10.0)  # linear

    return rays


def prepare_hybrid_polarization(rays, rt_paths, scenario_params, pol):
    """Prepare statistical polarization for RT-derived rays. Probe co/cross-polar power ratios adjust the XPR mean while retaining random samples; native complex RT phase is not restored."""
    mu = scenario_params.get("mu_XPR", 9.0)
    for ray in rays:
        if ray.get("source") != "RT":
            continue
        ray["hybrid_cluster_polarization"] = True
        ray["xpr_provenance"] = "selected_table_assumption"
        parent = rt_paths[ray["rt_index"]]
        if not ray.get("is_los_ray", False) and pol.get("rt_xpr_from_rt", True):
            mean = parent.get("rt_xpr_db")
            if mean is None and "rt_jones" in parent:
                j = np.asarray(parent["rt_jones"], complex)
                co = float(abs(j[0, 0]) ** 2 + abs(j[1, 1]) ** 2)
                cross = float(abs(j[0, 1]) ** 2 + abs(j[1, 0]) ** 2)
                if co > 0:
                    mean = 10 * np.log10(co / max(cross, co * 1e-12))
            if mean is not None:
                if not np.isfinite(mean):
                    raise ValueError("RT XPR must be finite in dB")
                # Same standardized deviate, same RNG consumption and all phases.
                ray["xpr"] *= 10 ** ((float(mean) - mu) / 10)
                ray["xpr_provenance"] = "rt_mean_with_existing_cluster_deviate"


def _discretize_pdp_from_rays(rays, pdp_bin_size=1e-9):
    "Bin generated rays on a delay grid for PDP postprocessing only; do not modify channel rays."
    if not rays:
        return np.array([]), np.array([])

    max_delay = max(r["delay"] for r in rays)
    n_bins = int(np.ceil(max_delay / pdp_bin_size)) + 1
    power_array = np.zeros(n_bins)

    for r in rays:
        idx = int(np.round(r["delay"] / pdp_bin_size))
        idx = min(idx, n_bins - 1)
        power_array[idx] += r["power"]

    delay_axis = np.arange(n_bins) * pdp_bin_size
    return delay_axis, power_array


def reverse_paths(paths):
    "Reverse the same rays by swapping arrival/departure angles and transposing cross-polar phases; do not resample."
    output = deepcopy(paths)
    for item in output:
        for arrival, departure in (("aoa", "aod"), ("zoa", "zod")):
            item[arrival], item[departure] = item[departure], item[arrival]
        if "phase" in item:
            phase = item["phase"]
            phase["theta_phi"], phase["phi_theta"] = (
                phase["phi_theta"],
                phase["theta_phi"],
            )
    return output


def port_vectors(polarization):
    """Return two-component local polarization vectors for antenna ports. VH has two ports, separate from the geometric element count."""
    if not isinstance(polarization, str):
        a = np.asarray(polarization, complex)
        if a.shape == (2,):
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] != 2 or not np.all(np.isfinite(a)):
            raise ValueError("Custom antenna fields must have shape (ports,2)")
        return a
    pols = {
        "V": [[1, 0]],
        "H": [[0, 1]],
        "VH": [[1, 0], [0, 1]],
        "cross": np.array([[1, -1], [1, 1]]) / np.sqrt(2),
        "RHCP": np.array([[1, -1j]]) / np.sqrt(2),
        "LHCP": np.array([[1, 1j]]) / np.sqrt(2),
    }
    if polarization not in pols:
        raise ValueError(f"Unsupported polarization {polarization}")
    return np.asarray(pols[polarization], complex)


def spherical_basis(theta, phi):
    return np.array(
        [
            [np.cos(theta) * np.cos(phi), -np.sin(phi)],
            [np.cos(theta) * np.sin(phi), np.cos(phi)],
            [-np.sin(theta), 0.0],
        ]
    )


def antenna_fields(theta, phi, polarization="V", orientation_deg=(0, 0, 0)):
    """Rotate local antenna polarization into global path theta/phi bases. Path angles are radians; orientation_deg is degrees."""
    a, b, g = np.deg2rad(orientation_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
    rot = rz @ ry @ rx
    direction = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    local = rot.T @ direction
    tl, pl = np.arccos(np.clip(local[2], -1, 1)), np.arctan2(local[1], local[0])
    transform = spherical_basis(theta, phi).T @ rot @ spherical_basis(tl, pl)
    return port_vectors(polarization) @ transform.T


def ray_matrix(ray):
    """Build a 2x2 ray polarization matrix. Hybrid uses XPR and four initial phases; the native RT branch does not apply statistical polarization twice."""
    if ray.get("hybrid_cluster_polarization", False):
        statistical = dict(ray)
        statistical.pop("rt_jones", None)
        statistical.pop("rt_contains_diffuse", None)
        statistical.pop("hybrid_cluster_polarization", None)
        statistical["source"] = "RC"  # Matrix construction only; actual ray remains RT.
        return ray_matrix(statistical)
    if ray.get("rt_contains_diffuse", False):
        raise ValueError(
            "Sionna diffuse fields require upstream RTFieldReplay, not VH linear reconstruction"
        )
    if "rt_jones" in ray:
        matrix = np.asarray(ray["rt_jones"], complex)
        ref = float(ray["rt_reference_power"])
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)) or ref <= 0:
            raise ValueError("Invalid RT Jones matrix or reference power")
        return matrix / np.sqrt(ref)
    if ray.get("source") == "RT":
        raise ValueError(
            "Polarized RT requires a full 2x2 rt_jones matrix; scalar VV cannot be inverted"
        )
    ph = ray["phase"]
    if ray.get("is_los_ray", False):
        return np.exp(1j * ph["theta_theta"]) * np.diag([1.0, -1.0])
    xpr = float(ray["xpr"])
    if not np.isfinite(xpr) or xpr <= 0:
        raise ValueError("Scattered XPR must be positive and finite")
    return np.array(
        [
            [
                np.exp(1j * ph["theta_theta"]),
                np.exp(1j * ph["theta_phi"]) / np.sqrt(xpr),
            ],
            [np.exp(1j * ph["phi_theta"]) / np.sqrt(xpr), np.exp(1j * ph["phi_phi"])],
        ]
    )


def polarized_path_gains(rays, cfg, state, frequency_hz):
    """Apply Faraday before antenna projection: DL RX @ P @ F @ TX.T, UL RX @ F @ P_UL @ TX.T. Preserve matrix order; return complex [ray, rx_pol, tx_pol] gains."""
    direction = cfg.get("direction", "downlink")
    if direction not in ("downlink", "uplink"):
        raise ValueError("Polarization direction must be downlink or uplink")
    if state["mode"] != "off" and state.get("direction", "downlink") != direction:
        raise ValueError("Faraday state and antenna/path direction must agree")
    psi = float(angle_at_frequency(state, frequency_hz))
    fr = rotation(psi)
    # Ideal transmitter pre-rotation is an explicit reference, before the
    # ionosphere. A receiver-side inverse cannot commute through ground P.
    if state["compensation"] == "ideal_tx":
        fr = fr @ rotation(-psi)
    gains = []
    for ray in rays:
        pattern_gain = path_pattern_amplitude(ray, cfg)
        if "rt_port_response" in ray:
            if direction == "uplink":
                raise NotImplementedError(
                    "Native RT field replay is not validated for uplink Faraday"
                )
            # F and antenna projection were evaluated upstream by native RT,
            # before its polarization-dependent diffuse reflection gamma.
            gains.append(
                pattern_gain
                * np.sqrt(ray["power"] / ray["rt_reference_power"])
                * ray["rt_port_response"](frequency_hz)
            )
            continue
        angles = ray.get("rt_polarization_angles", ray)
        tx = antenna_fields(
            angles["zod"],
            angles["aod"],
            cfg.get("tx", "V"),
            cfg.get("tx_orientation_deg", [0, 0, 0]),
        )
        rx = antenna_fields(
            angles["zoa"],
            angles["aoa"],
            cfg.get("rx", "V"),
            cfg.get("rx_orientation_deg", [0, 0, 0]),
        )
        # The satellite is TX in DL and RX in UL. Thus F acts before the
        # urban matrix in DL and after it in UL (matrix action is right to left).
        p = ray_matrix(ray)
        field = rx @ p @ fr @ tx.T if direction == "downlink" else rx @ fr @ p @ tx.T
        gains.append(pattern_gain * np.sqrt(ray["power"]) * field)
    return np.asarray(gains)


def pattern_amplitude(theta, phi, config=None, orientation_deg=(0, 0, 0)):
    cfg = config or {}
    kind = cfg.get("kind", "iso")
    if kind == "iso":
        return 1.0
    k = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    if kind == "circular_aperture":
        if "boresight_gcs" not in cfg:
            raise ValueError("Circular aperture requires boresight_gcs or CLI tracking")
        b = np.asarray(cfg["boresight_gcs"], float)
        if b.shape != (3,) or not np.isfinite(b).all() or np.linalg.norm(b) == 0:
            raise ValueError("Invalid boresight")
        cos = float(np.clip(k @ (b / np.linalg.norm(b)), -1, 1))
        radius = float(cfg.get("radius_wavelengths", 10))
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("Aperture radius must be positive wavelengths")
        # A forward hemisphere is an explicit implementation convention.
        if cos < 0:
            return 0.0
        x = 2 * np.pi * radius * np.sqrt(max(0, 1 - cos * cos))
        return 1.0 if abs(x) < 1e-8 else float(abs(2 * j1(x) / x))
    if kind == "short_dipole":
        from scipy.spatial.transform import Rotation

        axis = Rotation.from_euler("ZYX", orientation_deg, degrees=True).apply(
            [0, 0, 1]
        )
        return float(np.sqrt(max(0, 1 - float(k @ axis) ** 2)))
    raise ValueError("Pattern must be iso, circular_aperture, or short_dipole")


def path_pattern_amplitude(ray, pol):
    angles = ray.get("rt_polarization_angles", ray)
    return pattern_amplitude(
        angles["zod"],
        angles["aod"],
        pol.get("tx_pattern"),
        pol.get("tx_orientation_deg", [0, 0, 0]),
    ) * pattern_amplitude(
        angles["zoa"],
        angles["aoa"],
        pol.get("rx_pattern"),
        pol.get("rx_orientation_deg", [0, 0, 0]),
    )


def build_hybrid(config, positions, rt, temporal_state=None):
    if not rt.anchors:
        raise ValueError(
            "Hybrid requires at least one nonzero RT anchor; zero paths is a valid RT-only result"
        )
    h = config["hybrid"]
    los = bool(rt.is_los.any())
    if (
        h["backend"] == "standard"
        and rt.metadata.get("link_los_exists", los)
        and (not los)
    ):
        raise NotImplementedError(
            "Physical LOS was filtered out; cannot relabel this link as NLOS for standard tables"
        )
    scenario = h["scenario"] + ("-LOS" if los else "-NLOS")
    options = {
        "M": h["rays_per_cluster"],
        "p0": h["rc_removal_p0"],
        "power_threshold_dB": config["rt"]["power_threshold_db"],
        "is_LOS_first_cluster": los,
    }
    fc = config["radio"]["frequency_hz"]
    arrays = {
        name: np.asarray(dev["array_offsets_m"], float)
        for name, dev in config["devices"].items()
    }
    faraday, context = faraday_inputs(config, positions)
    tx_name, rx_name = endpoint_names(config["direction"])
    pol = {
        "enabled": True,
        "direction": config["direction"],
        "rt_xpr_from_rt": h["rt_xpr_source"] == "probe_power_ratio",
    }
    for side, name in [("tx", tx_name), ("rx", rx_name)]:
        pol[side] = config["devices"][name]["polarization"]
        pol[side + "_orientation_deg"] = config["devices"][name]["orientation_deg"]
    options.update(
        table_profile=h["table_profile"],
        ray_delay_mode=h["ray_delay_mode"],
        polarization=pol,
        faraday=faraday,
    )
    generate = (
        generate_full_channel if temporal_state is None else temporal_state.evaluate
    )
    temporal_inputs = {} if temporal_state is None else {"positions": positions}
    raw = generate(
        rt.anchors,
        scenario,
        arrays[rx_name],
        arrays[tx_name],
        elev_deg=elevation_deg(positions),
        fc_GHz=fc / 1000000000.0,
        rng=np.random.default_rng(config["seed"]),
        config=options,
        tx_rx_distance_3D=float(
            np.linalg.norm(positions["satellite"] - positions["ue"])
        ),
        propagation_context=context,
        **temporal_inputs,
    )
    backend_name = __file__
    rays = raw["rays"]
    ids = {cid: i for i, cid in enumerate(sorted({r["cluster_id"] for r in rays}))}
    indices = [(ids[r["cluster_id"]], r["ray_id"]) for r in rays]
    coeff = np.stack([raw["channel"]["H"][:, :, i, j] for i, j in indices], axis=-1)
    delays = np.array([raw["channel"]["tau"][i, j] for i, j in indices])
    angles = np.array([[r[k] for k in ("aoa", "aod", "zoa", "zod")] for r in rays])
    sp = raw["scenario_params"]
    sampled_lsp = {
        key: sp.get(key) for key in ("DS", "K_R_dB", "ASA", "ASD", "ZSA", "ZSD")
    }
    if config["direction"] == "uplink":
        sampled_lsp["ASA"], sampled_lsp["ASD"] = (
            sampled_lsp["ASD"],
            sampled_lsp["ASA"],
        )
        sampled_lsp["ZSA"], sampled_lsp["ZSD"] = (
            sampled_lsp["ZSD"],
            sampled_lsp["ZSA"],
        )
    metadata = {
        "backend": backend_name,
        "scenario": scenario,
        "elevation_deg": elevation_deg(positions),
        "standards_status": (
            "corrected Clause-8 KB=1 subset; model-specific limitations remain"
            if h["backend"] == "standard"
            else "unsupported legacy"
        ),
        "model_revision": raw.get("config", {}).get("model_revision", "legacy"),
        "table_profile": h["table_profile"],
        "xpr_mu_db": float(sp["mu_XPR"]),
        "xpr_sigma_db": float(sp["sigma_XPR"]),
        "xpr_source": str(sp.get("source", "legacy project table")),
        "polarization_model": "statistical per-ray Jones before antenna projection",
        "hybrid_power": "RT VV path powers anchor RC power; no additional whole-link FSPL or shadow-fading multiplier",
        "native_rt_phase_retained": False,
        "sampled_lsp": sampled_lsp,
        "sampled_lsp_convention": "current link direction; DS in s, K_R_dB in dB, angular spreads in deg",
        "faraday": raw["channel"].get("metadata", {}).get("faraday"),
    }
    extra = {
        "xpr_linear": np.array([r["xpr"] for r in rays]),
        "cluster_ray_ids": np.array([[r["cluster_id"], r["ray_id"]] for r in rays]),
        "is_los_ray": np.array([r.get("is_los_ray", False) for r in rays]),
        "xpr_provenance": np.array(
            [r.get("xpr_provenance", "selected_table") for r in rays]
        ),
    }
    if temporal_state is not None:
        state = raw["temporal"]
        metadata.update(
            standards_status="experimental fixed-scatterer segment initialized by standard KB=1; not full 3GPP spatial consistency",
            initial_table_elevation_deg=state["initial_table_elevation_deg"],
            lsp_policy="fixed initial draw in this local segment; no redraw at 10-degree nodes",
        )
        extra.update(
            persistent_ray_ids=np.array(state["ray_ids"]),
            geometric_lengths_m=state["geometric_lengths_m"],
            interval_doppler_hz=state["interval_doppler_hz"],
        )
    return (
        coeff,
        delays,
        angles,
        np.array([r["source"] for r in rays]),
        metadata,
        extra,
    )

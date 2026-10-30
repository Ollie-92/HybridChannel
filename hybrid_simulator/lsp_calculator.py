"""TR 38.811 tables and LSP sampling for Hybrid; OpenNTN uses its own native tables."""

from pathlib import Path
import json
import numpy as np

TABLES = json.loads(
    (Path(__file__).parent / "tables/tr38811/dense_urban.json").read_text()
)
C_PHI_TABLE = {
    4: 0.779,
    5: 0.86,
    8: 1.018,
    10: 1.09,
    11: 1.123,
    12: 1.146,
    14: 1.19,
    15: 1.211,
    16: 1.226,
    19: 1.273,
    20: 1.289,
    25: 1.358,
}
C_THETA_TABLE = {
    8: 0.889,
    10: 0.957,
    11: 1.031,
    12: 1.104,
    15: 1.1088,
    19: 1.184,
    20: 1.178,
    25: 1.282,
}
ALPHA = np.array(
    [
        v
        for a in [
            0.0447,
            0.1413,
            0.2492,
            0.3715,
            0.5129,
            0.6797,
            0.8844,
            1.1481,
            1.5195,
            2.1551,
        ]
        for v in (a, -a)
    ]
)
C_PHI = {
    2: 0.501,
    3: 0.68,
    4: 0.779,
    5: 0.86,
    8: 1.018,
    10: 1.09,
    11: 1.123,
    12: 1.146,
    14: 1.19,
    15: 1.211,
    16: 1.226,
    19: 1.273,
    20: 1.289,
}
C_THETA = {
    2: 0.43,
    3: 0.594,
    4: 0.697,
    8: 0.889,
    10: 0.957,
    11: 1.031,
    12: 1.104,
    15: 1.1088,
    19: 1.184,
    20: 1.178,
}


def _lookup_nearest(table, N):
    "Choose the nearest table node, choosing the smaller node on ties; no interpolation."
    keys = np.array(sorted(table.keys()))
    idx = int(np.argmin(np.abs(keys - N)))
    return table[keys[idx]]


def lookup_C_phi(N, K_dB=None, is_LOS=False, table_profile="legacy"):
    "Look up azimuth scaling by cluster count and apply the LOS K-factor correction."
    table = C_PHI_TABLE
    if table_profile == "tr38811_corrected":
        table = C_PHI
    C_nlos = _lookup_nearest(table, N)
    if is_LOS and K_dB is not None:
        K = K_dB
        # TR 38.901 Table 7.5-2 footnote
        C_los = C_nlos * (1.1035 - 0.028 * K - 0.002 * K**2 + 0.0001 * K**3)
        return C_los
    return C_nlos


def lookup_C_theta(N, K_dB=None, is_LOS=False, table_profile="legacy"):
    "Look up zenith scaling by cluster count and apply the LOS K-factor correction."
    table = C_THETA_TABLE
    if table_profile == "tr38811_corrected":
        table = C_THETA
    C_nlos = _lookup_nearest(table, N)
    if is_LOS and K_dB is not None:
        K = K_dB
        # TR 38.901 Table 7.5-4 footnote
        C_los = C_nlos * (1.3086 + 0.0339 * K - 0.0077 * K**2 + 0.0002 * K**3)
        return C_los
    return C_nlos


def corrected_ntn_table(scenario_name, fc_GHz, elev_deg):
    "Arrange table values in the existing six-dimensional LSP draw order without extra SF draws or RNG changes."
    p = scenario_table(scenario_name, fc_GHz, elev_deg)
    names = ["K", "DS", "ASD", "ASA", "ZSD", "ZSA"]
    p["C"] = np.eye(6)
    active = p["active_lsp_names"]
    corr = p["active_lsp_correlation"]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a in active and b in active:
                p["C"][i, j] = corr[active.index(a), active.index(b)]
    # Zero-spread variables stay in the six-draw stream but are exact constants.
    p["scenario"] = scenario_name
    p["table_profile"] = "tr38811_corrected"
    return p


def sample_lsps(scenario_params, rng):
    """Draw correlated K, DS, ASD, ASA, ZSD and ZSA. Return DS in seconds, spreads in degrees and K in dB. Preserve Cholesky correction, angle caps and RNG order."""
    sp = scenario_params

    # ── Mean and std vectors (order: K, DS, ASD, ASA, ZSD, ZSA) ──
    mu = np.array(
        [
            sp["mu_K"],
            sp["mu_lgDS"],
            sp["mu_lgASD"],
            sp["mu_lgASA"],
            sp["mu_lgZSD"],
            sp["mu_lgZSA"],
        ]
    )
    sigma = np.array(
        [
            sp["sig_K"],
            sp["sig_lgDS"],
            sp["sig_lgASD"],
            sp["sig_lgASA"],
            sp["sig_lgZSD"],
            sp["sig_lgZSA"],
        ]
    )

    # ── Cholesky of cross-correlation matrix ──
    C = np.array(sp["C"], dtype=np.float64)
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
    lgDS = s[1]
    lgASD = s[2]
    lgASA = s[3]
    lgZSD = s[4]
    lgZSA = s[5]

    # ── Convert from log10 to linear ──
    DS_s = 10.0**lgDS  # seconds
    ASD_s = 10.0**lgASD  # degrees
    ASA_s = 10.0**lgASA  # degrees
    ZSD_s = 10.0**lgZSD  # degrees
    ZSA_s = 10.0**lgZSA  # degrees

    # ── Apply angle upper limits (TR 38.901 §7.5 Step 4 Note) ──
    ASA_s = min(ASA_s, 104.0)
    ASD_s = min(ASD_s, 104.0)
    ZSA_s = min(ZSA_s, 52.0)
    ZSD_s = min(ZSD_s, 52.0)

    # ── Determine LOS ──
    is_LOS = sp.get("sig_K", 0) > 0  # NLOS has sig_K = 0

    result = {
        "DS": float(DS_s),
        "K_R_dB": float(K_dB) if is_LOS else None,
        "ASA": float(ASA_s),
        "ASD": float(ASD_s),
        "ZSA": float(ZSA_s),
        "ZSD": float(ZSD_s),
        "r_tau": sp["r_tau"],
        "zeta_dB": sp["zeta_dB"],
        "N_cluster": sp["N_cluster"],
        "is_LOS": is_LOS,
        "c_DS": sp.get("c_DS", 0.0),
        "mu_offset_ZOD": sp.get("mu_offset_ZOD", 0.0),
        "mu_XPR": sp.get("mu_XPR", 9.0),
        "sigma_XPR": sp.get("sigma_XPR", 3.0),
    }

    if sp.get("table_profile") == "tr38811_corrected":
        for key in (
            "table_profile",
            "band",
            "table_elev_deg",
            "c_ASA",
            "c_ASD",
            "c_ZSA",
            "c_ZSD",
            "M",
        ):
            result[key] = sp[key]
    return result


def get_scenario_params(
    scenario_name, fc_GHz=None, elev_deg=None, table_profile="tr38811_corrected"
):
    """Read current NTN model tables; frequency is GHz and elevation is degrees."""
    if table_profile != "tr38811_corrected":
        raise NotImplementedError(
            "此入口保留目前 standard Hybrid；舊版模型仍在原專案。"
        )
    return corrected_ntn_table(scenario_name, fc_GHz, elev_deg)


def scenario_table(name, fc_GHz, elev_deg):
    "Select DenseUrban band, LOS state and elevation tables. Inputs are GHz/degrees; satellite ASD/ZSD are zero."
    if name not in ("NTN-DenseUrban-LOS", "NTN-DenseUrban-NLOS"):
        raise ValueError("v2 currently implements DenseUrban LOS/NLOS only")
    if fc_GHz is None or not np.isfinite(fc_GHz):
        raise ValueError("NTN v2 requires a frequency")
    band = "S" if 1.5 <= fc_GHz <= 4 else "Ka" if 17 <= fc_GHz <= 40 else None
    if band is None:
        raise ValueError(
            "Provide S-band (1.5..4 GHz) or Ka reference (17..40 GHz); other bands are unsupported"
        )
    if elev_deg is None or not np.isfinite(elev_deg) or not 10 <= elev_deg <= 90:
        raise ValueError("Elevation must be 10..90 degrees")
    # Explicit project policy: nearest 10-degree node, exact ties to the lower node.
    i = int(np.argmin(abs(np.asarray(TABLES["elevations"]) - elev_deg)))
    los = name.endswith("-LOS")
    t = TABLES["tables"][band + ("_LOS" if los else "_NLOS")]
    sp = dict(
        model_version="v2",
        is_LOS=los,
        band=band,
        fc_GHz=float(fc_GHz),
        elev_deg=float(elev_deg),
        table_elev_deg=TABLES["elevations"][i],
        source=t["source"],
        scenario=name,
    )
    for x in ["DS", "ASA", "ZSA"]:
        sp["mu_lg" + x] = t["mu_" + x][i]
        sp["sig_lg" + x] = t["sigma_" + x][i]
    sp.update(
        mu_lgASD=-np.inf,
        sig_lgASD=0.0,
        mu_lgZSD=-np.inf,
        sig_lgZSD=0.0,
        mu_offset_ZOD=0.0,
    )
    sp.update(
        mu_K=t["mu_K"][i] if los else 0.0,
        sig_K=t["sigma_K"][i] if los else 0.0,
        sigma_SF=t["sigma_SF"][i],
        CL_dB=t["CL_dB"][i],
    )
    for k in [
        "r_tau",
        "N_cluster",
        "M",
        "c_ASD",
        "c_ASA",
        "c_ZSA",
        "zeta_dB",
        "mu_XPR",
        "sigma_XPR",
    ]:
        sp[k] = t[k][i]
    sp["N_cluster"] = int(sp["N_cluster"])
    sp["M"] = int(sp["M"])
    sp["c_DS"] = t["c_DS_ns"][i] * 1e-9
    sp["c_ZSD"] = 0.0
    # Only nondegenerate random variables enter the covariance; departure spreads are exact constants.
    names = ["SF", "K", "DS", "ASA", "ZSA"] if los else ["SF", "DS", "ASA", "ZSA"]
    corr = np.eye(len(names))
    for pair, values in t["correlations"].items():
        a, b = pair.split(":")
        if a in names and b in names:
            corr[names.index(a), names.index(b)] = corr[
                names.index(b), names.index(a)
            ] = values[i]
    if np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError("Invalid standard active LSP correlation matrix")
    sp["active_lsp_names"] = names
    sp["active_lsp_correlation"] = corr
    return sp

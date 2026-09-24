# Source: YuChein/cluster_splitter.py; SHA-256 in docs/provenance.json.
# Modified for this local review on 2026-09-23: packaging and documentation.
# Original code rights: research team (confirmed by project owner); LICENSE pending.
# B08: extract Sionna RT scalar VV paths. CIR coefficients become squared magnitudes, so native complex RT phase is not preserved by the hybrid input interface.

import numpy as np


def extract_paths_from_rt(paths_all, paths_los,
                          rx_idx=0, tx_idx=0, tstep=0,
                          eps_tau=1e-9):
    """
    Extract scalar VV powers, absolute delays (s), and angles (rad).
    Classify paths within eps_tau seconds of LOS arrival as LOS.
    """
    a_all, tau_all = paths_all.cir(normalize_delays=False, out_type="numpy")
    a_los, tau_los = paths_los.cir(normalize_delays=False, out_type="numpy")

    a0  = a_all[rx_idx, 0, tx_idx, 0, :, tstep]
    tau = tau_all[rx_idx, tx_idx, :]
    valid = (tau >= 0) & (np.abs(a0) > 1e-15)

    if not np.any(valid):
        return None

    tau_v = tau[valid]
    p_v   = np.abs(a0[valid])**2

    # Angles
    phi_r   = np.asarray(paths_all.phi_r)
    phi_t   = np.asarray(paths_all.phi_t)
    theta_r = np.asarray(paths_all.theta_r)
    theta_t = np.asarray(paths_all.theta_t)

    def _get(arr):
        a = arr
        if a.ndim == 4:
            a = a[..., tstep]
        return a[rx_idx, tx_idx, :][valid]

    aoa_v = _get(phi_r)
    aod_v = _get(phi_t)
    zoa_v = _get(theta_r)
    zod_v = _get(theta_t)

    # LOS arrival time
    aL   = a_los[rx_idx, 0, tx_idx, 0, :, tstep]
    tauL = tau_los[rx_idx, tx_idx, :]
    vL   = (tauL >= 0) & (np.abs(aL) > 1e-15)
    tau_los0 = np.min(tauL[vL]) if np.any(vL) else np.min(tau_v)

    tau_ex   = tau_v - tau_los0
    is_los_v = tau_ex <= eps_tau

    return dict(
        delays=tau_v,
        powers=p_v,
        aoa=aoa_v,
        aod=aod_v,
        zoa=zoa_v,
        zod=zod_v,
        is_los=is_los_v,
        tau_los=tau_los0,
    )

"""Experimental local Hybrid evolution; static generation is in hybrid_model.py. Reuse equivalent fixed-scatterer geometry with the current initialization, power anchoring and Jones synthesis. Fixed LSPs/clusters are not full 3GPP spatial consistency."""

from copy import deepcopy, copy
import numpy as np
from scipy.constants import c
from . import hybrid_model as hybrid

from .hybrid_model import reverse_paths


def unit(azimuth, zenith):
    return np.array(
        [
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(zenith),
        ]
    )


def angles(vector):
    length = np.linalg.norm(vector)
    if length <= 0:
        raise ValueError("Endpoint coincides with an equivalent scatterer")
    return np.array(
        [
            np.arctan2(vector[1], vector[0]),
            np.arccos(np.clip(vector[2] / length, -1, 1)),
        ]
    )


def scatterer_from_delay(satellite, ue, length, aoa, zoa):
    """Solve |S-Q|+|Q-U|=L with Q=U+r*u, using r=(L-d)(L+d)/(2(L-a dot u)) to avoid cancellation. Scatterers are equivalent points, not city walls. Preserve initial angle offsets when sampled NTN AOD is incompatible with single-bounce geometry."""
    a = satellite - ue
    d = np.linalg.norm(a)
    u = unit(aoa, zoa)
    denominator = 2 * (length - np.dot(a, u))
    if length <= d or denominator <= 0:
        raise ValueError(
            "RC delay/arrival angle cannot define a positive single-bounce scatterer"
        )
    return ue + (length - d) * (length + d) / denominator * u


def _length(scatterer, positions):
    return np.linalg.norm(positions["satellite"] - scatterer) + np.linalg.norm(
        positions["ue"] - scatterer
    )


def _shift_angles(ray, delta):
    for key, value in zip(("aoa", "aod", "zoa", "zod"), delta):
        x = ray[key] + value
        ray[key] = float(
            (x + np.pi) % (2 * np.pi) - np.pi
            if key in ("aoa", "aod")
            else np.arccos(np.cos(x))
        )


class FixedScattererSegment:
    """One state per trajectory, without global or cross-user caches. Reject path birth/death, primitive changes and LOS/NLOS transitions. Initialize XPR, four phases, angle coupling and LSPs only once."""

    def __init__(self):
        self.initial = None
        self.rt_keys = None
        self.time_s = 0.0
        self.previous_time_s = None
        self.previous_lengths = None

    def prepare_rt(self, rt, positions, time_s):
        """Match native interaction/object/primitive sequences rather than sorted indices. Use float64 geometric increments with native t0 delays for static regression. Native vertices remain float32; this is not a float64 scene solve."""
        if not np.isfinite(time_s) or (
            self.previous_time_s is not None and time_s <= self.previous_time_s
        ):
            raise ValueError(
                "Time must strictly increase inside a fixed-scatterer segment"
            )
        if not rt.anchors:
            raise ValueError(
                "Fixed-scatterer segment needs RT anchors at every step; zero-path transition needs a birth/death model"
            )
        keys = rt.metadata.get("path_identity")
        lengths = rt.metadata.get("geometric_lengths_m")
        if keys is None or lengths is None:
            raise ValueError(
                "Temporal Hybrid requires native path identity and geometric lengths"
            )
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Ambiguous native path identity; refusing to merge distinct paths"
            )
        if self.rt_keys is None:
            self.rt_keys = list(keys)
            self.rt_lengths0 = np.asarray(lengths, float)
            self.rt_tau0 = rt.delays_s.copy()
            self.positions0 = {k: v.copy() for k, v in positions.items()}
        if set(keys) != set(self.rt_keys):
            raise ValueError(
                "RT topology changed (path birth/death or primitive crossing). Segment ended; no silent RC reset."
            )
        order = [keys.index(key) for key in self.rt_keys]
        # Align adapter output to fixed segment identities without modifying native Paths.
        rt = copy(rt)
        rt.anchors = [deepcopy(rt.anchors[i]) for i in order]
        rt.coefficients = rt.coefficients[:, :, order]
        rt.angles_rad = rt.angles_rad[order]
        rt.is_los = rt.is_los[order]
        rt.metadata = deepcopy(rt.metadata)
        self.rt_lengths = np.asarray(lengths, float)[order]
        rt.delays_s = self.rt_tau0 + (self.rt_lengths - self.rt_lengths0) / c
        for path, delay in zip(rt.anchors, rt.delays_s):
            path["tau_rt"] = float(delay)
        for key in ("path_identity", "geometric_lengths_m", "selected_raw_indices"):
            rt.metadata[key] = [rt.metadata[key][i] for i in order]
        if rt.vertices is not None:
            rt.vertices = rt.vertices[..., order, :]
        self.time_s = float(time_s)
        return rt

    def evaluate(
        self,
        rt_paths,
        scenario_name,
        rx_array,
        tx_array,
        *,
        elev_deg,
        fc_GHz,
        rng,
        config,
        tx_rx_distance_3D,
        propagation_context,
        positions,
    ):
        """Initialize with the standard model; subsequently update geometry, power and phase before existing Jones synthesis."""
        if self.initial is None:
            raw = hybrid.generate_full_channel(
                rt_paths,
                scenario_name,
                rx_array,
                tx_array,
                elev_deg=elev_deg,
                fc_GHz=fc_GHz,
                rng=rng,
                config=config,
                tx_rx_distance_3D=tx_rx_distance_3D,
                propagation_context=propagation_context,
            )
            self._initialize(raw, rt_paths, positions, scenario_name)
        else:
            if scenario_name != self.scenario:
                raise ValueError(
                    "LOS/NLOS changed; this segment does not implement state transitions"
                )
            raw = self._advance(
                rt_paths,
                positions,
                fc_GHz,
                config,
                rx_array,
                tx_array,
                propagation_context,
            )
        raw["config"] = dict(raw["config"], model_revision="fixed_scatterer_segment_v1")
        lengths = np.array(
            [
                (
                    self.rt_lengths[r["rt_index"]]
                    if r["source"] == "RT"
                    else _length(self.scatterers[i], positions)
                )
                for i, r in enumerate(self.rays0)
            ]
        )
        dt = (
            None if self.previous_time_s is None else self.time_s - self.previous_time_s
        )
        # Derive average Doppler from the same distance change for diagnostics; do not apply phase twice.
        doppler = (
            np.full(len(lengths), np.nan)
            if dt is None
            else -(lengths - self.previous_lengths) * fc_GHz * 1e9 / (c * dt)
        )
        raw["temporal"] = {
            "ray_ids": [
                f"{r['source']}:{r['cluster_id']}:{r['ray_id']}" for r in self.rays0
            ],
            "geometric_lengths_m": lengths,
            "interval_doppler_hz": doppler,
            "initial_table_elevation_deg": self.initial["scenario_params"][
                "table_elev_deg"
            ],
        }
        self.previous_lengths, self.previous_time_s = lengths.copy(), self.time_s
        return raw

    def _initialize(self, raw, paths, positions, scenario):
        self.initial = deepcopy(raw)
        self.scenario = scenario
        self.paths0 = deepcopy(paths)
        self.rays0 = deepcopy(raw["rays"])
        if raw["config"]["polarization"]["direction"] == "uplink":
            self.rays0 = reverse_paths(self.rays0)  # Store state in canonical downlink geometry.
        self.reference0 = min(p["tau_rt"] for p in paths)
        sat, ue = positions["satellite"], positions["ue"]
        self.scatterers = {}
        for i, ray in enumerate(self.rays0):
            if ray["source"] == "RC":
                self.scatterers[i] = scatterer_from_delay(
                    sat,
                    ue,
                    c * (self.reference0 + ray["delay"]),
                    ray["aoa"],
                    ray["zoa"],
                )
        sampling = raw["sampling_state"]
        self.cluster_scatterers = [
            scatterer_from_delay(
                sat,
                ue,
                c * (self.reference0 + delay),
                sampling["rc_angles"]["aoa"][i],
                sampling["rc_angles"]["zoa"][i],
            )
            for i, delay in enumerate(sampling["rc_delays_s"])
        ]

    def _advance(self, paths, positions, fc_GHz, config, rx_array, tx_array, context):
        sp = self.initial["scenario_params"]
        sampling = self.initial["sampling_state"]
        tau_rt = np.array([p["tau_rt"] for p in paths])
        reference = tau_rt.min()
        relative_rt = tau_rt - reference
        # Freeze LSPs; retain Step 6 virtual powers, K peak and RT power anchoring.
        delta_cluster = (
            np.array(
                [
                    _length(q, positions) - _length(q, self.positions0)
                    for q in self.cluster_scatterers
                ]
            )
            / c
        )
        reference_shift = self.reference0 - reference
        rc_delays = sampling["rc_delays_s"] + delta_cluster + reference_shift
        if np.any(rc_delays < 0):
            raise ValueError(
                "RC precedes the earliest RT reference; end this local segment"
            )
        ct = (
            0.7705
            - 0.0433 * sp["K_R_dB"]
            + 0.0002 * sp["K_R_dB"] ** 2
            + 0.000017 * sp["K_R_dB"] ** 3
            if sp["is_LOS"]
            else 1.0
        )
        # Supplying Z avoids RNG use here and preserves the initial per-cluster shadowing.
        power_delays = (
            sampling["rc_power_delays_s"] + (delta_cluster + reference_shift) * ct
        )
        vrc, vrt = hybrid.compute_virtual_powers(
            power_delays,
            relative_rt,
            sp,
            None,
            Z_RC=sampling["zrc_db"],
            Z_RT=sampling["zrt_db"],
        )
        prc, prt = hybrid.normalize_virtual_powers(vrc, vrt, sp)
        powers_rc = hybrid.anchor_rc_power(
            prc, prt, np.array([p["power_rt"] for p in paths])
        )
        clusters = self.initial["hybrid"]["clusters"]
        rays = deepcopy(self.rays0)
        for i, ray in enumerate(rays):
            cluster = clusters[ray["cluster_id"]]
            if ray["source"] == "RT":
                j = ray["rt_index"]
                delta_length = self.rt_lengths[j] - self.rt_lengths0[j]
                ray["delay"] = float(relative_rt[j])
                ray["power"] *= paths[j]["power_rt"] / self.paths0[j]["power_rt"]
                delta = np.array(
                    [
                        paths[j][k] - self.paths0[j][k]
                        for k in ("aoa", "aod", "zoa", "zod")
                    ]
                )
            else:
                q = self.scatterers[i]
                delta_length = _length(q, positions) - _length(q, self.positions0)
                ray["delay"] += self.reference0 - reference + delta_length / c
                ray["power"] *= powers_rc[cluster["rc_index"]] / cluster["power"]
                arrival = angles(q - positions["ue"]) - angles(
                    q - self.positions0["ue"]
                )
                departure = angles(q - positions["satellite"]) - angles(
                    q - self.positions0["satellite"]
                )
                delta = np.array([arrival[0], departure[0], arrival[1], departure[1]])
            _shift_angles(ray, delta)
            if ray["delay"] < 0:
                raise ValueError(
                    "Evolving ray precedes earliest RT; local segment is no longer valid"
                )
            phase_delta = -2 * np.pi * fc_GHz * 1e9 * delta_length / c
            ray["phase"] = {k: float(v + phase_delta) for k, v in ray["phase"].items()}
        # Keep the initial -25 dB selected set to avoid threshold flicker; this is not an independent snapshot.
        if config["polarization"]["direction"] == "uplink":
            rays = reverse_paths(rays)
        channel = hybrid.assemble_coefficients(
            rays,
            rx_array,
            tx_array,
            fc_GHz * 1e9,
            config["polarization"],
            config["faraday"],
            context,
        )
        return dict(rays=rays, channel=channel, scenario_params=sp, config=config)

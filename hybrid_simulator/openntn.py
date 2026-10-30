"""Thin native OpenNTN adapter; do not rewrite LSP, ray, path-loss or coefficient formulas. Based on third_party/openntn/example/channel_response.ipynb cells 6/8/10/15/27. See UPSTREAM.json for revision, hashes and import compatibility changes."""

import json
import sys
from pathlib import Path
from importlib.metadata import version
import numpy as np
from .config import ROOT
from .topology import elevation_deg, endpoint_names
from .result import ChannelResult, offsets


def use_vendored_openntn():
    """Use only the vendored model; require kernel restart if a different version is already imported."""
    vendor = ROOT / "third_party/openntn"
    loaded = sys.modules.get("OpenNTN")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(
        vendor.resolve()
    ):
        raise RuntimeError(
            "Another OpenNTN is already imported. Restart the kernel to use third_party/openntn."
        )
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))


def generate_openntn(config, positions):
    """One satellite/UE fixed-topology time block; native direction handles uplink/downlink."""
    upstream = use_vendored_openntn()
    import tensorflow as tf
    import OpenNTN
    from OpenNTN import AntennaArray, DenseUrban, Urban, SubUrban
    from sionna.phy import config as sionna_config
    from sionna.phy.channel import cir_to_ofdm_channel
    from sionna.phy.constants import SPEED_OF_LIGHT

    options = config["openntn"]
    frequency = config["radio"]["frequency_hz"]
    # Seed native OpenNTN/Sionna RNGs; each call creates an independent realization.
    sionna_config.seed = config["seed"]
    # This upstream version uses both config.tf_rng and tf.random; seed both for reproducibility.
    tf.random.set_seed(config["seed"])
    arrays = {}
    port_maps = {}
    array_settings = {}
    for name in ("ue", "satellite"):
        device = config["devices"][name]
        antenna = device.get("antenna", {
            "num_rows": 1, "num_cols": 1,
            "vertical_spacing": 0.5, "horizontal_spacing": 0.5,
        })
        # Official channel_response.ipynb cell 6: native AntennaArray.
        # Spacings are wavelengths; orientations are passed to set_topology below.
        arrays[name] = AntennaArray(
            num_rows=antenna["num_rows"],
            num_cols=antenna["num_cols"],
            vertical_spacing=antenna["vertical_spacing"],
            horizontal_spacing=antenna["horizontal_spacing"],
            polarization="dual" if device["polarization"] == "VH" else "single",
            polarization_type=device["polarization"],
            antenna_pattern="omni",
            carrier_frequency=frequency,
            precision="single",
        )
        count = antenna["num_rows"] * antenna["num_cols"]
        polarizations = 2 if device["polarization"] == "VH" else 1
        # Native ports are polarization-major. Public results are element-major.
        port_maps[name] = np.arange(count * polarizations).reshape(polarizations, count).T.reshape(-1)
        array_settings[name] = {
            **antenna, "native_pattern": "omni",
            "native_local_port_positions_m": arrays[name].ant_pos.numpy().tolist(),
            "output_to_native_port": port_maps[name].tolist(),
        }

    # Use the official DenseUrban call and the native classes for the other scenarios.
    scenario_class = {"dense_urban": DenseUrban, "urban": Urban, "suburban": SubUrban}[
        options["scenario"]
    ]
    channel_model = scenario_class(
        carrier_frequency=frequency,
        ut_array=arrays["ue"],
        bs_array=arrays["satellite"],
        direction=config["direction"],
        elevation_angle=elevation_deg(positions),
        enable_pathloss=options["enable_pathloss"],
        enable_shadow_fading=options["enable_shadow_fading"],
        doppler_enabled=options["doppler_enabled"],
        precision="single",
    )

    # Use configured endpoints, orientations and UE velocity instead of random example topology.
    channel_model.set_topology(
        ut_loc=tf.constant([[positions["ue"]]], dtype=tf.float32),
        bs_loc=tf.constant([[positions["satellite"]]], dtype=tf.float32),
        ut_orientations=tf.constant(
            [[np.deg2rad(config["devices"]["ue"]["orientation_deg"])]], dtype=tf.float32
        ),
        bs_orientations=tf.constant(
            [[np.deg2rad(config["devices"]["satellite"]["orientation_deg"])]],
            dtype=tf.float32,
        ),
        ut_velocities=tf.constant([[options["ue_velocity_m_s"]]], dtype=tf.float32),
        in_state=tf.constant([[False]]),
        los=options["los"],
    )
    a, tau = channel_model(
        num_time_samples=options["num_time_samples"],
        sampling_frequency=options["sampling_frequency_hz"],
    )
    frequencies = offsets(config)
    # Use native cir_to_ofdm_channel without normalization, retaining absolute gain.
    h_frequency = cir_to_ofdm_channel(
        tf.constant(frequencies, dtype=tf.float32),
        a,
        tau,
        normalize=False,
    )
    a_native, tau_native, h_native = a.numpy(), tau.numpy(), h_frequency.numpy()
    if not all(np.isfinite(x).all() for x in (a_native, tau_native, h_native)):
        raise ValueError(
            "Native OpenNTN produced nonfinite CIR/CFR; no replacement or clipping was applied."
        )

    # Reshape only; aggregated taps do not have individual geometric angles or LOS flags.
    tx_name, rx_name = endpoint_names(config["direction"])
    rx_order, tx_order = port_maps[rx_name], port_maps[tx_name]
    coefficients_time = a_native[0, 0, :, 0, :, :, :][rx_order][:, tx_order]
    cfr_time = h_native[0, 0, :, 0, :, :, :][rx_order][:, tx_order]
    delays = tau_native[0, 0, 0, :]
    count = len(delays)
    # Read upstream distance/LOS diagnostics without modifying private state; no public accessor exists.
    native_distance = float(channel_model._scenario.distance_3d.numpy().ravel()[0])
    native_los = bool(channel_model._scenario.los.numpy().ravel()[0])
    metadata = dict(
        mode="3gpp",
        backend="vendored OpenNTN native system-level channel",
        direction=config["direction"],
        seed=config["seed"],
        endpoints=endpoint_names(config["direction"]),
        positions_scene_m={k: v.tolist() for k, v in positions.items()},
        frequency_hz=frequency,
        polarization_ports={k: v["polarization"] for k, v in config["devices"].items()},
        port_order="element-major then polarization; native arrays remain polarization-major",
        openntn=dict(
            repository=upstream["repository"],
            commit=upstream["commit"],
            import_path=OpenNTN.__file__,
            options=options,
            arrays=array_settings,
            sionna_version=version("sionna"),
            tensorflow_version=tf.__version__,
            native_los_state=native_los,
            elevation_deg=elevation_deg(positions),
            native_slant_range_m=native_distance,
            endpoint_euclidean_distance_m=float(
                np.linalg.norm(positions["satellite"] - positions["ue"])
            ),
            diagnostics_access="read-only _scenario.distance_3d / .los; pinned upstream revision",
        ),
        angles_available=False,
        los_flags_available=False,
        output_granularity="native aggregated cluster/subcluster taps; not RT rays",
        angle_columns=["aoa", "aod", "zoa", "zod"],
        angle_convention="not available per aggregated tap; angles_rad contains NaN",
        timing_reference="native excess delays; reference = OpenNTN spherical-Earth slant range / c",
        dynamic_scope="native fixed-topology Doppler block; independent realization per call; no orbit evolution or spatial consistency",
        propagation_model="native path loss/shadowing/atmospheric terms only; no additional project path loss or ionosphere",
        applied_scalar_effects={"applied": []},
        ionosphere=config["ionosphere"],
        shapes=dict(
            openntn_a=list(a_native.shape),
            openntn_tau=list(tau_native.shape),
            coefficients_time=list(coefficients_time.shape),
            cfr_time=list(cfr_time.shape),
        ),
    )
    extra = dict(
        openntn_a=a_native,
        openntn_tau=tau_native,
        openntn_cfr=h_native,
        output_to_native_rx_port=rx_order,
        output_to_native_tx_port=tx_order,
        coefficients_time=coefficients_time,
        cfr_time=cfr_time,
        time_s=np.arange(options["num_time_samples"])
        / options["sampling_frequency_hz"],
    )
    return ChannelResult(
        coefficients_time[..., 0],
        delays,
        native_distance / float(SPEED_OF_LIGHT),
        np.full((count, 4), np.nan),
        np.full(count, "OpenNTN"),
        frequencies,
        cfr_time[..., 0, :],
        metadata,
        extra,
        runtime_context={"openntn_model": channel_model},
    )

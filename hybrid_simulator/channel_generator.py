"""Single snapshot: configuration, geometry, native RT, optional Hybrid, effects, and CIR/CFR."""

import time
import numpy as np
from .config import validate_config
from .topology import scene_positions, endpoint_names
from .rt import trace_paths
from .rt_postprocess import export_paths, load_rt_cache
from .hybrid_model import build_hybrid
from .ionosphere import resolve_correction, apply_scalar_effects
from .result import ChannelResult, offsets, frequency_response, checked_output


def generate_channel(
    config,
    *,
    run_directory=None,
    rt_session=None,
    temporal_state=None,
    time_s=0.0,
    rt_data=None
):
    """Generate one snapshot without updating positions, creating time loops, or writing outside the workspace."""
    c = validate_config(config)
    if run_directory is not None:
        run_directory = checked_output(run_directory)
    if c.get("trajectory") is not None:
        raise ValueError(
            "Use generate_trajectory for a trajectory config; single snapshots require trajectory=null"
        )
    if temporal_state is not None:
        from .orbit import validate_temporal_mode

        validate_temporal_mode(c)
    positions = scene_positions(c)
    # Route 3GPP-only directly to OpenNTN without creating an RT scene.
    if c["mode"] == "3gpp":
        if temporal_state is not None or rt_session is not None:
            raise ValueError(
                "OpenNTN uses a native fixed-topology block, not an RT/Hybrid temporal state"
            )
        from .openntn import generate_openntn

        return generate_openntn(c, positions)
    correction = resolve_correction(c, positions, run_directory)
    started = time.perf_counter()
    # RT and Hybrid share native RT; rt_data accepts externally computed results.
    if rt_data is not None:
        rt = rt_data
    elif c["rt"]["source"] == "cache":
        # Cache loading is for historical regression; the notebook traces each direction afresh.
        rt = load_rt_cache(c, positions)
    else:
        scene_reused = bool(rt_session and "scene" in rt_session)
        scene, paths, a, tau = trace_paths(c, positions, rt_session)
        # Custom postprocessing is separate from native RT and does not call PathSolver.
        rt = export_paths(
            paths, a, tau, scene, c, positions,
            solve_started=started, reused=scene_reused,
        )
    if rt.metadata.get("trace_direction") != c["direction"]:
        raise ValueError(
            "RT 搜尋方向與本次通道方向不符；請分別執行 DL / UL PathSolver。"
        )
    if rt.metadata.get("rt_source") == "native":
        tx_name, rx_name = endpoint_names(c["direction"])
        for key, name in [
            ("native_tx_position_m", tx_name),
            ("native_rx_position_m", rx_name),
        ]:
            if not np.array_equal(
                np.asarray(rt.metadata[key], np.float32),
                np.asarray(positions[name], np.float32),
            ):
                raise ValueError("RT 結果的位置已過期，請按目前 TX/RX 重新搜尋。")
    if temporal_state is not None:
        if c["mode"] != "hybrid" or c["hybrid"]["backend"] != "standard":
            raise ValueError("Temporal state requires the standard Hybrid backend")
        rt = temporal_state.prepare_rt(rt, positions, time_s)
    rt_seconds = time.perf_counter() - started
    reference = float(rt.delays_s.min()) if len(rt.delays_s) else 0.0
    if c["mode"] == "hybrid" and not rt.anchors:
        # TR 38.901 section 8.4 Step 3: no RT anchors means a zero channel, without RC generation.
        tx, rx = endpoint_names(c["direction"])

        def ports(name):
            d = c["devices"][name]
            return len(d["array_offsets_m"]) * (2 if d["polarization"] == "VH" else 1)

        h = np.zeros((ports(rx), ports(tx), 0), complex)
        tau = np.zeros(0)
        angles = np.zeros((0, 4))
        sources = np.array([], dtype="U2")
        metadata = {
            "backend": "hybrid skipped: no RT anchors",
            "standards_status": "38.901 Step 3 zero-channel rule",
        }
        extra = {"is_los_ray": np.zeros(0, bool)}
    elif c["mode"] == "hybrid":
        # Only Hybrid adds statistical clusters and channel synthesis.
        h, tau, angles, sources, metadata, extra = build_hybrid(
            c, positions, rt, temporal_state
        )
    else:
        h, tau, angles, sources = (
            rt.coefficients,
            rt.delays_s - reference,
            rt.angles_rad,
            np.full(len(rt.delays_s), "RT"),
        )
        metadata = {
            "backend": "native RT fields",
            "native_rt_phase_retained": True,
            "polarization_model": "native material and antenna projection",
        }
        extra = {"is_los_ray": rt.is_los}
        if rt.vertices is not None:
            extra["native_vertices_m"] = rt.vertices
    # Group delay and carrier phase are distinct; physical dispersion should enable both.
    h, tau, effects = apply_scalar_effects(h, tau, c, correction)
    # Keep the pre-correction RT reference for analysis without retracing.
    # The Hybrid RT reference is the unrotated VV probe, not the user's full RT-only array channel.
    extra.update(
        rt_reference_coefficients=rt.coefficients,
        rt_reference_delays_s=rt.delays_s - reference,
        rt_reference_angles_rad=rt.angles_rad,
        rt_reference_is_los=rt.is_los,
    )
    freq = offsets(c)
    metadata.update(
        mode=c["mode"],
        direction=c["direction"],
        seed=c["seed"],
        endpoints=endpoint_names(c["direction"]),
        positions_scene_m={k: v.tolist() for k, v in positions.items()},
        frequency_hz=c["radio"]["frequency_hz"],
        rt=rt.metadata,
        rt_seconds=rt_seconds,
        post_rt_seconds=time.perf_counter() - started - rt_seconds,
        ionosphere=c["ionosphere"],
        applied_scalar_effects=effects,
        timing_reference="earliest retained geometric RT delay; common medium excess not renormalized away",
        angle_columns=["aoa", "aod", "zoa", "zod"],
        angle_convention="outward looking at each endpoint, radians",
        polarization_ports={k: v["polarization"] for k, v in c["devices"].items()},
        port_order="element-major then polarization",
        frequency_approximation="fields, Faraday and array steering at fc; first-order delay slope across bandwidth",
        dynamic_scope="single snapshot; no evolution of random clusters or Doppler time series",
    )
    if temporal_state is not None:
        metadata["dynamic_scope"] = (
            "fixed-scatterer local segment; carrier phase advanced by geometric distance change, no extra Doppler multiplier"
        )
        metadata["timing_reference"] = (
            "native t0 absolute delay + float64 geometric distance increments; no per-step carrier-phase reset"
        )
    return ChannelResult(
        h,
        tau,
        reference,
        angles,
        sources,
        freq,
        frequency_response(h, tau, freq),
        metadata,
        extra,
        runtime_context=rt.runtime_context,
    )


def generate_hybrid_channel(config, *, run_directory=None):
    """Explicit Hybrid alias; reject RT configurations instead of silently changing the mode."""
    if config["mode"] != "hybrid":
        raise ValueError("generate_hybrid_channel requires mode=hybrid")
    return generate_channel(config, run_directory=run_directory)


class HybridChannel:
    """One link generator; call once for fixed positions or reuse for a short temporal segment."""

    def __init__(self, config, *, persistent=False, device_names=None):
        from copy import deepcopy
        from .config import configure_runtime

        configure_runtime()
        self.config = validate_config(deepcopy(config))
        self.rt_session = None if self.config["mode"] == "3gpp" else {}
        # Names such as sat1/ue1 label the scene; the model uses satellite/ue roles.
        if device_names is not None:
            if (set(device_names) != {"satellite", "ue"}
                    or any(not isinstance(n, str) or not n for n in device_names.values())
                    or len(set(device_names.values())) != 2):
                raise ValueError("device_names 需指定不同的 satellite 與 ue 名稱。")
            if self.rt_session is not None:
                self.rt_session["device_names"] = dict(device_names)
        self.temporal_state = None
        if persistent:
            from .orbit import validate_temporal_mode
            from .temporal import FixedScattererSegment

            validate_temporal_mode(self.config)
            self.temporal_state = FixedScattererSegment()

    def __call__(
        self,
        *,
        satellite_position_m=None,
        ue_position_m=None,
        time_s=0.0,
        run_directory=None,
        rt_data=None
    ):
        """Accept ENU positions in meters and relative time in seconds; return ChannelResult."""
        from copy import deepcopy

        config = deepcopy(self.config)
        if satellite_position_m is not None or ue_position_m is not None:
            if (
                config["coordinates"]["frame"] != "scene_enu"
                or config["coordinates"]["position_unit"] != "m"
            ):
                raise ValueError("逐時點位置必須使用 scene_enu / m。")
            for name, value in [
                ("satellite", satellite_position_m),
                ("ue", ue_position_m),
            ]:
                if value is not None:
                    config["devices"][name]["position"] = np.asarray(
                        value, float
                    ).tolist()
        return generate_channel(
            config,
            run_directory=run_directory,
            rt_session=self.rt_session,
            temporal_state=self.temporal_state,
            time_s=time_s,
            rt_data=rt_data,
        )

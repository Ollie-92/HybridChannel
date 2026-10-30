"""Custom RT postprocessing: native CIR to filtered, ordered Hybrid inputs. No PathSolver call occurs here. Cache loading is only for historical regression; the notebook uses freshly traced paths."""
from dataclasses import dataclass, field
import hashlib
import json
import logging
import time
import numpy as np
from .config import input_path
from .topology import endpoint_names


@dataclass
class RtData:
    coefficients: np.ndarray
    delays_s: np.ndarray
    angles_rad: np.ndarray
    is_los: np.ndarray
    anchors: list
    metadata: dict
    vertices: np.ndarray | None = None
    runtime_context: dict = field(default_factory=dict, repr=False)


def _assemble(a, tau, angles, interactions, config, provenance, vertices=None):
    """Remove invalid padding while retaining the existing VV probe and RC power-anchor definitions."""
    a = np.asarray(a)
    tau = np.asarray(tau, float)
    powers = (
        np.abs(a[0, 0]) ** 2
        if config["mode"] == "hybrid"
        else np.sum(np.abs(a) ** 2, axis=(0, 1))
    )
    valid = np.isfinite(tau) & (tau >= 0) & np.isfinite(powers) & (powers > 0)
    all_los = np.all(interactions == 0, axis=0)
    link_los = bool(np.any(valid & all_los))
    if valid.any():
        valid &= powers >= powers[valid].max() * 10 ** (
            config["rt"]["power_threshold_db"] / 10
        )
    idx = np.flatnonzero(valid)
    idx = idx[np.argsort(tau[idx], kind="stable")]
    # Zero interactions means no surface interaction; do not infer LOS from minimum delay proximity.
    los = np.all(interactions[:, idx] == 0, axis=0)
    anchors = [
        dict(
            tau_rt=float(tau[i]),
            power_rt=float(powers[i]),
            aoa=float(angles[i, 0]),
            aod=float(angles[i, 1]),
            zoa=float(angles[i, 2]),
            zod=float(angles[i, 3]),
            rt_jones=a[:, :, i].copy(),
            rt_contains_diffuse=bool(np.any(interactions[:, i] & 2)),
            is_los=bool(los[j]),
        )
        for j, i in enumerate(idx)
    ]
    return RtData(
        a[:, :, idx],
        tau[idx],
        angles[idx],
        los,
        anchors,
        dict(
            provenance,
            selected_raw_indices=idx.tolist(),
            los_count=int(los.sum()),
            link_los_exists=link_los,
        ),
        None if vertices is None else vertices[..., idx, :],
    )


def export_paths(
    paths,
    a,
    tau,
    scene,
    config,
    positions,
    *,
    scene_seconds=0.0,
    solve_started=None,
    reused=False,
):
    """Adapt, threshold and sort native CIR without retracing. Inputs come from paths.cir(normalize_delays=False, out_type=numpy). Hybrid requires an unrotated single-element iso VH probe; final antennas are applied during Hybrid synthesis."""
    import importlib.metadata
    import mitsuba as mi
    import sionna.rt as rt

    scene_path = config["rt"]["scene_xml"]
    options = config["rt"]["solver"]
    scattering = config["rt"].get("scattering_coefficient")
    tx_name, rx_name = endpoint_names(config["direction"])
    # Check native endpoints to reject results from a different direction or position.
    expected_tx = np.asarray(positions[tx_name], dtype=np.float32)
    expected_rx = np.asarray(positions[rx_name], dtype=np.float32)
    actual_tx = np.asarray(paths.sources).reshape(3, -1)[:, 0]
    actual_rx = np.asarray(paths.targets).reshape(3, -1)[:, 0]
    if not (
        np.array_equal(actual_tx, expected_tx)
        and np.array_equal(actual_rx, expected_rx)
    ):
        raise ValueError("原生 Paths 端點與本次 direction / 位置不符。")
    native_a = np.asarray(a)[0, :, 0, :, :, 0]
    native_tau = np.asarray(tau).reshape(-1)
    # The statistical core uses satellite/UE fields; adapt the freshly traced UL result.
    # The public API reverses CIR port order without retracing or mutating native Paths.
    canonical_ul = config["mode"] == "hybrid" and config["direction"] == "uplink"
    if canonical_ul:
        a, tau = paths.cir(
            normalize_delays=False, reverse_direction=True, out_type="numpy"
        )
    a = np.asarray(a)[0, :, 0, :, :, 0]
    if config["mode"] == "rt":
        # Convert Sionna pattern-major ports to element-major order, matching Hybrid.
        def port_indices(name):
            dev = config["devices"][name]
            ne = len(dev["array_offsets_m"])
            npol = 2 if dev["polarization"] == "VH" else 1
            return np.arange(ne * npol).reshape(npol, ne).T.reshape(-1)

        a = a[port_indices(rx_name)][:, port_indices(tx_name)]
    tau = np.asarray(tau).reshape(-1)
    angles = np.stack(
        [
            np.asarray(getattr(paths, k)).reshape(-1)
            for k in ("phi_r", "phi_t", "theta_r", "theta_t")
        ],
        axis=1,
    )
    if canonical_ul:
        angles = angles[:, [1, 0, 3, 2]]
    interaction = np.asarray(paths.interactions)
    interaction = interaction.reshape(interaction.shape[0], len(tau))
    vertices = np.asarray(paths.vertices)
    solver_seconds = (
        0.0 if solve_started is None else time.perf_counter() - solve_started
    )
    logging.info("Native paths and NumPy export completed in %.3f s", solver_seconds)
    provenance = {
        "rt_source": "native",
        "fresh_rt_search": True,
        "trace_direction": config["direction"],
        "native_tx_endpoint": tx_name,
        "native_rx_endpoint": rx_name,
        "native_tx_position_m": actual_tx.tolist(),
        "native_rx_position_m": actual_rx.tolist(),
        "native_valid_path_count": (
            int(
                np.sum(
                    np.any(np.asarray(paths.valid).reshape(-1, len(native_tau)), axis=0)
                )
            )
            if len(native_tau)
            else 0
        ),
        "coefficient_angle_basis": (
            "satellite_to_ue" if config["mode"] == "hybrid" else config["direction"]
        ),
        "independent_direction_search": True,
        "sionna_version": importlib.metadata.version("sionna-rt"),
        "sionna_import": rt.__file__,
        "variant": mi.variant(),
        "scene_xml": scene_path,
        "solver": options,
        "seed": config["seed"],
        "scattering_coefficient": scattering,
        "scene_load_seconds": scene_seconds,
        "scene_reused": reused,
        "solver_and_export_seconds": solver_seconds,
        "hybrid_probe": (
            "unrotated 1-element iso VH; user array applied only in Hybrid assembler"
            if config["mode"] == "hybrid"
            else None
        ),
    }
    result = _assemble(a, tau, angles, interaction, config, provenance, vertices)
    if config["mode"] == "hybrid":
        # The probe has one geometric element per endpoint; surface sequences identify paths over a segment.
        objects = np.asarray(paths.objects).reshape(interaction.shape)
        primitives = np.asarray(paths.primitives).reshape(interaction.shape)
        points = vertices.reshape(interaction.shape + (3,)).astype(np.float64)
        keys, lengths = [], []
        for i in result.metadata["selected_raw_indices"]:
            active = interaction[:, i] != 0
            sequence = [
                [int(a), int(b), int(d)]
                for a, b, d in zip(
                    interaction[active, i], objects[active, i], primitives[active, i]
                )
            ]
            keys.append(json.dumps(sequence, separators=(",", ":")))
            # Compute length in the actual tracing direction: UE to satellite for UL.
            polyline = np.vstack(
                (positions[tx_name], points[active, i], positions[rx_name])
            )
            lengths.append(
                float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())
            )
        result.metadata.update(
            path_identity=keys,
            geometric_lengths_m=lengths,
            geometry_precision="float64 distances from configured endpoints and native float32 vertices",
        )
    # Reuse native scene and paths for plotting without calling PathSolver again.
    result.metadata["retained_rt_path_count"] = len(result.delays_s)
    result.runtime_context.update(
        scene=scene, paths=paths, native_cir_a=native_a, native_cir_tau=native_tau
    )
    return result


def load_rt_cache(config, positions):
    """Load cache without recomputing the scene; reject mismatched endpoints, frequency or antenna probe."""
    if config["direction"] != "downlink":
        raise ValueError(
            "舊快取只有下行搜尋；上行請使用 source=native。讀取完成通道請用 ChannelResult.load。"
        )
    path = input_path(config["rt"]["cache_path"])
    with np.load(path, allow_pickle=False) as d:
        spec = json.loads(str(d["spec_json"]))
        for name, key in [("satellite", "satellite_position_m"), ("ue", "position_m")]:
            if not np.allclose(positions[name], spec[key], rtol=0, atol=1e-6):
                raise ValueError(
                    f"RT cache {name} endpoint differs; use native RT for changed positions"
                )
        if not np.isclose(
            config["radio"]["frequency_hz"],
            spec["config"]["carrier_Hz"],
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError(
                "RT cache frequency differs; no rescaling of cached fields is allowed"
            )
        if config["mode"] == "rt":
            for dev in config["devices"].values():
                if (
                    np.asarray(dev["array_offsets_m"]).shape != (1, 3)
                    or np.any(dev["array_offsets_m"])
                    or np.any(dev["orientation_deg"])
                ):
                    raise ValueError(
                        "Cached RT only contains one zero-offset unrotated VH probe per endpoint"
                    )
        a = d["raw_cir_a"][0, :, 0, :, :, 0].copy()
        tau = d["raw_cir_tau"].reshape(-1).copy()
        angles = np.stack(
            [
                d[k].reshape(-1)
                for k in ("raw_phi_r", "raw_phi_t", "raw_theta_r", "raw_theta_t")
            ],
            axis=1,
        )
        interactions = (
            d["raw_interactions"].reshape(d["raw_interactions"].shape[0], -1).copy()
        )
    if config["mode"] == "rt":
        ports = {"V": [0], "H": [1], "VH": [0, 1]}
        a = a[ports[config["devices"]["ue"]["polarization"]]][
            :, ports[config["devices"]["satellite"]["polarization"]]
        ]
    provenance = {
        "rt_source": "cache",
        "cache": str(path),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cache_spec": spec,
        "fresh_rt_search": False,
        "trace_direction": "downlink",
        "coefficient_angle_basis": "satellite_to_ue",
    }
    return _assemble(a, tau, angles, interactions, config, provenance)

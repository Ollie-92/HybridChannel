"""Native RT: scene, antennas, TX/RX, PathSolver, CIR. Follow Sionna 1.2.2 Introduction; server version is 1.2.1. https://nvlabs.github.io/sionna/v1.2.2/rt/tutorials/Introduction.html . Thresholding and Hybrid field conversion are in rt_postprocess.py."""

import json
import numpy as np
from .config import configure_runtime, input_path, rt_antenna_config
from .topology import endpoint_names


def trace_paths(config, positions, session=None):
    """Trace one direction and snapshot; return scene, paths, a and tau. Positions are scene meters; a retains native dimensions and tau absolute seconds. Session reuse retains scene/solver but still traces each call."""
    configure_runtime()
    import mitsuba as mi
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

    settings = config["rt"]
    solver_settings = settings["solver"]
    tx_name, rx_name = endpoint_names(config["direction"])
    tx_device = config["devices"][tx_name]
    rx_device = config["devices"][rx_name]
    hybrid_probe = config["mode"] == "hybrid"
    if mi.variant() != settings.get("variant", "cuda_ad_mono_polarized"):
        raise RuntimeError("Mitsuba variant 與設定不符，請核對環境。")

    # 1. Load the scene; reuse it along the trajectory to avoid reloading city geometry.
    signature = json.dumps(
        {key: config[key] for key in ("mode", "direction", "rt", "radio")}, sort_keys=True
    )
    if session is not None and "scene" in session:
        if session["signature"] != signature:
            raise ValueError("同一軌跡內的 RT 設定已改變，請重新建立 HybridChannel。")
        scene = session["scene"]
    else:
        scene_file = settings["scene_xml"]
        scene = load_scene(None if scene_file is None else str(input_path(scene_file)))
    scene.frequency = config["radio"]["frequency_hz"]
    if settings.get("scattering_coefficient") is not None:
        for material in scene.radio_materials.values():
            material.scattering_coefficient = settings["scattering_coefficient"]

    # 2. Read endpoint antennas; swap TX/RX for DL/UL. Hybrid uses its internal probe.
    tx_antenna = rt_antenna_config(config, tx_device)
    rx_antenna = rt_antenna_config(config, rx_device)
    scene.tx_array = PlanarArray(
        num_rows=tx_antenna["num_rows"],
        num_cols=tx_antenna["num_cols"],
        vertical_spacing=tx_antenna["vertical_spacing"],
        horizontal_spacing=tx_antenna["horizontal_spacing"],
        pattern=tx_antenna["pattern"],
        polarization=tx_antenna["polarization"],
    )
    scene.rx_array = PlanarArray(
        num_rows=rx_antenna["num_rows"],
        num_cols=rx_antenna["num_cols"],
        vertical_spacing=rx_antenna["vertical_spacing"],
        horizontal_spacing=rx_antenna["horizontal_spacing"],
        pattern=rx_antenna["pattern"],
        polarization=rx_antenna["polarization"],
    )
    # Legacy custom element offsets; the notebook supplies full PlanarArray settings instead.
    if not hybrid_probe:
        if "antenna" not in tx_device:
            scene.tx_array = apply_array_offsets(scene.tx_array, tx_device, scene.wavelength)
        if "antenna" not in rx_device:
            scene.rx_array = apply_array_offsets(scene.rx_array, rx_device, scene.wavelength)

    # 3. Place TX/RX at this snapshot: satellite to UE for DL, UE to satellite for UL.
    for name in list(scene.transmitters) + list(scene.receivers):
        scene.remove(name)
    device_names = {} if session is None else session.get("device_names", {})
    tx = Transmitter(
        name=device_names.get(tx_name, "configured_tx"),
        position=positions[tx_name].tolist(),
        orientation=[0, 0, 0] if hybrid_probe else np.deg2rad(tx_device["orientation_deg"]).tolist(),
    )
    rx = Receiver(
        name=device_names.get(rx_name, "configured_rx"),
        position=positions[rx_name].tolist(),
        orientation=[0, 0, 0] if hybrid_probe else np.deg2rad(rx_device["orientation_deg"]).tolist(),
    )
    scene.add(tx)
    scene.add(rx)

    # 4. Trace native paths. Omitted parameters retain the verified Sionna 1.2.1 defaults.
    path_solver = session["solver"] if session is not None and "solver" in session else PathSolver()
    paths = path_solver(
        scene=scene,
        max_depth=solver_settings.get("max_depth", 3),
        samples_per_src=solver_settings.get("samples_per_src", 1_000_000),
        max_num_paths_per_src=solver_settings.get("max_num_paths_per_src", 1_000_000),
        los=solver_settings.get("los", True),
        specular_reflection=solver_settings.get("specular_reflection", True),
        diffuse_reflection=solver_settings.get("diffuse_reflection", False),
        refraction=solver_settings.get("refraction", True),
        diffraction=solver_settings.get("diffraction", False),
        edge_diffraction=solver_settings.get("edge_diffraction", False),
        diffraction_lit_region=solver_settings.get("diffraction_lit_region", True),
        synthetic_array=True,
        seed=config["seed"],
    )

    # 5. Obtain native CIR without filtering paths, recomputing phases or reversing direction here.
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    if session is not None:
        session.update(scene=scene, solver=path_solver, signature=signature)
    return scene, paths, a, tau


def apply_array_offsets(array, device, wavelength):
    """Convert existing custom offsets from scene meters to antenna-local wavelengths. Undo device rotation because Sionna rotates arrays itself. This adapts input coordinates without custom antenna-field or array-phase computation."""
    offsets = np.asarray(device["array_offsets_m"], float)
    if offsets.shape == (1, 3) and not np.any(offsets):
        return array
    import mitsuba as mi
    from sionna.rt import AntennaArray

    a, b, g = np.deg2rad(device["orientation_deg"])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
    local_offsets = offsets @ (rz @ ry @ rx)
    return AntennaArray(array.antenna_pattern, mi.Point3f((local_offsets / wavelength).T))

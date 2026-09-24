# Source: YuChein/run_satellite_trajectory_channel.py; SHA-256 in docs/provenance.json.
# Modified for this local review on 2026-09-23: packaging and documentation.
# Original code rights: research team (confirmed by project owner); LICENSE pending.
# B01-B09, B17-B18, B21: original straight-line trajectory driver. No TLE or general channel-mode dispatcher. Use the experiment wrapper for isolated output and metadata.

"""
Generate hybrid channel responses along a simple satellite trajectory.

This script is a trajectory driver around the existing Sionna RT + hybrid
pipeline:

    satellite position at t_k
      -> Sionna PathSolver
      -> extract RT paths
      -> generate_full_channel()
      -> save H(t_k), tau(t_k), and metadata

The default trajectory is a short straight pass at fixed altitude. It is meant
for local time-varying channel experiments, not orbital mechanics from TLE.
"""

import argparse
import json
import os
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import mitsuba as mi
import numpy as np

from sionna.rt import (
    LambertianPattern,
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)

from .cluster_splitter import extract_paths_from_rt
from .hybrid_channel import generate_full_channel
from .consistent_random_clusters import (
    evaluate_random_clusters,
    freeze_random_cluster_state,
)
from .lsp_calculator import has_los


C0 = 299_792_458.0


def as_float_list(values):
    return [float(v) for v in values]


def set_mitsuba_variant():
    try:
        mi.set_variant("cuda_ad_mono_polarized")
        return "cuda_ad_mono_polarized"
    except Exception:
        mi.set_variant("llvm_ad_mono_polarized")
        return "llvm_ad_mono_polarized"


def build_scene(scene_path, fc_hz, tx_power_dbm, rx_position):
    """Load a self-contained scene and set the original VV probes and materials.

    Frequency is Hz, positions meters, and configured TX power dBm.
    Material scattering overrides are preserved from the original experiment."""
    scene = load_scene(scene_path, merge_shapes=True)
    scene.frequency = fc_hz
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    for mat in scene.radio_materials.values():
        if hasattr(mat, "scattering_coefficient"):
            mat.scattering_coefficient = 0.25
            mat.scattering_pattern = LambertianPattern()

    scene.add(
        Transmitter(
            name="Sat",
            position=[0.0, 0.0, 600_000.0],
            orientation=[0.0, 0.0, 0.0],
            power_dbm=tx_power_dbm,
        )
    )
    scene.add(
        Receiver(
            name="RX_probe",
            position=as_float_list(rx_position),
            orientation=[0.0, 0.0, 0.0],
        )
    )
    return scene


def make_linear_trajectory(
    rx_position,
    initial_elev_deg,
    altitude_m,
    speed_mps,
    heading_deg,
    duration_s,
    num_steps,
):
    """Short straight trajectory in the local scene coordinate system."""
    rx = np.asarray(rx_position, dtype=np.float64)
    horizontal_range = altitude_m / np.tan(np.radians(initial_elev_deg))
    center = np.array([rx[0] + horizontal_range, rx[1], altitude_m], dtype=np.float64)

    heading = np.radians(heading_deg)
    velocity = speed_mps * np.array([np.cos(heading), np.sin(heading), 0.0])

    times = np.linspace(0.0, duration_s, num_steps)
    t_mid = 0.5 * duration_s
    positions = np.array([center + velocity * (t - t_mid) for t in times])
    velocities = np.repeat(velocity[np.newaxis, :], num_steps, axis=0)
    return times, positions, velocities


def elevation_deg(sat_position, rx_position):
    """Return elevation in degrees from the scene-local satellite/receiver XYZ meters."""
    rel = np.asarray(sat_position, dtype=np.float64) - np.asarray(rx_position, dtype=np.float64)
    horizontal = np.linalg.norm(rel[:2])
    return float(np.degrees(np.arctan2(rel[2], horizontal)))


def make_array_geom(fc_ghz, num_elements):
    """Return an x-axis ULA of shape (elements, 3) in meters.

    Preserve the original approximate wavelength 0.3 / fc_GHz."""
    wavelength = 0.3 / fc_ghz
    spacing = wavelength / 2.0
    return np.array([[i * spacing, 0.0, 0.0] for i in range(num_elements)], dtype=np.float64)


def rt_paths_to_hybrid_input(path_dict, pre_filter_db):
    """Filter by relative power (dB), then sort RT records by absolute delay.

    Output delays are seconds, powers linear gains, and all angles radians."""
    powers = path_dict["powers"]
    if len(powers) == 0:
        return []

    p_max = np.max(powers)
    keep = powers >= p_max * 10.0 ** (pre_filter_db / 10.0)
    kept_indices = np.where(keep)[0]
    kept_indices = kept_indices[np.argsort(path_dict["delays"][kept_indices])]

    rt_paths = []
    for idx in kept_indices:
        rt_paths.append(
            {
                "tau_rt": float(path_dict["delays"][idx]),
                "power_rt": float(path_dict["powers"][idx]),
                "aoa": float(path_dict["aoa"][idx]),
                "aod": float(path_dict["aod"][idx]),
                "zoa": float(path_dict["zoa"][idx]),
                "zod": float(path_dict["zod"][idx]),
            }
        )
    return rt_paths


def summarize_channel(result):
    """Summarize dimensions and sampled LSPs; mean H power includes padded zeros."""
    H = result["channel"]["H"]
    tau = result["channel"]["tau"]
    clusters = result["hybrid"]["clusters"]
    rays = result["rays"]
    sp = result["scenario_params"]
    return {
        "H_shape": list(H.shape),
        "tau_shape": list(tau.shape),
        "num_clusters": len(clusters),
        "num_rt_clusters": int(sum(c["source"] == "RT" for c in clusters)),
        "num_rc_clusters": int(sum(c["source"] == "RC" for c in clusters)),
        "num_rays": len(rays),
        "sampled_DS_ns": float(sp["DS"] * 1e9),
        "sampled_K_dB": None if sp.get("K_R_dB") is None else float(sp["K_R_dB"]),
        "mean_channel_power": float(np.mean(np.abs(H) ** 2)),
    }


def main():
    """Execute the original LOS-gated trajectory with independent Hybrid or RC-only state.

    Save variable-size CIR tensors and geometry after all positions are evaluated."""
    parser = argparse.ArgumentParser(
        description="Compute hybrid channel responses along a satellite trajectory."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", default="satellite_trajectory_channel.npz")
    parser.add_argument("--summary", default="satellite_trajectory_channel_summary.txt")
    parser.add_argument("--rx-x", type=float, default=-244.4)
    parser.add_argument("--rx-y", type=float, default=-1435.2)
    parser.add_argument("--rx-z", type=float, default=2.0)
    parser.add_argument("--fc-ghz", type=float, default=28.0)
    parser.add_argument("--tx-power-dbm", type=float, default=60.0)
    parser.add_argument("--altitude-m", type=float, default=600_000.0)
    parser.add_argument("--initial-elev-deg", type=float, default=40.0)
    parser.add_argument("--speed-mps", type=float, default=7_560.0)
    parser.add_argument("--heading-deg", type=float, default=90.0)
    parser.add_argument("--duration-s", type=float, default=0.2)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--samples-per-src", type=int, default=3_000_000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-num-paths", type=int, default=2000)
    parser.add_argument("--rt-pre-filter-db", type=float, default=-25.0)
    parser.add_argument("--num-ant", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--consistent-random-clusters",
        action="store_true",
        help=(
            "Freeze random clusters at the first valid step and evolve only by "
            "satellite geometry. H_by_step/tau_by_step then contain the "
            "random-cluster-only channel; deterministic RT combining is out of scope."
        ),
    )
    args = parser.parse_args()

    basedir = os.getcwd()
    scene_path = args.scene
    if not os.path.isabs(scene_path):
        scene_path = os.path.join(basedir, scene_path)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(basedir, output_path)
    summary_path = args.summary
    if not os.path.isabs(summary_path):
        summary_path = os.path.join(basedir, summary_path)

    variant = set_mitsuba_variant()
    rx_position = np.array([args.rx_x, args.rx_y, args.rx_z], dtype=np.float64)
    fc_hz = args.fc_ghz * 1e9

    scene = build_scene(scene_path, fc_hz, args.tx_power_dbm, rx_position)
    solver = PathSolver()
    tx = scene.transmitters["Sat"]
    rx = scene.receivers["RX_probe"]
    rx.position = as_float_list(rx_position)

    path_all_kwargs = dict(
        max_depth=args.max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=True,
        synthetic_array=True,
        samples_per_src=args.samples_per_src,
        max_num_paths_per_src=args.max_num_paths,
    )
    path_los_kwargs = dict(
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=True,
        samples_per_src=args.samples_per_src,
        max_num_paths_per_src=1,
    )

    times, sat_positions, sat_velocities = make_linear_trajectory(
        rx_position=rx_position,
        initial_elev_deg=args.initial_elev_deg,
        altitude_m=args.altitude_m,
        speed_mps=args.speed_mps,
        heading_deg=args.heading_deg,
        duration_s=args.duration_s,
        num_steps=args.num_steps,
    )

    rx_array = make_array_geom(args.fc_ghz, args.num_ant)
    tx_array = make_array_geom(args.fc_ghz, args.num_ant)

    H_list = []
    tau_list = []
    valid_steps = []
    step_summaries = []
    consistent_rc_state = None

    print(f"Mitsuba variant: {variant}")
    print(f"Scene: {scene_path}")
    print(f"RX: {rx_position.tolist()}")
    print(f"Steps: {args.num_steps}, duration={args.duration_s}s")
    print(f"samples_per_src={args.samples_per_src}")

    t_start = time.time()
    for step_idx, (t_abs, sat_pos, sat_vel) in enumerate(
        zip(times, sat_positions, sat_velocities)
    ):
        tx.position = as_float_list(sat_pos.astype(np.float32))
        elev = elevation_deg(sat_pos, rx_position)

        print(
            f"\n[{step_idx + 1}/{args.num_steps}] "
            f"t={t_abs:.6f}s elev={elev:.3f}deg "
            f"sat=[{sat_pos[0]:.1f}, {sat_pos[1]:.1f}, {sat_pos[2]:.1f}]"
        )

        paths_los = solver(scene=scene, seed=args.seed, **path_los_kwargs)
        if not has_los(paths_los):
            print("  No LOS path; step skipped")
            step_summaries.append({"step": step_idx, "valid": False, "reason": "no_los"})
            continue

        rt_t0 = time.time()
        paths_all = solver(
            scene=scene,
            seed=args.seed + 10_000 + step_idx,
            **path_all_kwargs,
        )
        rt_elapsed = time.time() - rt_t0

        path_dict = extract_paths_from_rt(paths_all, paths_los)
        if path_dict is None:
            print("  No valid RT paths; step skipped")
            step_summaries.append({"step": step_idx, "valid": False, "reason": "no_paths"})
            continue

        rt_paths = rt_paths_to_hybrid_input(path_dict, args.rt_pre_filter_db)
        if not rt_paths:
            print("  No RT paths after pre-filter; step skipped")
            step_summaries.append(
                {"step": step_idx, "valid": False, "reason": "all_filtered"}
            )
            continue

        ch_t0 = time.time()
        if args.consistent_random_clusters:
            if consistent_rc_state is None:
                rng = np.random.default_rng(args.seed)
                consistent_rc_state = freeze_random_cluster_state(
                    rt_paths,
                    "NTN-DenseUrban-LOS",
                    rx_position,
                    sat_pos,
                    args.fc_ghz,
                    elev_deg=elev,
                    rng=rng,
                    config={"is_LOS_first_cluster": True},
                )
                meta = consistent_rc_state["metadata"]
                print(
                    "  Consistent RC freeze: "
                    f"clusters={meta['num_frozen_rc_clusters']} "
                    f"dropped_los_aligned={meta['dropped_los_aligned']} "
                    f"dropped_nonpositive={meta['dropped_nonpositive_delay']}"
                )

            rc_result = evaluate_random_clusters(
                consistent_rc_state,
                sat_pos,
                rt_paths_current=rt_paths,
                rx_array_geom=rx_array,
                tx_array_geom=tx_array,
                t=t_abs,
            )
            result = {
                "channel": rc_result["channel"],
                "hybrid": {"clusters": rc_result["clusters"]},
                "rays": rc_result["rays"],
                "scenario_params": consistent_rc_state["scenario_params"],
                "consistent_random_clusters": rc_result,
            }
        else:
            rng = np.random.default_rng(args.seed + step_idx)
            result = generate_full_channel(
                rt_paths,
                "NTN-DenseUrban-LOS",
                rx_array,
                tx_array,
                elev_deg=elev,
                fc_GHz=args.fc_ghz,
                velocity_vec=sat_vel,
                t=t_abs,
                rng=rng,
                config={"is_LOS_first_cluster": True},
                tx_rx_distance_3D=float(np.linalg.norm(sat_pos - rx_position)),
            )
        ch_elapsed = time.time() - ch_t0

        H = result["channel"]["H"]
        tau = result["channel"]["tau"]
        summary = summarize_channel(result)
        summary["consistent_random_clusters"] = bool(args.consistent_random_clusters)
        summary.update(
            {
                "step": step_idx,
                "valid": True,
                "time_s": float(t_abs),
                "elev_deg": float(elev),
                "sat_position_m": sat_pos.tolist(),
                "sat_velocity_mps": sat_vel.tolist(),
                "num_raw_rt_paths": int(len(path_dict["delays"])),
                "num_prefiltered_rt_paths": int(len(rt_paths)),
                "rt_seconds": float(rt_elapsed),
                "channel_seconds": float(ch_elapsed),
            }
        )
        step_summaries.append(summary)
        valid_steps.append(step_idx)
        H_list.append(H)
        tau_list.append(tau)

        print(
            "  RT paths: "
            f"{len(path_dict['delays'])} raw, {len(rt_paths)} kept; "
            f"H={H.shape}, tau={tau.shape}; "
            f"RT={rt_elapsed:.2f}s, channel={ch_elapsed * 1000:.1f}ms"
        )

    H_by_step = np.empty(len(H_list), dtype=object)
    tau_by_step = np.empty(len(tau_list), dtype=object)
    for i, (H, tau) in enumerate(zip(H_list, tau_list)):
        H_by_step[i] = H
        tau_by_step[i] = tau

    np.savez_compressed(
        output_path,
        times_s=times,
        sat_positions_m=sat_positions,
        sat_velocities_mps=sat_velocities,
        valid_steps=np.array(valid_steps, dtype=np.int64),
        H_by_step=H_by_step,
        tau_by_step=tau_by_step,
        rx_position_m=rx_position,
        fc_ghz=np.array(args.fc_ghz),
        summaries_json=np.array(json.dumps(step_summaries, indent=2)),
    )

    lines = [
        "Satellite trajectory hybrid channel summary",
        "=" * 52,
        f"scene: {scene_path}",
        f"output: {output_path}",
        f"rx_position_m: {rx_position.tolist()}",
        f"fc_ghz: {args.fc_ghz}",
        f"num_steps: {args.num_steps}",
        f"consistent_random_clusters: {args.consistent_random_clusters}",
        f"valid_steps: {len(valid_steps)}",
        f"elapsed_s: {time.time() - t_start:.2f}",
        "",
    ]
    for item in step_summaries:
        if not item.get("valid"):
            lines.append(f"step {item['step']}: skipped ({item.get('reason')})")
            continue
        lines.append(
            f"step {item['step']}: t={item['time_s']:.6f}s "
            f"elev={item['elev_deg']:.3f}deg "
            f"H={tuple(item['H_shape'])} rays={item['num_rays']} "
            f"RT/RC={item['num_rt_clusters']}/{item['num_rc_clusters']} "
            f"rawRT={item['num_raw_rt_paths']} keptRT={item['num_prefiltered_rt_paths']} "
            f"DS={item['sampled_DS_ns']:.3f}ns K={item['sampled_K_dB']:.3f}dB "
            f"consistentRC={item.get('consistent_random_clusters', False)}"
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

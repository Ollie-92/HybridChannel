"""Analyze completed channels: PDP, CFR, LSP and complex coefficients without regeneration. Selected-port powers are dimensionless gains, not dBm; no extra transmit power or path loss. Angular spreads use TR 38.901 V19.2.0 Annex A.1 circular standard deviation."""

import csv
import json
from html import escape
import numpy as np
from .result import checked_output


def power_db(power):
    """Return -inf for zero power without inventing weak paths through an artificial floor."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.asarray(power, float))


def angular_spread_deg(angle, weights):
    """Compute sqrt(-2 ln |sum(w exp(j angle))|), including wraparound. Rotate near the first angle and evaluate 1-|resultant| to avoid cancellation for very small satellite-side spreads."""
    angle, weights = np.asarray(angle, float), np.asarray(weights, float)
    if not len(angle) or not np.any(weights > 0):
        return None
    w = weights / weights.sum()
    delta = (angle - angle[0] + np.pi) % (2 * np.pi) - np.pi
    delta -= np.sum(w * delta)
    a = np.sum(w * 2 * np.sin(delta / 2) ** 2)
    b = np.sum(w * np.sin(delta))
    q = float(np.clip(2 * a - a * a - b * b, 0.0, 1.0))
    deficit = q / (1 + np.sqrt(1 - q))
    if deficit >= 1:
        return None  # A fully dispersed direction has no finite circular spread.
    return float(np.rad2deg(np.sqrt(-2 * np.log1p(-deficit))))


def channel_metrics(
    delays_s, powers, angles_rad, is_los, *, angles_available=True, los_available=True
):
    """Single-realization metrics, not multi-position LSP mean/std. K is labeled LOS-to-other-ray power ratio, not a time-series Rice estimate. Spreads include LOS, unlike legacy NLOS-only ASA/ASD."""
    tau, p, angles = map(np.asarray, (delays_s, powers, angles_rad))
    los = np.asarray(is_los, bool)
    good = (p > 0) & np.isfinite(p) & np.isfinite(tau)
    if angles_available and not np.all(np.isfinite(angles[good])):
        raise ValueError("Non-finite angles in a nonzero channel ray")
    tau, p, angles, los = tau[good], p[good], angles[good], los[good]
    stats = dict(
        nonzero_rays=int(len(p)),
        total_gain_db=None,
        mean_delay_ns=None,
        DS_ns=None,
        K_db=None,
        K_status="no_signal",
        ASA_deg=None,
        ASD_deg=None,
        ZSA_deg=None,
        ZSD_deg=None,
    )
    if not len(p):
        return stats
    w = p / p.sum()
    mean = float(np.sum(w * tau))
    stats.update(
        total_gain_db=float(power_db(p.sum())),
        mean_delay_ns=mean * 1e9,
        DS_ns=float(np.sqrt(np.sum(w * (tau - mean) ** 2)) * 1e9),
    )
    if los_available:
        plos, pnlos = float(p[los].sum()), float(p[~los].sum())
        if plos and pnlos:
            stats.update(K_db=float(10 * np.log10(plos / pnlos)), K_status="finite")
        else:
            stats["K_status"] = "only_los (+inf)" if plos else "no_los (-inf)"
    else:
        stats["K_status"] = "unavailable: aggregated OpenNTN taps"
    if angles_available:
        for i, key in enumerate(("ASA_deg", "ASD_deg", "ZSA_deg", "ZSD_deg")):
            stats[key] = angular_spread_deg(angles[:, i], p)
    return stats


def _port_label(result, side, index):
    name = result.metadata["endpoints"][0 if side == "tx" else 1]
    pol = result.metadata["polarization_ports"][name]
    polarizations = ["V", "H"] if pol == "VH" else [pol]
    return f"{side.upper()} {name} element {index//len(polarizations)}, {polarizations[index%len(polarizations)]}"


def _component_label(result, source):
    """RT labels in Hybrid identify cluster origin, not native geometric paths for every output ray."""
    if result.metadata["mode"] == "hybrid":
        return "Hybrid RT-derived ray" if source == "RT" else "Hybrid RC ray"
    if result.metadata["mode"] == "3gpp":
        return "OpenNTN aggregated tap"
    return "Native RT path"


def _metric_rows(output_metrics, reference_metrics, sampled):
    """Keep sampled model parameters separate from measured output metrics; they need not match per realization."""
    rows = []
    specs = [
        ("DS_ns", "DS", "ns", 1e9),
        ("K_db", "K_R_dB", "dB", 1),
        ("ASA_deg", "ASA", "deg", 1),
        ("ASD_deg", "ASD", "deg", 1),
        ("ZSA_deg", "ZSA", "deg", 1),
        ("ZSD_deg", "ZSD", "deg", 1),
    ]
    for key, sample_key, unit, scale in specs:

        def value(metrics):
            return (
                metrics["K_status"]
                if key == "K_db" and metrics[key] is None
                else metrics[key]
            )

        sv = sampled.get(sample_key)
        rows.append(
            dict(
                metric=sample_key,
                unit=unit,
                RT_reference=value(reference_metrics),
                generated=value(output_metrics),
                sampled_model_input=None if sv is None else float(sv) * scale,
            )
        )
    rows.append(
        dict(
            metric="SF",
            unit="dB",
            RT_reference=None,
            generated=None,
            sampled_model_input=None,
            note="A single link snapshot cannot estimate shadow-fading statistics.",
        )
    )
    return rows


def analyze_channel(result, *, rx_port=0, tx_port=0):
    """Read ChannelResult and return displayable CSV/JSON values. Hybrid uses the pre-generation VV reference, possibly differing from the selected output port; labels identify this. RT-only uses matching ports."""
    if not (
        0 <= rx_port < result.coefficients.shape[0]
        and 0 <= tx_port < result.coefficients.shape[1]
    ):
        raise ValueError(
            f"Port index outside coefficients shape {result.coefficients.shape}"
        )
    h = result.coefficients[rx_port, tx_port]
    p = np.abs(h) ** 2
    angles_available = result.metadata.get("angles_available", True)
    los_available = result.metadata.get("los_flags_available", True)
    los = result.extra_arrays.get("is_los_ray", np.zeros(len(p), bool))
    if los_available and "is_los_ray" not in result.extra_arrays:
        raise ValueError("Missing is_los_ray in a ray-level result")
    output_metrics = channel_metrics(
        result.delays_s,
        p,
        result.angles_rad,
        los,
        angles_available=angles_available,
        los_available=los_available,
    )
    extra = result.extra_arrays
    has_reference = "rt_reference_coefficients" in extra
    if has_reference:
        # The Hybrid reference uses a VV probe in canonical DL order; swap UL angle columns for comparison.
        hybrid = result.metadata["mode"] == "hybrid"
        ri, ti = (0, 0) if hybrid else (rx_port, tx_port)
        ref_p = np.abs(extra["rt_reference_coefficients"][ri, ti]) ** 2
        ref_tau = extra["rt_reference_delays_s"]
        ref_angles = extra["rt_reference_angles_rad"].copy()
        if hybrid and result.metadata["direction"] == "uplink":
            ref_angles = ref_angles[:, [1, 0, 3, 2]]
        ref_metrics = channel_metrics(
            ref_tau, ref_p, ref_angles, extra["rt_reference_is_los"]
        )
        ref_label = (
            "RT reference: unrotated 1-element VV probe, before ionosphere"
            if hybrid
            else "RT reference: same ports, before ionosphere"
        )
    else:
        ref_p, ref_tau = np.zeros(0), np.zeros(0)
        ref_metrics = channel_metrics([], [], np.empty((0, 4)), [])
        ref_label = (
            "Not applicable: 3GPP-only does not run RT"
            if result.metadata["mode"] == "3gpp"
            else "RT reference unavailable in this saved result"
        )
    # Show native RT separately; 20 expanded Hybrid subrays are not 20 native paths.
    reference_rows = []
    raw_indices = result.metadata.get("rt", {}).get("selected_raw_indices", [])
    if has_reference:
        for i, (delay, power, angle) in enumerate(zip(ref_tau, ref_p, ref_angles)):
            row = dict(
                retained_rt_id=i,
                raw_path_id=int(raw_indices[i]) if i < len(raw_indices) else None,
                is_los=bool(extra["rt_reference_is_los"][i]),
                delay_ns=float(delay * 1e9),
                absolute_delay_s=float(delay + result.delay_reference_s),
                gain_db=float(power_db(power)) if power > 0 else None,
            )
            row.update(
                {
                    key: float(value) if np.isfinite(value) else None
                    for key, value in zip(
                        ("AoA_deg", "AoD_deg", "ZoA_deg", "ZoD_deg"), np.rad2deg(angle)
                    )
                }
            )
            reference_rows.append(row)
    port_label = (
        _port_label(result, "rx", rx_port) + " / " + _port_label(result, "tx", tx_port)
    )
    path_rows = []
    ids = extra.get("cluster_ray_ids")
    for i, (coefficient, power, delay, angle) in enumerate(
        zip(h, p, result.delays_s, result.angles_rad)
    ):
        row = dict(
            ray_id=i,
            source=str(result.sources[i]),
            component=_component_label(result, result.sources[i]),
            cluster_id=int(ids[i, 0]) if ids is not None else None,
            cluster_ray_id=int(ids[i, 1]) if ids is not None else None,
            is_los=bool(los[i]) if los_available else None,
            delay_ns=float(delay * 1e9),
            absolute_delay_s=float(result.absolute_delays_s[i]),
            real=float(coefficient.real),
            imag=float(coefficient.imag),
            magnitude=float(abs(coefficient)),
            phase_deg=float(np.rad2deg(np.angle(coefficient))) if power > 0 else None,
            gain_db=float(power_db(power)) if power > 0 else None,
        )
        row.update(
            {
                key: float(value) if np.isfinite(value) else None
                for key, value in zip(
                    ("AoA_deg", "AoD_deg", "ZoA_deg", "ZoD_deg"), np.rad2deg(angle)
                )
            }
        )
        if "xpr_linear" in extra:
            row["sampled_xpr_db"] = float(power_db(extra["xpr_linear"][i]))
        path_rows.append(row)
    # Cluster summaries expose shared delays and subray counts; powers are explicitly incoherent sums.
    cluster_rows = []
    if ids is not None:
        for cid in np.unique(ids[:, 0]):
            mask = ids[:, 0] == cid
            cluster_sources = np.unique(result.sources[mask])
            cluster_rows.append(
                dict(
                    cluster_id=int(cid),
                    source="/".join(cluster_sources),
                    ray_count=int(mask.sum()),
                    distinct_delays=int(len(np.unique(result.delays_s[mask]))),
                    delay_min_ns=float(result.delays_s[mask].min() * 1e9),
                    delay_max_ns=float(result.delays_s[mask].max() * 1e9),
                    incoherent_gain_db=(
                        float(power_db(p[mask].sum())) if p[mask].sum() > 0 else None
                    ),
                )
            )
    # H(fc) sums complex coefficients; singular values use the port matrix, not one dimension per ray.
    matrix_fc = result.coefficients.sum(axis=-1)
    singular_values = np.linalg.svd(matrix_fc, compute_uv=False)
    report = dict(
        rx_port=rx_port,
        tx_port=tx_port,
        port_label=port_label,
        output_metrics=output_metrics,
        reference_metrics=ref_metrics,
        reference_label=ref_label,
        reference_powers=ref_p,
        reference_delays_s=ref_tau,
        powers=p,
        metric_rows=_metric_rows(
            output_metrics, ref_metrics, result.metadata.get("sampled_lsp", {})
        ),
        path_rows=path_rows,
        reference_path_rows=reference_rows,
        cluster_rows=cluster_rows,
        singular_values=singular_values,
        definitions={
            "PDP": "Per-ray |h|^2, incoherent powers; coincident-delay rays remain separate. Not a bandwidth-filtered PDP.",
            "delay": "Same geometric RT reference as ChannelResult; common excess delay is preserved.",
            "AS": "TR 38.901 V19.2.0 Annex A.1, sqrt(-2 ln |sum w exp(j angle)|), all nonzero rays including LoS.",
            "K": "LoS/non-LoS ray-power ratio for this realization; not a temporal Rice distribution estimate.",
            "gain": "10 log10 of dimensionless power gain, not received dBm.",
            "sampled_lsp": "Model input for this seed, with departure/arrival exchanged for UL; not measured channel output.",
            "RT_source": "In Hybrid, RT marks the originating cluster; expanded rays are not independent native RT paths. See reference_path_rows and cluster_rows.",
            "scope": "Single snapshot; no ensemble mean/std, CDF, shadow-fading estimate or dynamic Doppler.",
        },
    )
    if result.metadata["mode"] == "3gpp":
        report["definitions"].update(
            PDP="Native aggregated tap |h|^2 at time index 0; no extra RT threshold or path loss.",
            delay="OpenNTN native excess delays. Absolute reference uses its spherical-Earth slant range / c.",
            AS="Unavailable per aggregated tap; no unique ray angle is returned by the native CIR API.",
            K="Unavailable from aggregated taps: the first tap mixes LoS and diffuse components.",
            RT_source="Not applicable; OpenNTN does not trace city geometry.",
            scope="Displayed statistics use time index 0; full native time block is in channel.npz.",
        )
    return report


def save_analysis(result, analysis, directory):
    """Save coefficient tables, LSP tables and definitions; result.save preserves raw complex arrays."""
    directory = checked_output(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def write_csv(name, rows):
        if not rows:
            (directory / name).write_text("", encoding="utf-8")
            return
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("path_coefficients.csv", analysis["path_rows"])
    write_csv("rt_reference_paths.csv", analysis["reference_path_rows"])
    write_csv("hybrid_clusters.csv", analysis["cluster_rows"])
    write_csv("lsp.csv", analysis["metric_rows"])
    summary = {
        k: analysis[k]
        for k in (
            "rx_port",
            "tx_port",
            "port_label",
            "output_metrics",
            "reference_metrics",
            "reference_label",
            "definitions",
        )
    }
    summary.update(
        sampled_lsp=result.metadata.get("sampled_lsp"),
        ray_count=len(result.delays_s),
        singular_values_fc=analysis["singular_values"].tolist(),
    )
    (directory / "analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def table_html(rows, *, max_rows=None):
    """Render notebook tables without pandas; max_rows does not truncate CSV exports."""
    from IPython.display import HTML

    rows = rows if max_rows is None else rows[:max_rows]
    if not rows:
        return HTML("<p>No rows / 無有效路徑。</p>")
    keys = list(dict.fromkeys(key for row in rows for key in row))

    def fmt(key, value):
        if value is None:
            return "N/A"
        # Preserve precision: absolute delays are milliseconds while path differences can be nanoseconds.
        if isinstance(value, float):
            return f"{value:.15g}" if key.endswith("delay_s") else f"{value:.7g}"
        return str(value)

    head = "".join("<th>" + escape(k) + "</th>" for k in keys)
    body = "".join(
        "<tr>"
        + "".join("<td>" + escape(fmt(k, row.get(k))) + "</td>" for k in keys)
        + "</tr>"
        for row in rows
    )
    return HTML(
        '<div style="overflow-x:auto"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def plot_channel(result, analysis, directory):
    """Create report PNGs without RT execution, data changes or drawing RCs as building paths."""
    import matplotlib.pyplot as plt

    directory = checked_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = analysis["powers"]
    tau = result.delays_s * 1e9
    positive = p > 0
    ref_p = analysis["reference_powers"]
    ref_tau = analysis["reference_delays_s"] * 1e9
    maximum = max(float(p.max(initial=0)), float(ref_p.max(initial=0)))
    floor = -80.0
    figures = {}
    colors = {"RT": "#1976b8", "RC": "#d85b21", "OpenNTN": "#6553a8"}
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.1), layout="constrained", sharex=True, sharey=True
    )
    for ax, delays, power, sources, title in (
        (axes[0], ref_tau, ref_p, np.full(len(ref_p), "RT"), "Retained RT reference"),
        (
            axes[1],
            tau,
            p,
            result.sources,
            f"Generated {result.metadata['mode'].upper()}",
        ),
    ):
        for source in ("RT", "RC", "OpenNTN"):
            mask = (sources == source) & (power > 0)
            if not mask.any():
                continue
            y = power_db(power[mask] / maximum)
            ax.vlines(delays[mask], floor, y, color=colors[source], alpha=0.65)
            label = (
                "RT-derived"
                if source == "RT"
                and ax is axes[1]
                and result.metadata["mode"] == "hybrid"
                else source
            )
            ax.scatter(
                delays[mask],
                y,
                s=18,
                color=colors[source],
                label=f"{label}: {mask.sum()} "
                + ("taps" if source == "OpenNTN" else "rays"),
            )
        if not np.any(power > 0):
            ax.text(0.5, 0.5, "No nonzero paths", transform=ax.transAxes, ha="center")
        ax.set(
            title=title,
            xlabel="Delay relative to geometric RT reference (ns)",
            ylim=(floor, 4),
        )
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)
    axes[0].set_ylabel("Ray power / common peak (dB)")
    fig.suptitle("Per-ray power vs delay; coincident subrays overlap")
    if result.metadata["mode"] == "3gpp":
        axes[0].set_title("No RT reference (3GPP-only)")
        axes[0].texts[0].set_text("No RT search performed")
        for ax in axes:
            ax.set_xlabel("Native excess delay (ns)")
        axes[0].set_ylabel("Tap power / peak (dB)")
        fig.suptitle("OpenNTN native tap PDP at time index 0")
    figures["pdp"] = fig

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.7), layout="constrained")
    h = result.coefficients[analysis["rx_port"], analysis["tx_port"]]
    response = result.cfr[analysis["rx_port"], analysis["tx_port"]]
    for source in ("RT", "RC", "OpenNTN"):
        mask = positive & (result.sources == source)
        if not mask.any():
            continue
        axes[0, 0].scatter(
            tau[mask],
            power_db(p[mask]),
            s=22,
            c=colors[source],
            label=_component_label(result, source),
        )
        axes[0, 1].scatter(
            tau[mask], np.rad2deg(np.angle(h[mask])), s=22, c=colors[source]
        )
    axes[0, 0].set(
        title="Channel coefficient magnitude",
        xlabel="Relative delay (ns)",
        ylabel="20 log10 |h| (dB)",
    )
    axes[0, 1].set(
        title="Channel coefficient phase",
        xlabel="Relative delay (ns)",
        ylabel="arg(h) (deg)",
        ylim=(-185, 185),
    )
    axes[0, 0].legend()
    freq = result.frequency_offsets_hz / 1e6
    good = np.abs(response) > 0
    axes[1, 0].plot(
        freq,
        np.where(good, power_db(np.abs(response) ** 2), np.nan),
        color="#1976b8",
        marker="." if len(freq) == 1 else None,
    )
    # Show wrapped phase in [-180, 180]; do not unwrap across a zero channel.
    axes[1, 1].plot(
        freq,
        np.where(good, np.rad2deg(np.angle(response)), np.nan),
        color="#1976b8",
        marker="." if len(freq) == 1 else None,
    )
    axes[1, 0].set(
        title="CFR magnitude (coherent sum)",
        xlabel="Frequency offset from carrier (MHz)",
        ylabel="20 log10 |H| (dB)",
    )
    axes[1, 1].set(
        title="CFR phase (wrapped)",
        xlabel="Frequency offset from carrier (MHz)",
        ylabel="arg(H) (deg)",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.suptitle(analysis["port_label"], fontsize=12)
    figures["channel_response"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, az, ze, title in ((axes[0], 0, 2, "Arrival"), (axes[1], 1, 3, "Departure")):
        for source in ("RT", "RC", "OpenNTN"):
            mask = positive & (result.sources == source)
            if not mask.any():
                continue
            if not result.metadata.get("angles_available", True):
                continue
            size = 15 + 75 * np.sqrt(p[mask] / p[positive].max())
            ax.scatter(
                np.rad2deg(result.angles_rad[mask, az]),
                np.rad2deg(result.angles_rad[mask, ze]),
                s=size,
                alpha=0.65,
                c=colors[source],
                label=_component_label(result, source),
            )
        ax.set(title=title, xlabel="Azimuth (deg)", ylabel="Zenith (deg)")
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.suptitle(
        "Power-weighted angular distribution; marker area encodes relative amplitude"
    )
    if not result.metadata.get("angles_available", True):
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Not returned per aggregated tap",
                transform=ax.transAxes,
                ha="center",
            )
        fig.suptitle("OpenNTN CIR does not expose unique tap angles")
    figures["angles"] = fig

    fig, ax = plt.subplots(figsize=(6.8, 3.7), layout="constrained")
    singular = analysis["singular_values"]
    nonzero = singular > 0
    if nonzero.any():
        relative = 20 * np.log10(singular[nonzero] / singular.max())
        bottom = min(-80.0, float(relative.min()) - 8)
        ax.bar(
            np.flatnonzero(nonzero) + 1,
            relative - bottom,
            bottom=bottom,
            color="#8665b1",
        )
        ax.set_ylim(bottom, 3)
    else:
        ax.text(0.5, 0.5, "Zero channel matrix", transform=ax.transAxes, ha="center")
    ax.set(
        title="Singular values of H at carrier",
        xlabel="Singular-value index",
        ylabel="20 log10 (sigma / sigma_max) [dB]",
    )
    ax.set_xticks(np.arange(1, len(singular) + 1))
    ax.grid(axis="y", alpha=0.2)
    figures["singular_values"] = fig
    for name, fig in figures.items():
        fig.savefig(directory / (name + ".png"), dpi=160, bbox_inches="tight")
    return figures


"""使用原生 Sionna 渲染本次配置的城市，不重跑 PathSolver。

Native 模式重用求解時的 scene/paths。Cache 模式從快取的場景來源載入
真實 XML，但沒有頂點的舊快取不會被偽造成可繪製的反射線。
"""
from pathlib import Path
import time
from .config import ROOT, configure_runtime


def scene_path_from_result(result):
    """Resolve the scene from result provenance so edited settings cannot mislabel old cached results."""
    rt = result.metadata["rt"]
    if rt["rt_source"] == "native":
        value = rt.get("scene_xml")
        if value is None:
            return None  # A native empty scene is valid.
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
    else:
        value = rt.get("cache_spec", {}).get("config", {}).get("scene")
        if not value:
            raise ValueError(
                "RT cache does not identify its scene XML; cannot infer geometry"
            )
        path = Path(value)
        path = path if path.is_absolute() else ROOT.parent / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene XML unavailable: {path}. Channel plots remain usable."
        )
    return path.resolve()


def _project(camera, point, resolution, fov_deg):
    """Mitsuba camera: forward +Z, screen-left +X, horizontal field of view."""
    import mitsuba as mi

    local = np.asarray(camera.world_transform.inverse() @ mi.Point3f(point)).reshape(3)
    x, y, z = local
    if z <= 0:
        return None
    width, height = resolution
    scale = width / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return np.array([width / 2 - scale * x / z, height / 2 - scale * y / z])


def _select_path_overlay(paths, show_rt_paths):
    """Pass only nonempty native Paths to the renderer without modifying results. Sionna RT 1.2.1 returns None for empty path segments while its renderer expects three arrays. Disable only the empty path overlay; keep scene/device markers. Check native valid masks, not Hybrid or filtered counts."""
    if not show_rt_paths:
        return None, "disabled", None
    if paths is None:
        return None, "unavailable", None
    valid_count = int(np.count_nonzero(np.asarray(paths.valid)))
    if valid_count == 0:
        return None, "empty", 0
    return paths, "available", valid_count


def render_scene(
    result,
    directory,
    *,
    camera_offset_m=(220, -320, 260),
    resolution=(900, 600),
    num_samples=16,
    show_rt_paths=True,
):
    """Return overview and UE close-up. Markers may overlay occluding buildings. Camera offsets are relative to UE; report off-screen satellite coordinates rather than moving a 600 km satellite beside buildings. Save renders separately from channel data."""
    configure_runtime()
    import matplotlib.pyplot as plt
    import sionna.rt as rt
    import mitsuba as mi

    directory = checked_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    offset = np.asarray(camera_offset_m, float)
    if (
        offset.shape != (3,)
        or not np.all(np.isfinite(offset))
        or np.linalg.norm(offset) == 0
    ):
        raise ValueError(
            "camera_offset_m must be a finite nonzero xyz vector in metres"
        )
    if num_samples < 1 or len(resolution) != 2 or min(resolution) < 1:
        raise ValueError("Positive rendering samples and width/height are required")
    start = time.perf_counter()
    path = scene_path_from_result(result)
    scene = result.runtime_context.get("scene")
    reused = scene is not None
    if scene is None:
        scene = rt.load_scene(None if path is None else str(path))
        scene.frequency = result.metadata["frequency_hz"]
        result.runtime_context["scene"] = scene
    paths, overlay_status, valid_count = _select_path_overlay(
        result.runtime_context.get("paths"), show_rt_paths
    )
    positions = {
        k: np.asarray(v, float) for k, v in result.metadata["positions_scene_m"].items()
    }
    ue = positions["ue"]
    satellite = positions["satellite"]
    tx_name, rx_name = result.metadata["endpoints"]
    ue_label = "UE / " + ("TX" if tx_name == "ue" else "RX")
    bbox = scene.mi_scene.bbox()
    lower = np.asarray(bbox.min).reshape(3)
    upper = np.asarray(bbox.max).reshape(3)
    if (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(upper >= lower)
    ):
        center = (lower + upper) / 2
        span = max(float(np.max(upper[:2] - lower[:2])), 200.0)
    else:
        center = ue.copy()
        span = 500.0
    # The native camera far clip is 10 km; use a closer camera and wider field of view.
    # Do not rescale the city or modify the shared renderer camera implementation.
    overview = rt.Camera(
        position=(center + [0, -0.45 * span, 0.75 * span]).tolist(),
        look_at=center.tolist(),
    )
    local = rt.Camera(position=(ue + offset).tolist(), look_at=ue.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
    records = []
    for ax, camera, title, fov in (
        (axes[0], overview, "Configured city", 75.0),
        (axes[1], local, "UE neighbourhood", 45.0),
    ):
        camera_position = np.asarray(camera.position).reshape(3)
        bitmap = scene.render(
            camera=camera,
            resolution=resolution,
            num_samples=num_samples,
            fov=fov,
            paths=paths,
            show_devices=False,
            return_bitmap=True,
        )
        rgb = bitmap.convert(component_format=mi.Struct.Type.UInt8, srgb_gamma=True)
        ax.imshow(np.asarray(rgb))
        ax.axis("off")
        ax.set_title(title)
        for name, label, color in (
            ("ue", ue_label, "#27ba57"),
            (
                "satellite",
                "Satellite / " + ("TX" if tx_name == "satellite" else "RX"),
                "#e34439",
            ),
        ):
            pixel = _project(camera, positions[name], resolution, fov)
            if (
                pixel is not None
                and 0 <= pixel[0] < resolution[0]
                and 0 <= pixel[1] < resolution[1]
            ):
                ax.scatter(
                    *pixel, s=90, c=color, edgecolors="white", linewidth=1.3, zorder=5
                )
                ax.annotate(
                    label,
                    pixel,
                    xytext=(12, 18),
                    textcoords="offset points",
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
                    arrowprops=dict(arrowstyle="-", color=color),
                    fontsize=10,
                )
        records.append(
            dict(
                title=title,
                camera_position_m=camera_position.tolist(),
                fov_deg=fov,
                far_clip_m=10000.0,
            )
        )
    caption = {
        "available": "Native RT geometry before power filtering",
        "empty": "Scene only; native RT found no valid paths",
        "unavailable": "Scene only; no native path vertices available",
        "disabled": "Scene only; path overlay disabled",
    }[overlay_status]
    fig.suptitle(
        f"{result.metadata['direction']} | {result.metadata['frequency_hz']/1e9:g} GHz | {caption}",
        fontsize=12,
    )
    fig.supxlabel(
        f"UE = {np.array2string(ue,precision=2)} m; satellite = {np.array2string(satellite,precision=2)} m\nMarkers indicate configured coordinates, including occluded positions. Statistical RC rays have no physical vertices.",
        fontsize=9,
    )
    fig.savefig(directory / "scene.png", dpi=160, bbox_inches="tight")
    report = dict(
        scene_xml=None if path is None else str(path),
        reused_scene=reused,
        render_seconds=time.perf_counter() - start,
        native_path_overlay=paths is not None,
        path_overlay_status=overlay_status,
        native_valid_path_count=valid_count,
        path_solver_called=False,
        ray_geometry_scope=caption,
        cameras=records,
        camera_offset_m=offset.tolist(),
        resolution=list(resolution),
        num_samples=num_samples,
    )
    (directory / "scene_view.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return fig, report

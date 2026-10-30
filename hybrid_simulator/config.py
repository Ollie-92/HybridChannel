"""Validate configuration before expensive RT operations."""

from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def keys(value, allowed, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    unknown = set(value) - set(allowed.split())
    if unknown:
        raise ValueError(f"{label}: unknown options {sorted(unknown)}")


def vector(value, label):
    a = np.asarray(value, float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{label}: expected three finite numbers")
    return a


def input_path(value):
    """Inputs may be read-only external files; output paths are separately restricted to results."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def validate_config(config):
    c = deepcopy(config)
    keys(
        c,
        "schema_version mode direction seed allow_experimental coordinates devices radio rt hybrid openntn ionosphere trajectory",
        "config",
    )
    if c.get("schema_version") != 1:
        raise ValueError("Use schema_version=1; examples are in tests/data/configs")
    if c.get("mode") not in ("rt", "hybrid", "3gpp"):
        raise ValueError("mode must be rt, hybrid or 3gpp (native OpenNTN)")
    if c.get("direction") not in ("uplink", "downlink"):
        raise ValueError("direction must be uplink or downlink")
    if type(c.get("seed")) is not int or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(c.get("allow_experimental", False)) is not bool:
        raise TypeError("allow_experimental must be boolean")

    geo = c["coordinates"]
    keys(geo, "frame position_unit origin_geodetic_deg_m", "coordinates")
    if geo["frame"] not in ("scene_enu", "ecef") or geo["position_unit"] not in (
        "m",
        "km",
    ):
        raise ValueError("Coordinates support scene_enu/ecef and m/km positions")
    if "origin_geodetic_deg_m" in geo:
        lon, lat, _ = vector(geo["origin_geodetic_deg_m"], "origin [lon,lat,h]")
        if abs(lon) > 180 or abs(lat) > 90:
            raise ValueError("Invalid WGS84 reference longitude/latitude")
    if geo["frame"] == "ecef" and "origin_geodetic_deg_m" not in geo:
        raise ValueError("ECEF requires the scene origin [longitude,latitude,height]")
    keys(c["devices"], "satellite ue", "devices")
    for name in ("satellite", "ue"):
        d = c["devices"][name]
        keys(d, "position array_offsets_m polarization pattern orientation_deg antenna", name)
        vector(d["position"], name + ".position")
        offsets = np.asarray(d["array_offsets_m"], float)
        if (
            offsets.ndim != 2
            or offsets.shape[1] != 3
            or len(offsets) < 1
            or not np.all(np.isfinite(offsets))
        ):
            raise ValueError(
                "array_offsets_m must be a finite (elements,3) scene-frame array"
            )
        vector(d["orientation_deg"], name + ".orientation_deg")
        if d["polarization"] not in ("V", "H", "VH") or d["pattern"] != "iso":
            raise NotImplementedError(
                "Unified entry currently validates iso antennas with V/H/VH ports only"
            )
        if "antenna" in d:
            validate_antenna(d["antenna"], name + ".antenna")
            antenna = d["antenna"]
            if (antenna["polarization"] != d["polarization"]
                    or antenna["pattern"] != d["pattern"]):
                raise ValueError(f"{name} 的 antenna 與既有極化／場型欄位不一致。")
            # Prevent edited JSON from giving native RT and Hybrid inconsistent array geometry.
            expected = np.zeros((1, 3))
            if antenna["num_rows"] * antenna["num_cols"] > 1:
                import mitsuba as mi
                from sionna.rt import PlanarArray
                from scipy.constants import c as light_speed

                array = PlanarArray(**antenna)
                expected = np.asarray(array.rotate(
                    light_speed / c["radio"]["frequency_hz"],
                    mi.Point3f(np.deg2rad(d["orientation_deg"])),
                )).T
            if (offsets.shape != expected.shape
                    or not np.allclose(offsets, expected, rtol=2e-6, atol=1e-8)):
                raise ValueError(f"{name} 的 antenna 與 array_offsets_m 不一致；請重新產生 config。")
    radio = c["radio"]
    keys(radio, "frequency_hz bandwidth_hz frequency_points", "radio")
    if not (np.isfinite(radio["frequency_hz"]) and radio["frequency_hz"] > 0):
        raise ValueError("frequency_hz must be positive")
    if not 0 <= radio["bandwidth_hz"] < 2 * radio["frequency_hz"]:
        raise ValueError("Invalid bandwidth")
    if type(radio["frequency_points"]) is not int or radio["frequency_points"] < 1:
        raise ValueError("frequency_points must be a positive integer")

    rt = c.get("rt")
    if c["mode"] == "3gpp":

        validate_openntn(c)
    else:
        if c.get("openntn") is not None:
            raise ValueError("openntn options only apply to mode=3gpp")
        keys(
            rt,
            "source cache_path scene_xml variant solver scattering_coefficient power_threshold_db probe_antenna",
            "rt",
        )
        if "probe_antenna" in rt:
            if c["mode"] != "hybrid" or rt["source"] != "native":
                raise ValueError("probe_antenna 只用於 Hybrid 的原生 RT 搜尋。")
            validate_antenna(rt["probe_antenna"], "rt.probe_antenna")
            if rt["probe_antenna"] != HYBRID_RT_PROBE:
                raise ValueError("既有 Hybrid 搜尋需使用 1x1 iso VH probe；最終天線請改 endpoints。")
        if rt["source"] == "cache":
            if any(
                k in rt
                for k in ("scene_xml", "variant", "solver", "scattering_coefficient")
            ):
                raise ValueError(
                    "Cached RT cannot change scene, material or search settings; use source=native"
                )
            if not input_path(rt["cache_path"]).is_file():
                raise FileNotFoundError(rt["cache_path"])
        elif rt["source"] == "native":
            if "cache_path" in rt:
                raise ValueError("Native RT does not consume cache_path")
            if (
                rt["scene_xml"] is not None
                and not input_path(rt["scene_xml"]).is_file()
            ):
                raise FileNotFoundError(rt["scene_xml"])
            keys(
                rt["solver"],
                "max_depth max_num_paths_per_src samples_per_src los specular_reflection diffuse_reflection refraction diffraction edge_diffraction diffraction_lit_region",
                "rt.solver",
            )
            for k, v in rt["solver"].items():
                if k in ("max_depth", "max_num_paths_per_src", "samples_per_src"):
                    if type(v) is not int or v < (0 if k == "max_depth" else 1):
                        raise ValueError(f"Invalid {k}")
                elif type(v) is not bool:
                    raise TypeError(f"{k} must be boolean")
            s = rt.get("scattering_coefficient")
            if s is not None and not 0 <= s <= 1:
                raise ValueError("scattering_coefficient must lie in [0,1]")
        else:
            raise ValueError("rt.source must be cache or native")
        if not np.isfinite(rt["power_threshold_db"]) or rt["power_threshold_db"] > 0:
            raise ValueError(
                "power_threshold_db is a nonpositive relative-power threshold"
            )

    h = c.get("hybrid")
    if c["mode"] == "hybrid":
        keys(
            h,
            "backend scenario table_profile rays_per_cluster ray_delay_mode rc_removal_p0 rt_xpr_source",
            "hybrid",
        )
        if h["backend"] != "standard":
            raise ValueError("此新入口使用 standard Hybrid；歷史模型留在舊專案。")
        if h["scenario"] != "NTN-DenseUrban":
            raise NotImplementedError(
                "Unified table audit currently covers NTN-DenseUrban only"
            )
        if h["table_profile"] not in ("legacy", "tr38811_corrected"):
            raise ValueError("Unknown table_profile")
        if h["ray_delay_mode"] not in ("kb1", "legacy_delay"):
            raise ValueError("Unknown ray_delay_mode")
        if (
            type(h["rays_per_cluster"]) is not int
            or not 1 <= h["rays_per_cluster"] <= 20
        ):
            raise ValueError("rays_per_cluster must be 1..20")
        if not 0 <= h["rc_removal_p0"] < 1:
            raise ValueError("Invalid RC removal configuration")
        if h["rt_xpr_source"] not in ("table", "probe_power_ratio"):
            raise ValueError("Unknown rt_xpr_source")
        if h["backend"] == "standard":
            if (
                h["table_profile"],
                h["ray_delay_mode"],
                h["rays_per_cluster"],
                h["rt_xpr_source"],
            ) != ("tr38811_corrected", "kb1", 20, "probe_power_ratio"):
                raise ValueError(
                    "Standard route requires corrected tables, kb1, M=20 and probe_power_ratio XPR"
                )
            if rt["power_threshold_db"] != -25:
                raise ValueError("Standard Clause-8 route uses power_threshold_db=-25")
            if rt["source"] == "native" and not rt["solver"].get("los", True):
                raise ValueError(
                    "Hybrid LOS classification needs los=true; selective no-LOS tracing remains available in RT mode"
                )
            from scipy.constants import c as light_speed

            aperture = max(
                float(
                    np.max(
                        np.linalg.norm(
                            np.array(d["array_offsets_m"])[:, None, :]
                            - np.array(d["array_offsets_m"])[None, :, :],
                            axis=-1,
                        )
                    )
                )
                for d in c["devices"].values()
            )
            if aperture > 0 and radio["bandwidth_hz"] > light_speed / aperture:
                raise NotImplementedError(
                    "B > c/D requires KB>1 frequency-bin fields; this route implements KB=1 only"
                )
    elif h is not None:
        raise ValueError(
            "Non-hybrid mode requires hybrid=null, not ignored Hybrid settings"
        )

    ion = c["ionosphere"]
    keys(
        ion,
        "group_delay phase_advance absorption scintillation faraday_rotation bending correction faraday",
        "ionosphere",
    )
    flags = (
        "group_delay",
        "phase_advance",
        "absorption",
        "scintillation",
        "faraday_rotation",
        "bending",
    )
    for flag in flags:
        if type(ion.get(flag)) is not bool:
            raise TypeError(f"ionosphere.{flag} must be boolean")
    if ion["bending"]:
        raise NotImplementedError(
            "Bending remains experimental in examples/city_bending.py. Its optical/group delay and field export are not yet separated for this entry; do not silently substitute straight RT."
        )
    scalar = any(ion[k] for k in flags[:4])
    if scalar != (ion.get("correction") is not None):
        raise ValueError(
            "Enabled scalar effects require correction; disabled effects require correction=null"
        )
    if ion["faraday_rotation"] != (ion.get("faraday") is not None):
        raise ValueError("Faraday switch/model mismatch")
    if (scalar or ion["faraday_rotation"]) and not c.get("allow_experimental", False):
        raise ValueError(
            "Ionosphere shared-prefix/data coupling requires explicit allow_experimental=true"
        )
    if ion["faraday_rotation"] and (
        c["mode"] != "hybrid" or h["backend"] != "standard"
    ):
        raise NotImplementedError(
            "Faraday needs polarized Hybrid before antenna projection; projected RT coefficients cannot be rotated directly"
        )
    if ion.get("correction"):
        corr = ion["correction"]
        keys(
            corr,
            "kind provenance scope_note group_excess_s optical_excess_m absorption_loss_db scintillation_gain_ri flux_f107 time_utc samples",
            "correction",
        )
        if corr["kind"] not in ("provided_common", "nequick2_straight_common"):
            raise ValueError("Unknown correction provider")
        if not corr.get("provenance") or not corr.get("scope_note"):
            raise ValueError("Correction requires provenance and scope_note")
        allowed = (
            "kind provenance scope_note flux_f107 time_utc samples"
            if corr["kind"] == "nequick2_straight_common"
            else "kind provenance scope_note group_excess_s optical_excess_m absorption_loss_db scintillation_gain_ri"
        )
        keys(corr, allowed, "correction provider")
        if corr["kind"] == "provided_common":
            for flag, field in [
                ("group_delay", "group_excess_s"),
                ("phase_advance", "optical_excess_m"),
                ("absorption", "absorption_loss_db"),
                ("scintillation", "scintillation_gain_ri"),
            ]:
                if ion[flag] and corr.get(field) is None:
                    raise ValueError(f"{flag} requires {field}")
    if ion.get("faraday"):
        f = ion["faraday"]
        keys(f, "mode compensation sampled_path_npz time_utc", "faraday")
        if f["mode"] not in ("reference", "sampled_path"):
            raise ValueError(
                "Faraday model: reference demonstrator or sampled_path with Ne and B"
            )
        if f.get("compensation", "none") not in ("none", "ideal_tx"):
            raise ValueError("Unknown compensation")
        if c["direction"] == "uplink" and f.get("compensation", "none") != "none":
            raise NotImplementedError(
                "Uplink ideal_tx compensation cannot commute through the urban polarization matrix; use compensation=none"
            )
        keys(
            f,
            (
                "mode compensation"
                if f["mode"] == "reference"
                else "mode compensation sampled_path_npz time_utc"
            ),
            "faraday provider",
        )
    from .orbit import validate_trajectory

    validate_trajectory(c)
    tle = c.get("trajectory") and c["trajectory"].get("orbit", {}).get("kind") == "tle"
    if c["mode"] == "hybrid" and h["table_profile"] == "tr38811_corrected" and not tle:
        from .topology import scene_positions, elevation_deg
        from .lsp_calculator import scenario_table

        scenario_table(
            "NTN-DenseUrban-LOS",
            radio["frequency_hz"] / 1e9,
            elevation_deg(scene_positions(c)),
        )
    return c


def validate_openntn(c):
    from .topology import scene_positions, elevation_deg

    if c.get("rt") is not None or c.get("hybrid") is not None:
        raise ValueError("3GPP-only requires rt=null and hybrid=null")
    if c.get("trajectory") is not None:
        raise NotImplementedError(
            "OpenNTN native time block is supported; orbit trajectories and spatial consistency are not connected"
        )
    o = c.get("openntn")
    fields = "scenario los enable_pathloss enable_shadow_fading doppler_enabled num_time_samples sampling_frequency_hz ue_velocity_m_s"
    keys(o, fields, "openntn")
    if set(o) != set(fields.split()):
        raise ValueError(
            "Specify all OpenNTN options explicitly; see Config/examples/openntn_3gpp.json"
        )
    if o["scenario"] not in ("dense_urban", "urban", "suburban"):
        raise ValueError("OpenNTN scenario must be dense_urban, urban or suburban")
    if o["los"] is not None and type(o["los"]) is not bool:
        raise TypeError(
            "openntn.los must be true, false or null (native probability model)"
        )
    for key in ("enable_pathloss", "enable_shadow_fading", "doppler_enabled"):
        if type(o[key]) is not bool:
            raise TypeError(f"openntn.{key} must be boolean")
    if type(o["num_time_samples"]) is not int or o["num_time_samples"] < 1:
        raise ValueError("num_time_samples must be a positive integer")
    if not np.isfinite(o["sampling_frequency_hz"]) or o["sampling_frequency_hz"] <= 0:
        raise ValueError("sampling_frequency_hz must be positive and finite")
    vector(o["ue_velocity_m_s"], "openntn.ue_velocity_m_s")
    for d in c["devices"].values():
        # Named regular arrays map directly to native OpenNTN AntennaArray.
        # Legacy one-element configurations remain supported; arbitrary offsets do not.
        if "antenna" not in d and (
            np.asarray(d["array_offsets_m"]).shape != (1, 3)
            or np.any(d["array_offsets_m"])
        ):
            raise NotImplementedError(
                "OpenNTN multi-element arrays require explicit antenna rows, columns and spacings; arbitrary offsets are unsupported"
            )
    if c["coordinates"]["frame"] != "scene_enu":
        raise NotImplementedError(
            "OpenNTN entry currently needs scene_enu, whose z is interpreted as native height; ECEF is not mapped"
        )
    if c["coordinates"].get("origin_geodetic_deg_m", [0, 0, 0])[2] != 0:
        raise ValueError(
            "Native OpenNTN interprets scene z as height; use a zero-height origin for this entry"
        )
    positions = scene_positions(c)
    if positions["satellite"][2] <= 0 or not 10 <= elevation_deg(positions) <= 90:
        raise ValueError(
            "Native OpenNTN requires positive satellite height and elevation 10..90 degrees"
        )
    frequency = c["radio"]["frequency_hz"]
    if not (1.9e9 <= frequency <= 4e9 or 19e9 <= frequency <= 40e9):
        raise ValueError("Native OpenNTN frequency range: 1.9..4 GHz or 19..40 GHz")
    ion = c["ionosphere"]
    if any(
        ion.get(k)
        for k in (
            "group_delay",
            "phase_advance",
            "absorption",
            "scintillation",
            "faraday_rotation",
            "bending",
        )
    ):
        raise NotImplementedError(
            "3GPP-only uses native OpenNTN losses; project ionosphere extensions are not connected"
        )


def configure_runtime():
    """Keep runtime caches inside this project without editing shared environment configuration files."""
    import os
    import sys

    for variable, name in {
        "XDG_CACHE_HOME": "cache",
        "CUDA_CACHE_PATH": "cuda",
        "OPTIX_CACHE_PATH": "optix",
        "MPLCONFIGDIR": "matplotlib",
        "JUPYTER_RUNTIME_DIR": "jupyter_runtime",
        "IPYTHONDIR": "ipython",
        "JUPYTER_CONFIG_DIR": "jupyter_config",
    }.items():
        path = ROOT / "results/.runtime" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


# Fixed internal probe; notebook users configure only the final endpoint antennas.
HYBRID_RT_PROBE = {
    "num_rows": 1, "num_cols": 1,
    "vertical_spacing": 0.5, "horizontal_spacing": 0.5,
    "pattern": "iso", "polarization": "VH",
}


def validate_antenna(antenna, label):
    """Validate the supported PlanarArray parameters; spacings are in wavelengths."""
    keys(antenna, "num_rows num_cols vertical_spacing horizontal_spacing pattern polarization", label)
    for key in ("num_rows", "num_cols"):
        if type(antenna[key]) is not int or antenna[key] < 1:
            raise ValueError(f"{label}.{key} 必須是正整數。")
    for key in ("vertical_spacing", "horizontal_spacing"):
        if not np.isfinite(antenna[key]) or antenna[key] < 0:
            raise ValueError(f"{label}.{key} 必須是非負有限值，單位為波長。")
    if antenna["pattern"] != "iso" or antenna["polarization"] not in ("V", "H", "VH"):
        raise NotImplementedError("目前主程式支援 iso 與 V/H/VH，未擴充其他場型。")


def rt_antenna_config(config, device):
    """Get native tracing antenna settings from configuration."""
    if config["mode"] == "hybrid":
        return config["rt"].get("probe_antenna", HYBRID_RT_PROBE)
    if "antenna" in device:
        return device["antenna"]
    # Legacy custom offsets: create antenna patterns here; rt.py supplies positions to AntennaArray.
    return {
        "num_rows": 1, "num_cols": 1,
        "vertical_spacing": 0.5, "horizontal_spacing": 0.5,
        "pattern": device["pattern"], "polarization": device["polarization"],
    }


def notebook_devices(settings):
    """Convert named PlanarArray endpoints to channel inputs. Sionna spacing is in wavelengths; Hybrid offsets are rotated scene coordinates in meters. Multi-element arrays use PlanarArray.rotate rather than custom geometry formulas. Reference: Sionna 1.2.2 Introduction and antenna_array API; installed version 1.2.1."""
    import re
    from scipy.constants import c as light_speed

    devices = {}
    for name, endpoint in settings["endpoints"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name):
            raise ValueError("端點名稱請用英數字，例如 sat1、ue1。")
        keys(endpoint, "enabled kind antenna orientation_deg position_m", name)
        if type(endpoint["enabled"]) is not bool:
            raise TypeError(f"{name}.enabled 請填 True 或 False。")
        if not endpoint["enabled"]:
            continue
        if endpoint["kind"] not in ("satellite", "ue"):
            raise ValueError(f"{name}.kind 請選 satellite 或 ue。")
        position = vector(endpoint["position_m"], name + " position_m")
        orientation = vector(endpoint["orientation_deg"], name + " orientation_deg")
        antenna = endpoint["antenna"]
        validate_antenna(antenna, name + " antenna")
        offsets = np.zeros((1, 3))
        if antenna["num_rows"] * antenna["num_cols"] > 1:
            import mitsuba as mi
            from sionna.rt import PlanarArray

            array = PlanarArray(**antenna)
            offsets = np.asarray(array.rotate(
                light_speed / settings["frequency_hz"],
                mi.Point3f(np.deg2rad(orientation)),
            )).T.astype(float)
        devices[name] = {
            "position": position.tolist(),
            "antenna": deepcopy(antenna),
            "array_offsets_m": offsets.tolist(),
            "polarization": antenna["polarization"],
            "pattern": antenna["pattern"],
            "orientation_deg": orientation.tolist(),
        }
    kinds = {settings["endpoints"][name]["kind"] for name in devices}
    if kinds != {"satellite", "ue"}:
        raise ValueError("至少需要啟用一顆衛星與一個 UE。")
    return devices


def notebook_config(settings, *, environment=None):
    """Convert notebook settings to a single-direction configuration; sections A/B supply devices."""
    keys(settings,
         "mode topology directions seed frequency_hz polarization scene_xml output_directory rt hybrid openntn ionosphere endpoints hybrid_rt_probe",
         "notebook settings")
    if settings["topology"] not in ("fixed", "orbit"):
        raise ValueError("topology 請選 fixed 或 orbit。")
    directions = settings["directions"]
    if (not directions or len(directions) != len(set(directions))
            or any(d not in ("downlink", "uplink") for d in directions)):
        raise ValueError("directions 請列出不重複的 downlink / uplink。")
    mode = settings["mode"]
    if mode == "hybrid" and "hybrid_rt_probe" in settings:
        if settings["hybrid_rt_probe"] != HYBRID_RT_PROBE:
            raise ValueError("既有 Hybrid 以 1x1 iso VH probe 取得功率/XPR；請在 endpoints 修改最終天線。")
    config = {
        "schema_version": 1,
        "mode": mode,
        "direction": directions[0],
        "seed": settings["seed"],
        "allow_experimental": settings["topology"] == "orbit",
        "coordinates": {"frame": "scene_enu", "position_unit": "m"},
        "devices": {},
        "radio": {"frequency_hz": settings["frequency_hz"],
                  "bandwidth_hz": 0.0, "frequency_points": 1},
        "rt": None,
        "hybrid": deepcopy(settings["hybrid"]) if mode == "hybrid" else None,
        "ionosphere": deepcopy(settings["ionosphere"]),
        "trajectory": None,
    }
    if mode == "3gpp":
        config["openntn"] = deepcopy(settings["openntn"])
    else:
        rt = deepcopy(settings["rt"])
        # Keep old callers compatible; the notebook sets the runtime variant separately.
        variant = rt.pop("variant", "cuda_ad_mono_polarized")
        if environment is not None:
            keys(environment, "variant", "environment")
            variant = environment["variant"]
        scattering = rt.pop("scattering_coefficient")
        threshold = rt.pop("power_threshold_db")
        rt["samples_per_src"] = rt["samples_per_src"][directions[0]]
        config["rt"] = {
            "source": "native",
            "scene_xml": None if settings["scene_xml"] is None else str(settings["scene_xml"]),
            "variant": variant,
            "scattering_coefficient": scattering,
            "power_threshold_db": threshold,
            "solver": rt,
        }
    if mode == "hybrid":
        config["rt"]["probe_antenna"] = deepcopy(HYBRID_RT_PROBE)
    return config


def save_run_config(path, settings, config, *, links=None, positions=None, trajectory_config=None):
    """Save the actual link/direction configurations. Position arrays are [time, xyz] in meters. Map endpoint names to satellite/ue roles; each model call still processes one link. Omit links to retain the legacy single-link interface."""
    import json
    import re

    bases = {None: config}
    endpoint_devices = notebook_devices(settings) if "endpoints" in settings else None
    if links is not None:
        if not links or not positions:
            raise ValueError("至少需要一顆衛星及一個 UE，請先執行 A 或 B。")
        for name, points in positions.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name):
                raise ValueError("端點名稱請用英數字，例如 sat1、ue1。")
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
                raise ValueError(f"{name} 的位置需要是有限值的 [時間點, xyz] 陣列，單位 m。")
        if len({len(points) for points in positions.values()}) != 1:
            raise ValueError("所有端點必須使用同一組時間點。")
        bases = {}
        for link, names in links.items():
            if set(names) != {"satellite", "ue"} or len(set(names.values())) != 2:
                raise ValueError("每條鏈路需要不同名稱的衛星與 UE。")
            if link != f"{names['satellite']}_{names['ue']}":
                raise ValueError("鏈路名稱需使用 衛星名稱_UE名稱。")
            current = deepcopy(config)
            current["devices"] = {}
            for role, name in names.items():
                if endpoint_devices is not None:
                    if settings["endpoints"][name]["kind"] != role:
                        raise ValueError(f"{name} 的 satellite / ue 角色與鏈路不符。")
                    device = deepcopy(endpoint_devices[name])
                else:
                    device = deepcopy(config["devices"].get(role, {
                        "array_offsets_m": [[0.0, 0.0, 0.0]],
                        "polarization": settings["polarization"],
                        "pattern": "iso", "orientation_deg": [0.0, 0.0, 0.0],
                    }))
                device["position"] = np.asarray(positions[name][0], dtype=float).tolist()
                current["devices"][role] = device
            bases[link] = current

    configs = {}
    for link, base in bases.items():
        configs[link] = {}
        for direction in settings["directions"]:
            current = deepcopy(base)
            current["direction"] = direction
            if current["mode"] != "3gpp":
                current["rt"]["solver"]["samples_per_src"] = settings["rt"]["samples_per_src"][direction]
            configs[link][direction] = validate_config(current)
    if links is None:
        configs = configs[None]
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("config.json 請存放在 hybrid_simulator/results 內。")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"settings": settings, "channels": configs, "trajectory": trajectory_config}
    if links is not None:
        payload["links"] = links
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False),
                    encoding="utf-8")
    return configs

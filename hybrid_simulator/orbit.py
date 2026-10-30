"""Trajectory geometry: TLE/SGP4 to ITRS/ECEF to fixed city ENU. Sampling does not download TLEs, move the city or generate channels. Users supply TLE and origin; Skyfield uses built-in time data without shared cache writes. Explicit fetch_tle is a separate opt-in download."""

from datetime import datetime, timedelta, timezone
import hashlib
import numpy as np
from .topology import earth_reference, scene_positions


def validate_trajectory(config):
    """Validate trajectory support before expensive RT without implicit mode changes."""
    from .config import keys, vector

    tr = config.get("trajectory")
    if tr is None:
        return
    keys(
        tr,
        "model duration_s interval_s satellite_velocity_m_s ue_velocity_m_s orbit start_utc max_elevation_change_deg",
        "trajectory",
    )
    if tr["model"] not in ("independent_snapshots", "fixed_scatterer_segment"):
        raise ValueError(
            "trajectory.model must be independent_snapshots or fixed_scatterer_segment"
        )
    if not config.get("allow_experimental"):
        raise ValueError("Trajectory requires allow_experimental=true")
    if config["rt"]["source"] != "native":
        raise ValueError("Moving geometry requires fresh native RT, not a fixed cache")
    for field in ("duration_s", "interval_s"):
        value = tr[field]
        if (
            isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
            or (field == "interval_s" and value == 0)
        ):
            raise ValueError("Invalid trajectory duration/interval")
    for name in ("satellite", "ue"):
        vector(tr.get(name + "_velocity_m_s", [0, 0, 0]), name + " ENU velocity")
    if tr.get("start_utc") is not None:
        utc_datetime(tr["start_utc"])
    orbit = tr.get("orbit", {"kind": "linear"})
    keys(orbit, "kind tle_lines tle_file max_epoch_age_days", "trajectory.orbit")
    if orbit["kind"] == "linear":
        keys(orbit, "kind", "linear orbit")
    elif orbit["kind"] == "tle":
        if not tr.get("start_utc") or not config["coordinates"].get(
            "origin_geodetic_deg_m"
        ):
            raise ValueError(
                "TLE requires start_utc and the actual city origin_geodetic_deg_m"
            )
        if ("tle_lines" in orbit) == ("tle_file" in orbit):
            raise ValueError("Specify exactly one of tle_lines / tle_file")
        age = orbit.get("max_epoch_age_days")
        if age is None or isinstance(age, bool) or not np.isfinite(age) or age <= 0:
            raise ValueError(
                "TLE requires an explicit positive max_epoch_age_days policy"
            )
        if "satellite_velocity_m_s" in tr:
            raise ValueError(
                "TLE derives satellite velocity; remove satellite_velocity_m_s"
            )
        _tle_lines(orbit)
    else:
        raise ValueError("orbit.kind must be linear or tle")
    if tr["model"] == "fixed_scatterer_segment":
        validate_temporal_mode(config)
        span = tr.get("max_elevation_change_deg", 1.0)
        if isinstance(span, bool) or not np.isfinite(span) or span <= 0:
            raise ValueError("max_elevation_change_deg must be positive")
    elif "max_elevation_change_deg" in tr:
        raise ValueError(
            "max_elevation_change_deg applies only to fixed_scatterer_segment"
        )


def validate_temporal_mode(config):
    """Apply the same restrictions to explicit API state and notebook trajectories."""
    if not config.get("allow_experimental"):
        raise ValueError("Persistent temporal state requires allow_experimental=true")
    if config["rt"]["source"] != "native":
        raise ValueError("Persistent temporal state requires fresh native RT")
    if config["mode"] != "hybrid" or config["hybrid"]["backend"] != "standard":
        raise ValueError(
            "fixed_scatterer_segment requires mode=hybrid, backend=standard"
        )
    unsupported = [
        k
        for k in ("diffuse_reflection", "refraction", "diffraction", "edge_diffraction")
        if config["rt"]["solver"].get(k, False)
    ]
    if unsupported:
        raise NotImplementedError(
            "Persistent identity currently supports LOS/specular only; disable "
            + ", ".join(unsupported)
        )
    if any(
        config["ionosphere"][k]
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
            "Temporal medium consistency is not validated; fixed-scatterer segments currently require ionosphere effects off"
        )


def utc_datetime(value):
    """Reject timezone-naive timestamps to avoid interpreting local time as UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("start_utc must include Z or an explicit UTC offset")
    return dt.astimezone(timezone.utc)


def _tle_lines(orbit):
    if "tle_file" in orbit:
        from .config import input_path

        lines = input_path(orbit["tle_file"]).read_text(encoding="ascii").splitlines()
    else:
        lines = orbit["tle_lines"]
    lines = [s.rstrip() for s in lines if s.strip()]
    if len(lines) == 3:
        lines = lines[1:]
    if (
        len(lines) != 2
        or not lines[0].startswith("1 ")
        or not lines[1].startswith("2 ")
    ):
        raise ValueError("Provide exactly one two-line TLE (optional name line)")
    if lines[0][2:7] != lines[1][2:7]:
        raise ValueError("The two TLE lines identify different satellites")
    # TLE is fixed-width. Catch accidental pasted spaces/truncation/checksum errors.
    for line in lines:
        if len(line) != 69 or not line[-1].isdigit():
            raise ValueError("TLE lines must have 69 columns including checksum")
        checksum = sum(int(x) if x.isdigit() else int(x == "-") for x in line[:68]) % 10
        if checksum != int(line[-1]):
            raise ValueError("TLE checksum mismatch")
    return lines


def sample_positions(config):
    """Return scene positions/velocities in m and m/s, UTC and orbit provenance. Linear motion retains configured states; TLE replaces satellite positions while UE motion remains ENU. Use simultaneous endpoints without light-time or Sagnac corrections."""
    tr = config["trajectory"]
    orbit = tr.get("orbit", {"kind": "linear"})
    times = (
        np.arange(int(np.floor(tr["duration_s"] / tr["interval_s"])) + 1)
        * tr["interval_s"]
    )
    # TLE replaces satellite positions; do not validate separation using the unused manual position.
    initial = scene_positions(config, require_separation=orbit["kind"] != "tle")
    velocities = {
        n: np.asarray(tr.get(n + "_velocity_m_s", [0, 0, 0]), float)
        for n in ("satellite", "ue")
    }
    positions = {
        n: initial[n][None, :] + times[:, None] * velocities[n] for n in initial
    }
    speed = {
        n: np.broadcast_to(velocities[n], positions[n].shape).copy() for n in initial
    }
    start = utc_datetime(tr["start_utc"]) if tr.get("start_utc") else None
    dates = [
        (start + timedelta(seconds=float(t))).isoformat() if start else None
        for t in times
    ]
    provenance = {
        "kind": orbit["kind"],
        "position_frame": "scene_enu",
        "position_unit": "m",
        "velocity_unit": "m/s",
        "same_epoch_endpoints": True,
    }
    if orbit["kind"] == "tle":
        try:
            from skyfield.api import EarthSatellite, load
            from skyfield.framelib import itrs
        except ImportError as exc:
            raise ImportError(
                "TLE needs the optional skyfield/sgp4 dependencies; linear motion does not"
            ) from exc
        lines = _tle_lines(orbit)
        ts = load.timescale(builtin=True)
        satellite = EarthSatellite(*lines, name="configured_satellite", ts=ts)
        t = ts.from_datetimes([start + timedelta(seconds=float(s)) for s in times])
        ages = np.abs(t - satellite.epoch)
        if np.max(ages) > orbit["max_epoch_age_days"]:
            raise ValueError(
                "TLE exceeds max_epoch_age_days; provide an element set near the simulation date"
            )
        state = satellite.at(t)
        if any(message is not None for message in np.atleast_1d(state.message)):
            raise ValueError(f"SGP4 propagation failed: {state.message}")
        xyz, velocity = state.frame_xyz_and_velocity(itrs)
        origin, rotation = earth_reference(config["coordinates"])
        positions["satellite"] = (xyz.m.T - origin) @ rotation
        speed["satellite"] = velocity.m_per_s.T @ rotation
        provenance.update(
            tle_lines=lines,
            tle_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            tle_epoch_utc=satellite.epoch.utc_iso(),
            max_epoch_age_days=float(np.max(ages)),
            earth_frame="Skyfield ITRS; built-in timescale, no external polar-motion table",
            origin_geodetic_deg_m=config["coordinates"]["origin_geodetic_deg_m"],
        )
    if not all(np.isfinite(v).all() for v in positions.values()):
        raise ValueError("Nonfinite trajectory position")
    if np.any(np.linalg.norm(positions["satellite"] - positions["ue"], axis=1) == 0):
        raise ValueError("Trajectory TX and RX cannot be coincident")
    return times, positions, speed, dates, provenance


def fetch_tle(norad_id, *, directory=None):
    """Explicitly download a NORAD TLE and save source text and retrieval time; do not update the channel. Fail rather than switching satellites. Source: https://celestrak.org/NORAD/documentation/gp-data-formats.php"""
    import json
    from pathlib import Path
    from urllib.request import urlopen
    from .config import ROOT

    identifier = str(norad_id)
    if not identifier.isdecimal() or not 1 <= int(identifier) <= 99999:
        raise ValueError("TLE 格式限五位 NORAD 編號；更大的編號需要另外接 OMM。")
    folder = ROOT / "results/tle" if directory is None else Path(directory).resolve()
    if not folder.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("TLE 與取得紀錄請保存於本專案 results。")
    url = (
        "https://celestrak.org/NORAD/elements/gp.php?CATNR="
        + identifier
        + "&FORMAT=TLE"
    )
    with urlopen(url, timeout=30) as response:
        text = response.read().decode("ascii")
    lines = _tle_lines({"tle_lines": text.splitlines()})
    if int(lines[0][2:7]) != int(identifier):
        raise ValueError("回傳 TLE 的編號與要求不符。")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identifier + ".tle")
    path.write_text(text, encoding="ascii")
    (folder / (identifier + ".json")).write_text(
        json.dumps(
            {
                "url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

"""Coordinate and direction handling; array offsets are already rotated into city ENU meters."""

import numpy as np


def earth_reference(coordinates):
    lon, lat, h = coordinates["origin_geodetic_deg_m"]
    lo, la = np.deg2rad([lon, lat])
    # WGS84 geodetic to ECEF without requiring pyproj for simple RT runs.
    a, e2 = 6378137.0, 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    origin = np.array(
        [
            (n + h) * np.cos(la) * np.cos(lo),
            (n + h) * np.cos(la) * np.sin(lo),
            (n * (1 - e2) + h) * np.sin(la),
        ]
    )
    rotation = np.array(
        [
            [-np.sin(lo), -np.sin(la) * np.cos(lo), np.cos(la) * np.cos(lo)],
            [np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la) * np.sin(lo)],
            [0, np.cos(la), np.sin(la)],
        ]
    )
    return origin, rotation


def scene_positions(config, *, require_separation=True):
    geo = config["coordinates"]
    scale = 1000 if geo["position_unit"] == "km" else 1
    out = {
        k: np.asarray(v["position"], float) * scale
        for k, v in config["devices"].items()
    }
    if geo["frame"] == "ecef":
        origin, r = earth_reference(geo)
        out = {k: r.T @ (v - origin) for k, v in out.items()}
    if require_separation and np.linalg.norm(out["satellite"] - out["ue"]) == 0:
        raise ValueError("TX and RX cannot be coincident")
    return out


def elevation_deg(positions):
    d = positions["satellite"] - positions["ue"]
    return float(np.rad2deg(np.arctan2(d[2], np.linalg.norm(d[:2]))))


def endpoint_names(direction):
    return ("satellite", "ue") if direction == "downlink" else ("ue", "satellite")

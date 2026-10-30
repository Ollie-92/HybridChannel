"""Common-medium corrections and Faraday rotation; preserve existing formulas."""

from dataclasses import dataclass
from datetime import datetime
import math
import numpy as np
from scipy.constants import elementary_charge, epsilon_0, m_e, c
from .config import input_path
from .topology import earth_reference, endpoint_names

FARADAY_SI = elementary_charge**3 / (8 * np.pi**2 * epsilon_0 * m_e**2 * c)


@dataclass(frozen=True)
class PropagationCorrection:
    """Excess propagation quantities at one carrier, excluding geometric L/c. Optical excess is usually negative; group excess positive. Absorption is power loss in dB; scintillation is complex field gain. None means unavailable, not zero."""

    frequency_hz: float
    provenance: str
    optical_excess_m: float | None = None
    group_excess_s: float | None = None
    absorption_loss_db: float | None = None
    scintillation_gain: complex | None = None

    def __post_init__(self):
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("Correction frequency must be positive finite Hz")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Every correction requires model/data provenance")
        for name in ("optical_excess_m", "group_excess_s", "absorption_loss_db"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
        if self.group_excess_s is not None and self.group_excess_s < 0:
            raise ValueError(
                "This cold-plasma adapter requires nonnegative group excess"
            )
        if self.absorption_loss_db is not None and self.absorption_loss_db < 0:
            raise ValueError("Passive absorption cannot have negative loss")
        if self.scintillation_gain is not None:
            gain = complex(self.scintillation_gain)
            if not math.isfinite(gain.real) or not math.isfinite(gain.imag):
                raise ValueError("Scintillation field gain must be finite")

    @classmethod
    def from_hamiltonian_trace(cls, trace, *, provenance):
        """Use directly integrated small increments rather than subtracting two large distances to recover delay."""
        from scipy.constants import c

        if trace.mode != "isotropic":
            raise NotImplementedError(
                "This hybrid adapter does not assemble magnetoionic O/X modes"
            )
        return cls(
            frequency_hz=float(trace.frequency_hz),
            provenance=provenance,
            optical_excess_m=float(trace.phase_excess_m),
            group_excess_s=float(trace.group_excess_m) / c,
            absorption_loss_db=float(trace.absorption_loss_db),
        )


def phase_advance_rad(optical_excess_m, frequency_hz):
    """Convert excess optical path in meters to phase in radians, using exp(+j phase); exclude vacuum geometric path."""
    if (
        not math.isfinite(optical_excess_m)
        or not math.isfinite(frequency_hz)
        or frequency_hz <= 0
    ):
        raise ValueError(
            "Expected a finite optical excess and positive frequency in Hz"
        )
    return -2 * math.pi * frequency_hz * optical_excess_m / c


def add_group_delay(delays_s, group_excess_s):
    """Add common-medium excess group delay; inputs and outputs are seconds."""
    return np.asarray(delays_s) + group_excess_s


def apply_phase_advance(coefficients, optical_excess_m, frequency_hz):
    """Apply medium phase to complex coefficients without counting group-delay carrier phase twice."""
    angle = phase_advance_rad(optical_excess_m, frequency_hz)
    return np.asarray(coefficients) * np.exp(1j * angle)


def apply_absorption(coefficients, loss_db):
    """Convert power loss in dB to field amplitude and apply it to complex coefficients."""
    return np.asarray(coefficients) * 10 ** (-loss_db / 20)


def apply_scintillation(coefficients, complex_gain):
    """Apply externally supplied complex scintillation gain without generating a scintillation model."""
    return np.asarray(coefficients) * complex_gain


def rotation(psi_rad):
    "Build a two-dimensional Faraday rotation matrix from psi in radians."
    psi = np.asarray(psi_rad, dtype=float)
    co, si = np.cos(psi), np.sin(psi)
    return np.stack([co, -si, si, co], axis=-1).reshape(psi.shape + (2, 2))


def integrate_magnetized_path(path_ecef_m, ne_m3, b_ecef_t):
    """Integrate electron density and magnetic-field projection along an ECEF polyline: positions in meters, Ne in m^-3, B in tesla; nodes follow propagation direction."""
    p = np.asarray(path_ecef_m, float)
    ne = np.asarray(ne_m3, float)
    b = np.asarray(b_ecef_t, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        raise ValueError("path_ecef_m must have shape (L>=2,3), ordered TX to RX")
    if ne.shape != (len(p),) or b.shape != p.shape:
        raise ValueError(
            "Ne [m^-3] and B [tesla, ECEF] must be supplied at every path node"
        )
    if not all(np.all(np.isfinite(x)) for x in (p, ne, b)) or np.any(ne < 0):
        raise ValueError("Physical samples must be finite with nonnegative Ne")
    dl = np.diff(p, axis=0)
    if np.any(np.linalg.norm(dl, axis=1) == 0):
        raise ValueError("Duplicate path nodes are not allowed")
    nb = ne[:, None] * b
    return float(np.sum(0.5 * (nb[:-1] + nb[1:]) * dl))


def physical_integral(cfg, context=None):
    """Validate Faraday endpoints, time and units before integration. Missing physical data raises an error rather than using demonstration constants."""
    ctx = context or {}
    kind = cfg.get("kind", "providers")
    if kind == "sampled_path":
        p = np.asarray(cfg["path_ecef_m"], float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
            raise ValueError("sampled_path needs at least two ECEF nodes")
        for key, endpoint in (("tx_ecef_m", p[0]), ("rx_ecef_m", p[-1])):
            if key in ctx and not np.allclose(ctx[key], endpoint, rtol=0, atol=1):
                raise ValueError(
                    "Physical path endpoints do not match this satellite/UE link"
                )
        if "time_utc" in ctx:
            if "time_utc" not in cfg:
                raise ValueError("Sampled physical path must include time_utc")
            expected = datetime.fromisoformat(ctx["time_utc"].replace("Z", "+00:00"))
            actual = datetime.fromisoformat(cfg["time_utc"].replace("Z", "+00:00"))
            if actual.tzinfo is None or actual != expected:
                raise ValueError("Sampled physical path time does not match link time")
        return integrate_magnetized_path(p, cfg["ne_m3"], cfg["b_ecef_t"])
    if kind not in ("providers", "uniform_synthetic"):
        raise ValueError("Unknown physical environment kind")
    if "tx_ecef_m" not in ctx or "rx_ecef_m" not in ctx:
        raise ValueError("physical mode requires actual TX/RX ECEF positions")
    count = int(cfg.get("num_samples", 1025))
    if count < 2:
        raise ValueError("num_samples must be >=2")
    p = np.linspace(ctx["tx_ecef_m"], ctx["rx_ecef_m"], count)
    if kind == "uniform_synthetic":
        # Explicit, repeatable test environment; never called by default.
        ne = np.full(count, float(cfg["ne_m3"]))
        if "altitude_bounds_m" in cfg:
            from pyproj import Transformer

            alt = Transformer.from_crs(4978, 4979, always_xy=True).transform(*p.T)[2]
            lo, hi = cfg["altitude_bounds_m"]
            ne *= (np.asarray(alt) >= lo) & (np.asarray(alt) <= hi)
        b = np.broadcast_to(cfg["b_ecef_t"], p.shape)
    else:
        if not all(k in cfg for k in ("density_model", "magnetic_model")):
            raise ValueError(
                "physical requires density_model and magnetic_model; no fallback"
            )
        if "time_utc" not in ctx:
            raise ValueError("Physical providers require time_utc")
        when = datetime.fromisoformat(ctx["time_utc"].replace("Z", "+00:00"))
        ne = cfg["density_model"].electron_density_ecef(p, when)
        b = cfg["magnetic_model"].magnetic_field_ecef(p, when)
    return integrate_magnetized_path(p, ne, b)


def faraday_state(cfg=None, context=None, *, direction="downlink"):
    """Prepare disabled, reference-angle or physical Ne/B Faraday state; antenna projection is performed later."""
    if direction not in ("downlink", "uplink"):
        raise ValueError("Faraday direction must be downlink or uplink")
    cfg = dict(cfg or {})
    mode = cfg.get("mode", "off")
    if mode == "off":
        coefficient = 0.0
    elif mode == "3gpp_reference":
        coefficient = np.deg2rad(108.0) * 1e18  # rad Hz^2
    elif mode == "physical":
        coefficient = FARADAY_SI * physical_integral(cfg.get("physical", {}), context)
        if direction == "uplink":
            coefficient = -coefficient  # Propagating basis -> satellite basis.
    else:
        raise ValueError("faraday.mode must be off, 3gpp_reference or physical")
    compensation = cfg.get("compensation", "none")
    if compensation not in ("none", "ideal_tx"):
        raise ValueError("Only none or explicitly ideal_tx compensation is supported")
    if mode != "off" and direction == "uplink" and compensation == "ideal_tx":
        raise NotImplementedError(
            "Uplink ideal_tx compensation cannot commute through the urban "
            "polarization matrix; use compensation='none'"
        )
    return {
        "mode": mode,
        "coefficient_rad_hz2": coefficient,
        "compensation": compensation,
        "direction": direction,
        "rotation_basis": "satellite_outgoing_gcs_theta_phi",
        "propagation_coefficient_rad_hz2": (
            coefficient if direction == "downlink" else -coefficient
        ),
        "operator_order": "P_DL @ F" if direction == "downlink" else "F @ P_UL",
        "scope": "common ionospheric segment; rotation evaluated at carrier frequency",
        "environment": (
            cfg.get("physical", {}).get("kind") if mode == "physical" else None
        ),
    }


def angle_at_frequency(state, frequency_hz):
    "Return rotation angle in radians at the specified frequency in Hz."
    f = np.asarray(frequency_hz, float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequency must be positive Hz")
    return state["coefficient_rad_hz2"] / f**2


def dispersion_on_path(vertices_ecef_m, frequency_hz, model, time, *, provenance):
    "Integrate NeQuick2 density along a supplied ECEF polyline for excess group delay and optical path without changing geometry."
    from .third_party.plasma_models.delay import compute_group_delay
    from .third_party.plasma_models.phase import compute_phase_advance

    points = np.asarray(vertices_ecef_m, dtype=float)
    delay = compute_group_delay(
        points, points[0], points[-1], frequency_hz, model, time
    )
    phase = compute_phase_advance(points, frequency_hz, model, time)
    return PropagationCorrection(
        frequency_hz=frequency_hz,
        provenance=provenance,
        group_excess_s=delay.medium_excess_group_delay_s,
        optical_excess_m=phase.medium_phase_advance_rad
        / (2 * math.pi * frequency_hz / c),
    )


def ecef_endpoints(config, positions):
    if "origin_geodetic_deg_m" not in config["coordinates"]:
        raise ValueError(
            "Physical ionosphere requires a verified scene geodetic origin"
        )
    origin, r = earth_reference(config["coordinates"])
    tx, rx = endpoint_names(config["direction"])
    return origin + r @ positions[tx], origin + r @ positions[rx]


def resolve_correction(config, positions, run_directory):
    """Return medium increments, excluding RT geometric distance and free-space loss."""
    cfg = config["ionosphere"]["correction"]
    if cfg is None:
        return None
    f = config["radio"]["frequency_hz"]
    if cfg["kind"] == "nequick2_straight_common":
        from .third_party.nequick2 import create_model

        if run_directory is None:
            raise ValueError(
                "NeQuick2 requires an isolated result directory for its runtime"
            )
        if any(config["ionosphere"][k] for k in ("absorption", "scintillation")):
            raise NotImplementedError(
                "NeQuick2 density alone does not predict absorption or scintillation"
            )
        when = datetime.fromisoformat(cfg["time_utc"].replace("Z", "+00:00"))
        if when.tzinfo is None:
            raise ValueError("NeQuick2 time must include UTC timezone")
        count = cfg["samples"]
        if type(count) is not int or count < 2:
            raise ValueError("NeQuick2 samples must be >=2")
        tx, rx = ecef_endpoints(config, positions)
        model = create_model(run_directory, flux_f107=cfg["flux_f107"])
        path = np.linspace(tx, rx, count)
        value = dispersion_on_path(path, f, model, when, provenance=cfg["provenance"])
        np.savez_compressed(run_directory / "ionosphere_path.npz", path_ecef_m=path)
        return value
    gain = cfg.get("scintillation_gain_ri")
    if gain is not None and (len(gain) != 2 or not np.all(np.isfinite(gain))):
        raise ValueError("scintillation_gain_ri must contain [real,imag]")
    return PropagationCorrection(
        f,
        cfg["provenance"],
        optical_excess_m=cfg.get("optical_excess_m"),
        group_excess_s=cfg.get("group_excess_s"),
        absorption_loss_db=cfg.get("absorption_loss_db"),
        scintillation_gain=None if gain is None else complex(*gain),
    )


def faraday_inputs(config, positions):
    """Prepare common-medium TX-to-RX data. Reverse position/Ne/B node order for the reverse link without negating magnetic-field vectors."""
    cfg = config["ionosphere"]["faraday"]
    if cfg is None:
        return {"mode": "off"}, None
    compensation = cfg.get("compensation", "none")
    if cfg["mode"] == "reference":
        return {"mode": "3gpp_reference", "compensation": compensation}, None
    tx, rx = ecef_endpoints(config, positions)
    with np.load(input_path(cfg["sampled_path_npz"]), allow_pickle=False) as data:
        physical = {k: data[k].copy() for k in ("path_ecef_m", "ne_m3", "b_ecef_t")}
        if "time_utc" not in data:
            raise ValueError(
                "Faraday sampled_path_npz must record its own time_utc; config time is not sample provenance"
            )
        sample_time = str(data["time_utc"].item())
    points = np.asarray(physical["path_ecef_m"])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Faraday path_ecef_m must have shape (L>=2,3)")
    if (
        physical["ne_m3"].shape != (len(points),)
        or physical["b_ecef_t"].shape != points.shape
    ):
        raise ValueError("Faraday Ne and B must be supplied at every path node")
    same = np.allclose(points[[0, -1]], [tx, rx], rtol=0, atol=1)
    reverse = np.allclose(points[[0, -1]], [rx, tx], rtol=0, atol=1)
    if not same and not reverse:
        raise ValueError("Physical path endpoints do not match this satellite/UE link")
    if not same:
        physical = {key: values[::-1].copy() for key, values in physical.items()}
    physical.update(kind="sampled_path", time_utc=sample_time)
    return {"mode": "physical", "physical": physical, "compensation": compensation}, {
        "tx_ecef_m": tx,
        "rx_ecef_m": rx,
        "time_utc": cfg["time_utc"],
    }


def apply_scalar_effects(h, delays_s, config, correction):
    """Apply common-medium scalars once after projection; Faraday matrices cannot be applied here."""
    ion = config["ionosphere"]
    h, tau = np.array(h, copy=True), np.array(delays_s, copy=True)
    details = {"applied": [], "scope": None}
    if correction is None:
        return h, tau, details
    details.update(
        scope=ion["correction"]["scope_note"], provenance=correction.provenance
    )
    if ion["group_delay"]:
        if correction.group_excess_s is None:
            raise ValueError("group_delay requires group_excess_s")
        tau = add_group_delay(tau, correction.group_excess_s)
        details["group_excess_s"] = correction.group_excess_s
    if ion["phase_advance"]:
        if correction.optical_excess_m is None:
            raise ValueError("phase_advance requires optical_excess_m")
        angle = phase_advance_rad(correction.optical_excess_m, correction.frequency_hz)
        h = apply_phase_advance(h, correction.optical_excess_m, correction.frequency_hz)
        details.update(
            optical_excess_m=correction.optical_excess_m,
            coefficient_phase_increment_rad=angle,
        )
    if ion["absorption"]:
        if correction.absorption_loss_db is None:
            raise ValueError(
                "absorption requires absorption_loss_db; Ne alone is insufficient"
            )
        h = apply_absorption(h, correction.absorption_loss_db)
        details["absorption_loss_db"] = correction.absorption_loss_db
    if ion["scintillation"]:
        if correction.scintillation_gain is None:
            raise ValueError(
                "scintillation requires a measured/modelled complex field gain"
            )
        h = apply_scintillation(h, correction.scintillation_gain)
        details["scintillation_gain_ri"] = [
            correction.scintillation_gain.real,
            correction.scintillation_gain.imag,
        ]
    details["applied"] = [
        k
        for k in ("group_delay", "phase_advance", "absorption", "scintillation")
        if ion[k]
    ]
    return h, tau, details

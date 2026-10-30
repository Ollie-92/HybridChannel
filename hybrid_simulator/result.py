"""Shared RT/Hybrid/OpenNTN outputs: CIR taps and complex CFR with units and provenance."""

from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path
from .config import ROOT


def checked_output(directory):
    path = Path(directory).resolve()
    if not path.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Output must remain inside hybrid_simulator/results")
    return path


@dataclass
class ChannelResult:
    coefficients: np.ndarray  # complex [rx_port, tx_port, path/ray]
    delays_s: np.ndarray  # Seconds; see metadata.timing_reference for the delay convention.
    delay_reference_s: float  # RT geometric reference or native OpenNTN slant range divided by c.
    angles_rad: np.ndarray  # [path/ray, 4], ordered AoA, AoD, ZoA, ZoD.
    sources: np.ndarray  # RT / RC / OpenNTN
    frequency_offsets_hz: np.ndarray
    cfr: np.ndarray  # complex [rx_port,tx_port,frequency]
    metadata: dict = field(default_factory=dict)
    extra_arrays: dict = field(default_factory=dict)
    # Keep native scene/paths only in memory for rendering; exclude them from NPZ/JSON.
    runtime_context: dict = field(default_factory=dict, repr=False)

    @property
    def a(self):
        """Complex [rx_port, tx_port, path, time] coefficients without additional phase."""
        return self.extra_arrays.get("coefficients_time", self.coefficients[..., None])

    @property
    def tau(self):
        """Excess path delays in seconds; see delay_reference_s for the reference."""
        return self.delays_s

    @property
    def h(self):
        """Narrowband [rx_port, tx_port, time] channel at the carrier, coherently summed over paths."""
        return np.sum(self.a, axis=2)

    @property
    def absolute_delays_s(self):
        return self.delays_s + self.delay_reference_s

    def save(self, directory):
        """Save full arrays without pickle, metadata as JSON and readable numerical CSV tables."""
        directory = checked_output(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "channel.npz",
            a=self.a,
            tau=self.tau,
            h=self.h,
            coefficients=self.coefficients,
            delays_s=self.delays_s,
            absolute_delays_s=self.absolute_delays_s,
            delay_reference_s=self.delay_reference_s,
            angles_rad=self.angles_rad,
            sources=self.sources,
            frequency_offsets_hz=self.frequency_offsets_hz,
            cfr=self.cfr,
            **self.extra_arrays
        )
        summary = dict(
            self.metadata,
            coefficients_shape=list(self.coefficients.shape),
            cfr_shape=list(self.cfr.shape),
            ray_count=len(self.delays_s),
        )
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        from .csv_export import export_channel_csv
        export_channel_csv(self, directory)

    @classmethod
    def load(cls, directory):
        """Load completed snapshots without retracing or deserializing pickle/native GPU objects."""
        directory = checked_output(directory)
        names = (
            "coefficients",
            "delays_s",
            "delay_reference_s",
            "angles_rad",
            "sources",
            "frequency_offsets_hz",
            "cfr",
        )
        with np.load(directory / "channel.npz", allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in names}
            extra = {
                k: data[k].copy()
                for k in data.files
                if k not in names and k not in ("absolute_delays_s", "a", "tau", "h")
            }
        arrays["delay_reference_s"] = float(arrays["delay_reference_s"])
        return cls(
            **arrays,
            metadata=json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            ),
            extra_arrays=extra
        )


def offsets(config):
    r = config["radio"]
    return (
        np.linspace(
            -r["bandwidth_hz"] / 2, r["bandwidth_hz"] / 2, r["frequency_points"]
        )
        if r["frequency_points"] > 1
        else np.zeros(1)
    )


def frequency_response(h, delays, frequency_offsets):
    return np.einsum(
        "rtp,pf->rtf",
        h,
        np.exp(-2j * np.pi * delays[:, None] * frequency_offsets[None, :]),
    )

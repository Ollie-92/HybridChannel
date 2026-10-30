"""Shared RT/Hybrid/OpenNTN outputs: CIR taps and complex CFR with units and provenance."""

from dataclasses import dataclass, field
import csv
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


def pdp_rows(result):
    """Sum powers over ports and exactly equal delays, separately per time sample."""
    a = result.a
    sample_times = result.extra_arrays.get("time_s", np.zeros(a.shape[-1]))
    for sample in range(a.shape[-1]):
        power = np.sum(np.abs(a[..., sample]) ** 2, axis=(0, 1))
        valid = np.isfinite(result.tau) & (result.tau >= 0) & (power > 0)
        for delay in np.unique(result.tau[valid]):
            selected = valid & (result.tau == delay)
            total = float(power[selected].sum())
            yield [sample, float(sample_times[sample]), float(delay), float(delay * 1e9),
                   int(selected.sum()), total, float(10 * np.log10(total))]


def export_channel_csv(result, directory):
    """Export every port/path/time coefficient, narrowband matrix, CFR and PDP."""
    a = result.a
    sample_times = result.extra_arrays.get("time_s", np.zeros(a.shape[-1]))
    if len(sample_times) != a.shape[-1]:
        raise ValueError("Time sample count differs from channel coefficient count")
    prefix = [result.metadata.get("link_name", ""), result.metadata.get("direction", ""),
              result.metadata.get("time_s", ""), result.metadata.get("time_utc", "")]
    prefix_names = ["link", "direction", "snapshot_time_s", "snapshot_utc"]

    def write(name, header, rows):
        # UTF-8 with BOM allows Excel to recognize the encoding.
        with (directory / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(prefix_names + header)
            for row in rows:
                writer.writerow(prefix + ["" if isinstance(v, (float, np.floating))
                                         and not np.isfinite(v) else v for v in row])

    def paths():
        for sample in range(a.shape[-1]):
            for rx in range(a.shape[0]):
                for tx in range(a.shape[1]):
                    for path in range(a.shape[2]):
                        value = a[rx, tx, path, sample]
                        yield [sample, float(sample_times[sample]), rx, tx, path,
                               str(result.sources[path]), float(result.tau[path]),
                               float(result.absolute_delays_s[path]),
                               float(value.real), float(value.imag), float(abs(value) ** 2),
                               *map(float, result.angles_rad[path])]

    write("channel_paths.csv", ["sample_index", "sample_offset_s", "rx_port", "tx_port",
          "path_index", "source", "excess_delay_s", "absolute_delay_s", "a_real", "a_imag",
          "power_linear", "aoa_rad", "aod_rad", "zoa_rad", "zod_rad"], paths())
    h = result.h
    write("channel_matrix.csv", ["sample_index", "sample_offset_s", "rx_port", "tx_port",
          "h_real", "h_imag", "power_linear"],
          ([sample, float(sample_times[sample]), rx, tx, float(h[rx, tx, sample].real),
            float(h[rx, tx, sample].imag), float(abs(h[rx, tx, sample]) ** 2)]
           for sample in range(h.shape[-1]) for rx in range(h.shape[0]) for tx in range(h.shape[1])))
    write("pdp.csv", ["sample_index", "sample_offset_s", "excess_delay_s", "excess_delay_ns",
          "path_count", "power_linear_sum", "power_db_sum"], pdp_rows(result))
    # OpenNTN stores [RX, TX, time, frequency]; static channels have one sample.
    cfr = result.extra_arrays.get("cfr_time", result.cfr[:, :, None, :])
    write("cfr.csv", ["sample_index", "sample_offset_s", "rx_port", "tx_port",
          "frequency_offset_hz", "cfr_real", "cfr_imag"],
          ([sample, float(sample_times[sample]), rx, tx, float(frequency),
            float(cfr[rx, tx, sample, k].real), float(cfr[rx, tx, sample, k].imag)]
           for sample in range(cfr.shape[2]) for rx in range(cfr.shape[0])
           for tx in range(cfr.shape[1]) for k, frequency in enumerate(result.frequency_offsets_hz)))
    (directory / "CSV_README.txt").write_text(
        "All indices are zero-based. Ports follow summary.json port_order and polarization_ports.\n"
        "snapshot_time_s identifies the trajectory snapshot; sample_offset_s is relative to that snapshot.\n"
        "channel_paths.csv: every path and port, including zero-power slots; complex values are split.\n"
        "channel_matrix.csv: h=sum(a over paths), then power=abs(h)^2 (coherent).\n"
        "pdp.csv: sum(abs(a)^2) over all ports and paths with exactly equal delays (incoherent).\n"
        "PDP excludes invalid/negative delays and zero-power paths. No delay rounding or new threshold.\n"
        "power_db_sum=10*log10(power_linear_sum), not dBm and not normalized to the peak.\n"
        "cfr.csv: complex frequency response for every port and available time sample.\n"
        "Blank numeric fields mean unavailable/non-finite values, e.g. OpenNTN per-tap angles.\n"
        "NPZ retains original arrays and shapes; JSON retains metadata.\n",
        encoding="utf-8")

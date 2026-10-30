"""Readable channel exports; no propagation or statistical model is changed."""

import csv
import json
import numpy as np


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

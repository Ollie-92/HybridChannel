# Notebook implementation and validation details

For the main installation and usage steps, see the [README](../README.md).


The root [hybrid_simulator.ipynb](../hybrid_simulator.ipynb) is an additional interactive entry point. The original `Hybrid_channel/`, `Config/`, `Scene/`, `tools/`, and `run_simulation.py` remain in place. The new `hybrid_simulator/` package reorganizes and extends the research implementation; it is not a drop-in replacement for the original CLI or its saved results.

From the repository root, in a Python 3.12 environment with a working NVIDIA driver:

```bash
python -m venv .venv-notebook
source .venv-notebook/bin/activate
python -m pip install -r requirements-notebook.txt
python -m jupyterlab hybrid_simulator.ipynb
```

The recorded environment uses Sionna/Sionna RT 1.2.1 and TensorFlow 2.20.0. RT API references use the Sionna RT 1.2.2 documentation with installed-version compatibility checks. The OpenNTN adapter loads the vendored TensorFlow revision `cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2`; it does not install current OpenNTN 2.x or upgrade Sionna. The notebook uses `cuda_ad_mono_polarized` for RT and TensorFlow CPU for OpenNTN. Installation on a fresh machine has not yet been verified.

Run Environment and Configuration, choose **A (fixed positions)** or **B (orbit)**, then run channel generation, output inspection and plotting. Do not execute both geometry alternatives as part of an unconditional Run All. The default uses the existing `Scene/NYC_scene/NYC_sionna.xml`, one enabled satellite/UE pair, downlink, 2 GHz, and 10 million RT samples per source. The ray budget is preserved from the prepared research notebook and can take substantial time; no RT computation is required merely to read the notebook.

| Mode | Channel source | Scope |
|---|---|---|
| `rt` | Native Sionna RT, followed by custom path filtering | Configured endpoint arrays; the configured power threshold is applied |
| `hybrid` | RT-derived and statistical random clusters | Fixed internal 1x1 iso VH RT probe; actual endpoint arrays enter channel synthesis; KB=1 ray delays |
| `3gpp` | Pinned native OpenNTN | Fixed topology, uplink/downlink, regular multi-element arrays; no OpenNTN orbit spatial-consistency implementation |

Edit endpoint `num_rows`, `num_cols`, spacing, polarization and orientation in Configuration. Spacing is in wavelengths and orientation in degrees. A VH element contributes two ports. Fixed-position links are formed from enabled satellite/UE combinations; this does not establish cross-UE spatial correlation. The current notebook orbit example is for one satellite/UE pair. Optional ionosphere corrections default to off; NeQuick2-backed corrections require an external compatible library/data installation that is not bundled here. No end-to-end native bent-ray solver is claimed.

#### Notebook outputs

Snapshots are saved under `hybrid_simulator/results/runs/<run>/<link>/<direction>/fixed` or `orbit_XXXX`. Reusing a directory overwrites matching files.

| File | Meaning |
|---|---|
| `channel_paths.csv` | Every RX/TX port, ray/tap and time sample; complex coefficients split into real/imaginary columns |
| `channel_matrix.csv` | Coherent sum of paths at the carrier for every port pair/time sample |
| `pdp.csv` | Unnormalized incoherent power sum across all ports and exactly equal delays |
| `cfr.csv` | Complex frequency response by port, time and frequency offset |
| `inspection_summary.csv`, `port_metrics.csv`, `source_summary.csv` | Written when Inspect outputs runs: selected-sample summary, per-port power/phase/delay statistics, source counts |
| `channel.npz`, `summary.json`, `CSV_README.txt` | Full numerical arrays, metadata and field definitions |

The notebook per-ray PDP follows the original plotting convention: select `RX_PORT`/`TX_PORT`, show individual rays versus excess delay, and normalize each panel to its strongest ray (0 dB). Defaults are snapshot 0 and time sample 0; `SNAPSHOT_INDEX=-1` selects the last snapshot. Equal-delay rays overlap without summation. The corresponding `ray_pdp_*.png` and `.csv` are saved in the link directory. This plot is distinct from `pdp.csv`; normalized panels do not compare absolute gains between port pairs. Gains are dimensionless channel gains in dB, not received power in dBm. OpenNTN outputs are aggregated native taps, not individual unresolved sub-rays.

#### Validation and limitations

Twelve native OpenNTN comparisons passed on the recorded server, covering single-element regression, MIMO, mixed polarization, uplink/downlink, LOS/NLOS and Doppler options. Raw CIR/delays/CFR and mapped public CIR/CFR/H matched exactly in those cases. See [validation details](../hybrid_simulator/OPENNTN_MIMO.md) and the [recorded comparison table](../hybrid_simulator/validation/openntn_mimo/validation.csv). Re-run with:

```bash
python -B hybrid_simulator/validate_openntn_mimo.py
```

This validates the adapter against the pinned upstream for the tested cases. It does not validate all physical models, complete paper-figure reproduction, or numerical equivalence between the new package and the preserved original CLI. [The module guide](../hybrid_simulator/MODULE_GUIDE.md) describes the new files. Restart the notebook kernel after changing module code.


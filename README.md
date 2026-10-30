# HybridChannel

Hybrid RT–RC satellite channel simulator based on the map-based hybrid model in **3GPP TR 38.901 (v19), Clause 8**.

Sionna RT computes propagation paths in a 3D scene. Statistical random clusters supplement those paths through delay generation, duplicate-cluster removal, and power anchoring. The simulator produces complex MIMO channel impulse responses along a configurable straight satellite trajectory.

[Quick start](#quick-start) · [Notebook simulator](#notebook-simulator) · [Experiments](#experiments) · [Model details](docs/model.md)

## Channel Generation

![Map-based hybrid channel generation procedure](docs/channel_flowchart.png)

Source: **3GPP TR 38.901 V19.1.0, Clause 8.4, Figure 8.4-1**, “Channel coefficient generation procedure.” [ETSI PDF](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf#page=151).

The figure describes the reference procedure. This implementation provides Hybrid RT–RC generation and a fixed-scatterer RC-only option. In the original CLI implementation, full Hybrid temporal consistency and TLE propagation are not implemented; see [model details](docs/model.md) for equations and assumptions. The additional notebook supports TLE position sampling and a restricted experimental temporal mode; these do not establish full Hybrid spatial/temporal consistency or numerical equivalence to the original CLI.

## Quick Start

The recorded research environment uses **Ubuntu 24.04, Python 3.12, Sionna RT 1.2.1, and an NVIDIA RTX 5080**. Install Git, Python with `venv` support, and a working NVIDIA driver first. The original CLI instructions below remain available alongside the notebook entry point.

```bash
git clone https://github.com/Ollie-92/HybridChannel.git
cd HybridChannel
bash tools/install.sh
source .venv/bin/activate
python tools/check_installation.py --run-example
```

`tools/install.sh` creates `.venv` and installs the pinned dependencies. The check runs a three-position ground-plane example and generates channel data, plots, and CSV metrics. On success it prints `Example check passed:` followed by the output directory. Fresh-environment RT execution and numerical reproduction have not yet been validated.

If installation fails, check that Python 3.12 is available. For GPU/backend errors, check `nvidia-smi` and the [Sionna RT prerequisites](https://nvlabs.github.io/sionna/rt/installation.html). For simulation errors, read `run.log` in the printed `RUN_DIR`.

### Notebook simulator

The root [hybrid_simulator.ipynb](hybrid_simulator.ipynb) is an additional interactive entry point. The original `Hybrid_channel/`, `Config/`, `Scene/`, `tools/`, and `run_simulation.py` remain in place. The new `hybrid_simulator/` package reorganizes and extends the research implementation; it is not a drop-in replacement for the original CLI or its saved results.

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

Twelve native OpenNTN comparisons passed on the recorded server, covering single-element regression, MIMO, mixed polarization, uplink/downlink, LOS/NLOS and Doppler options. Raw CIR/delays/CFR and mapped public CIR/CFR/H matched exactly in those cases. See [validation details](hybrid_simulator/OPENNTN_MIMO.md) and the [recorded comparison table](hybrid_simulator/validation/openntn_mimo/validation.csv). Re-run with:

```bash
python -B hybrid_simulator/validate_openntn_mimo.py
```

This validates the adapter against the pinned upstream for the tested cases. It does not validate all physical models, complete paper-figure reproduction, or numerical equivalence between the new package and the preserved original CLI. [The module guide](hybrid_simulator/MODULE_GUIDE.md) describes the new files. Restart the notebook kernel after changing module code.

## Experiments

The following original CLI experiments remain available. Run commands from the repository root. Each JSON file contains the scene path and simulation parameters.

| Configuration | Scene | Channel | Positions |
|---|---|---|---|
| [example.json](Config/example.json) | Generated ground plane | Hybrid RT–RC | 3 |
| [nyc_hybrid.json](Config/nyc_hybrid.json) | NYC | Hybrid RT–RC | 5 |
| [nyc_consistent_rc.json](Config/nyc_consistent_rc.json) | NYC | Fixed-scatterer RC-only, with RT power anchoring | 5 |

```bash
python run_simulation.py --config Config/nyc_hybrid.json

# Fixed-scatterer RC-only
python run_simulation.py --config Config/nyc_consistent_rc.json

# Plot one completed run
python tools/plot_results.py Result/<run-directory> --trusted-local-output
```

Replace `<run-directory>` with the directory printed by the simulation. The NYC settings use 28 GHz and a 600 km scene altitude. The scene XML and referenced meshes are included in [Scene/NYC_scene](Scene/NYC_scene).

Edit the `cli` fields in a configuration to change frequency, antennas, receiver coordinates, trajectory, seed, or RT settings. For another scene, add `--scene /absolute/path/to/scene.xml` to the simulation command and adjust its geometry settings.

Each run saves CIR coefficients, delays, geometry, configuration, package versions, and logs under `Result/`. Analysis adds PDP/CFR plots and CSV metrics in `analysis/`. Only analyze trusted files: the saved channel uses NumPy object arrays.

## Repository Structure

| Path | Purpose |
|---|---|
| [hybrid_simulator.ipynb](hybrid_simulator.ipynb) | Additional interactive notebook entry point |
| [hybrid_simulator/](hybrid_simulator) | New RT/Hybrid/OpenNTN package, CSV exports and pinned OpenNTN source |
| [requirements-notebook.txt](requirements-notebook.txt) | Notebook additions to the original pinned environment |
| [Hybrid_channel/](Hybrid_channel) | RT path extraction, random clusters, channel assembly, and trajectory simulation |
| [Config/](Config) | Experiment settings |
| [Scene/](Scene) | NYC scene XML and meshes |
| [docs/](docs) | Model equations and flowchart |
| [run_simulation.py](run_simulation.py) | Simulation entry point |
| [tools/](tools) | Installation, environment checks, and result plotting |
| [requirements.txt](requirements.txt) | Pinned dependencies |

For the role of each channel module, see the [code guide](docs/model.md#code-guide).

## References

- 3GPP TR 38.901 V19.1.0, *Study on channel model for frequencies from 0.5 to 100 GHz*, Release 19, October 2025. [ETSI PDF](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf).
- [Sionna RT](https://github.com/NVlabs/sionna-rt), ray-tracing library.

### OpenNTN

T. Düe, M. Vakilifard, C. Bockelmann, D. Wübben, and A. Dekorsy, “OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations,” *International Workshop on Smart Antennas (WSA)*, Erlangen, Germany, September 16–18, 2025. [Paper](https://www.ant.uni-bremen.de/sixcms/media.php/102/15183/OpenNTN_An_OpenSource_Framework_For_NTN_Simulations.pdf) · [Code](https://github.com/ant-uni-bremen/OpenNTN).

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{OpenNTNPaper,
  author    = {T. D{\"u}e and M. Vakilifard and C. Bockelmann and D. W{\"u}bben and A. Dekorsy},
  title     = {{OpenNTN}: An Open-Source Framework for Non-Terrestrial Network Channel Simulations},
  booktitle = {International Workshop on Smart Antennas (WSA)},
  address   = {Erlangen, Germany},
  year      = {2025},
  month     = sep,
  url       = {https://www.ant.uni-bremen.de/sixcms/media.php/102/15183/OpenNTN_An_OpenSource_Framework_For_NTN_Simulations.pdf}
}
```

</details>

## Publication

**An Open-Source Hybrid Ray-Tracing and 3GPP Channel Simulator for Non-Terrestrial Network**

[arXiv:2610.37536](https://arxiv.org/abs/2610.37536)

If you use HybridChannel, its simulation results, or its implementation methods in academic work, you must cite this paper.

## Usage Notice

The authors permit academic research use of the original HybridChannel code provided that the associated paper above is correctly cited. All other rights are reserved. Redistribution, modification, or commercial use requires prior written permission from the authors, unless separately authorized.

Academic work using this implementation must cite *An Open-Source Hybrid Ray-Tracing and 3GPP Channel Simulator for Non-Terrestrial Network* ([arXiv:2610.37536](https://arxiv.org/abs/2610.37536)). Please also include the repository URL, https://github.com/Ollie-92/HybridChannel, and identify the commit used for reproducibility.

This notice applies only to original project material. Third-party software, scene assets, and reproduced figures remain subject to their respective rights and licenses.

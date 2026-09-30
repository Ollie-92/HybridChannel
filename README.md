# HybridChannel

Hybrid RT–RC satellite channel simulator based on the map-based hybrid model in **3GPP TR 38.901 (v19), Clause 8**.

Sionna RT computes propagation paths in a 3D scene. Statistical random clusters supplement those paths through delay generation, duplicate-cluster removal, and power anchoring. The simulator produces complex MIMO channel impulse responses along a configurable straight satellite trajectory.

[Quick start](#quick-start) · [Experiments](#experiments) · [Model details](docs/model.md)

## Channel Generation

![Map-based hybrid channel generation procedure](docs/channel_flowchart.png)

Source: **3GPP TR 38.901 V19.1.0, Clause 8.4, Figure 8.4-1**, “Channel coefficient generation procedure.” [ETSI PDF](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf#page=151).

The figure describes the reference procedure. This implementation provides Hybrid RT–RC generation and a fixed-scatterer RC-only option. Full Hybrid temporal consistency and TLE propagation are not implemented; see [model details](docs/model.md) for equations and assumptions.

## Quick Start

The recorded research environment uses **Ubuntu 24.04, Python 3.12, Sionna RT 1.2.1, and an NVIDIA RTX 5080**. Install Git, Python with `venv` support, and a working NVIDIA driver first. Access to this private repository is required.

```bash
git clone https://github.com/Ollie-92/HybridChannel.git
cd HybridChannel
bash tools/install.sh
source .venv/bin/activate
python tools/check_installation.py --run-example
```

`tools/install.sh` creates `.venv` and installs the pinned dependencies. The check runs a three-position ground-plane example and generates channel data, plots, and CSV metrics. On success it prints `Example check passed:` followed by the output directory. Fresh-environment RT execution and numerical reproduction have not yet been validated.

If installation fails, check that Python 3.12 is available. For GPU/backend errors, check `nvidia-smi` and the [Sionna RT prerequisites](https://nvlabs.github.io/sionna/rt/installation.html). For simulation errors, read `run.log` in the printed `RUN_DIR`.

## Experiments

Run commands from the repository root. Each JSON file contains the scene path and simulation parameters.

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

## Usage Notice

The authors permit academic research use of the original HybridChannel code provided that this repository is cited. All other rights are reserved. Redistribution, modification, or commercial use requires prior written permission from the authors, unless separately authorized.

For academic work using this implementation, cite *HybridChannel: Hybrid RT–RC Satellite Channel Simulator*, https://github.com/Ollie-92/HybridChannel, and identify the commit used.

This notice applies only to original project material. Third-party software, scene assets, and reproduced figures remain subject to their respective rights and licenses.

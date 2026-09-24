# Hybrid RT–RC Satellite Channel Simulator

This repository implements a satellite-to-ground channel model combining Sionna ray tracing (RT) with statistical random clusters (RC). The hybrid channel generation methodology is based on the map-based hybrid model in **3GPP TR 38.901 (v19), Clause 8**.

RT paths describe the scene-specific propagation environment. Random clusters supplement these paths through delay generation, duplicate-cluster removal, and power anchoring. The combined rays form a complex MIMO channel impulse response along a configurable straight satellite trajectory.

## Channel Generation Procedure

![Map-based hybrid channel generation procedure](docs/channel_flowchart.png)

Source: **3GPP TR 38.901 V19.1.0 (Release 19), Clause 8.4, Figure 8.4-1**, “Channel coefficient generation procedure.” [ETSI reference](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf#page=151). The figure shows the reference methodology; [model notes](docs/model.md) describe this implementation.

## Files

```text
run_simulation.py       Run a simulation from a configuration file
plot_results.py         Plot PDP/CFR and export CSV metrics
Hybrid_channel/        Channel generation and trajectory modules
Config/                Example, NYC Hybrid, and consistent-RC settings
docs/                  Flowchart, model notes, and dependency notices
requirements.txt       Fixed dependency versions
```

## Installation

The tested simulation environment is Ubuntu 24.04, Python 3.12, Sionna RT 1.2.1, and an NVIDIA RTX 5080. Other platforms have not been verified for RT execution.

```bash
git clone https://github.com/Ollie-92/HybridChannel.git
cd HybridChannel
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

## Usage

Run commands from the repository root.

### Basic example

```bash
python run_simulation.py --config Config/example.json
```

This example generates a small ground-plane scene and evaluates three satellite positions. No external scene is required.

### NYC scene

The NYC scene XML and its referenced meshes are included in this private repository:

```text
Scene/NYC_scene/
    NYC_sionna.xml
    meshes/
```

The scene contains one XML file and 2,905 mesh files (about 94 MB in total). Blender project files and other scenes are not included. Public redistribution permission for these assets has not been established.

```bash
# Hybrid RT + RC
python run_simulation.py --config Config/nyc_hybrid.json

# Fixed-scatterer RC-only
python run_simulation.py --config Config/nyc_consistent_rc.json
```

Edit the `cli` fields in `Config/` to change carrier frequency, antenna count, receiver coordinates, trajectory, seed, or RT settings. The NYC configurations use 28 GHz, 600 km scene altitude, and five positions over 0.2 seconds.

For another scene, add `--scene /absolute/path/to/scene.xml` and adjust receiver coordinates and trajectory settings for that scene.

### Results

Each simulation prints a new directory under `Result/`, containing channel coefficients, delays, satellite positions, configuration, and logs. Use that directory to generate PDP/CFR plots and CSV metrics:

```bash
python plot_results.py Result/<run-directory> --trusted-local-output
```

Only analyze trusted simulation files. Analysis reads NumPy object arrays and saves its outputs under the run's `analysis/` directory.

## Model Notes

The consistent-RC option retains fixed scatterers and outputs an RC-only channel. Full Hybrid temporal consistency and TLE propagation are not implemented. Hybrid delays are excess delays; consistent-RC delays are absolute delays. Equations, units, and the retained NTN parameter approximations are documented in [docs/model.md](docs/model.md).

Original numerical comparisons and development tests are preserved on the [archive/research-validation branch](https://github.com/Ollie-92/HybridChannel/tree/archive/research-validation). Its instructions apply to that branch's layout. Independent 3GPP/OpenNTN conformance has not been established.

## References and Usage

- 3GPP TR 38.901 V19.1.0, *Study on channel model for frequencies from 0.5 to 100 GHz*, Release 19, October 2025. [ETSI PDF](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf).
- Ray tracing uses Sionna RT. See [dependency notices](docs/third_party/README.md).

### OpenNTN

[OpenNTN](https://github.com/ant-uni-bremen/OpenNTN) is an independent NTN channel-model reference used by a separate reference script in the original research project. The simulation code in this repository does not import OpenNTN. Following its [citation instructions](https://github.com/ant-uni-bremen/OpenNTN#citing-openntn), we acknowledge:

T. Düe, M. Vakilifard, C. Bockelmann, D. Wübben, and A. Dekorsy, “OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations,” *International Workshop on Smart Antennas (WSA)*, Erlangen, Germany, September 16–18, 2025. [Paper](https://www.ant.uni-bremen.de/sixcms/media.php/102/15183/OpenNTN_An_OpenSource_Framework_For_NTN_Simulations.pdf).

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

A project license has not yet been selected. Third-party components retain their respective licenses. Research publication and citation information will be added when available.

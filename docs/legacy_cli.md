# Original command-line workflow

The original implementation and its configurations remain available. Run these commands from the repository root. For the notebook workflow, see the [README](../README.md).

## Installation


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

## Experiments


The following original CLI experiments remain available. Run commands from the repository root. Each JSON file contains the scene path and simulation parameters.

| Configuration | Scene | Channel | Positions |
|---|---|---|---|
| [example.json](../Config/example.json) | Generated ground plane | Hybrid RT–RC | 3 |
| [nyc_hybrid.json](../Config/nyc_hybrid.json) | NYC | Hybrid RT–RC | 5 |
| [nyc_consistent_rc.json](../Config/nyc_consistent_rc.json) | NYC | Fixed-scatterer RC-only, with RT power anchoring | 5 |

```bash
python run_simulation.py --config Config/nyc_hybrid.json

# Fixed-scatterer RC-only
python run_simulation.py --config Config/nyc_consistent_rc.json

# Plot one completed run
python tools/plot_results.py Result/<run-directory> --trusted-local-output
```

Replace `<run-directory>` with the directory printed by the simulation. The NYC settings use 28 GHz and a 600 km scene altitude. The scene XML and referenced meshes are included in [Scene/NYC_scene](../Scene/NYC_scene).

Edit the `cli` fields in a configuration to change frequency, antennas, receiver coordinates, trajectory, seed, or RT settings. For another scene, add `--scene /absolute/path/to/scene.xml` to the simulation command and adjust its geometry settings.

Each run saves CIR coefficients, delays, geometry, configuration, package versions, and logs under `Result/`. Analysis adds PDP/CFR plots and CSV metrics in `analysis/`. Only analyze trusted files: the saved channel uses NumPy object arrays.


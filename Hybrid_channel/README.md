# Hybrid Channel Simulation

## Requirements

The tested environment is Ubuntu 24.04, Python 3.12, Sionna RT 1.2.1, and an NVIDIA RTX 5080. Dependency versions are fixed in `requirements.txt`. Other platforms have not been verified.

From the repository root:

```bash
cd Hybrid_channel
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation .
.venv/bin/python -m pip check
```

Run the commands below from this directory.

## Files

```text
src/hybridchannel/
    hybrid_channel.py              Hybrid channel generation
    lsp_calculator.py              Large-scale parameter sampling
    cluster_splitter.py            RT path extraction
    consistent_random_clusters.py  Fixed-scatterer RC evolution
    trajectory.py                  Satellite trajectory simulation
configs/                           Experiment settings
scripts/                           Simulation and analysis entry points
data/                              Scene input instructions
tests/                             Numerical checks
docs/                              Model equations and execution flow
third_party/                       Dependency notices
```

## Ground-Plane Example

```bash
.venv/bin/python scripts/run_experiment.py --config configs/E01-smoke.json
```

This example generates a ground-plane scene and evaluates three satellite positions. The command prints the output directory under `runs/`.

## City Scene

Place `NYC_sionna.xml` and its referenced meshes in `data/external/NYC_scene/`, preserving their relative paths. See [scene instructions](data/README.md).

```bash
# Hybrid RT + RC
.venv/bin/python scripts/run_experiment.py --config configs/E02-nyc-hybrid.json

# Consistent RC-only
.venv/bin/python scripts/run_experiment.py --config configs/E03-nyc-consistent-rc.json
```

To use another scene:

```bash
.venv/bin/python scripts/run_experiment.py --config configs/E02-nyc-hybrid.json --scene /absolute/path/to/scene.xml
```

Set receiver coordinates and trajectory parameters for the selected scene in the configuration file. Changing the scene defines a different experiment.

## Parameters

Edit the `cli` fields in `configs/` to set carrier frequency, antenna count, receiver position, trajectory, random seed, and RT settings. E02 uses 28 GHz, 600 km scene altitude, 7,560 m/s satellite speed, and five positions over 0.2 seconds.

The consistent-RC option freezes the initial random clusters and outputs RC-only channels. It does not track a combined RT/RC channel. TLE propagation, a general RT/RC/Hybrid selector, and full Hybrid temporal consistency are not implemented.

## Results

Each run creates a new directory containing `channel.npz`, `summary.txt`, configuration, environment information, input hashes, and logs.

To generate PDP/CFR plots and CSV metrics, substitute the directory printed by the simulation:

```bash
.venv/bin/python scripts/analyze_run.py runs/<run-directory> --trusted-local-output
```

Only analyze trusted outputs: channel files contain NumPy object arrays. Hybrid delays are excess delays; consistent-RC delays are absolute delays. See [model and units](docs/model_and_units.md) and [execution flow](docs/flowcharts.md).

## Numerical Checks

```bash
.venv/bin/python -m unittest discover -s tests -p test_contracts.py
```

Recorded validation on the environment above includes three ground-plane positions, five NYC positions for each branch, and 16 exact comparisons against the original implementation. These are prior results, not a new simulation run after this directory reorganization. The retained NTN parameter table contains approximations; independent 3GPP/OpenNTN conformance has not been established.

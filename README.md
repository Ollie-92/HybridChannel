# Hybrid RT–RC Satellite Channel Simulator

This repository implements a hybrid satellite-to-ground channel model combining Sionna ray tracing (RT) with statistical random clusters (RC). The model uses scene-specific RT paths to generate and anchor random clusters, then assembles the resulting rays into a complex MIMO channel impulse response.

## Channel Model

The implementation includes:

- RT path extraction and power-based filtering
- Large-scale parameter sampling from a local NTN parameter table
- RT-conditioned RC delays, duplicate-cluster removal, and power anchoring
- Cluster-to-ray expansion and complex channel coefficient generation
- Channel simulation along a configurable straight satellite trajectory

A separate consistent-RC option keeps fixed scatterers across trajectory positions and produces an RC-only channel. TLE trajectories and full Hybrid temporal consistency are planned extensions. See [model details](docs/model_and_units.md) and [execution flow](docs/flowcharts.md).

## Project Structure

```text
src/hybridchannel/
    hybrid_channel.py              Hybrid channel generation
    lsp_calculator.py              Large-scale parameter sampling
    cluster_splitter.py            RT path extraction
    consistent_random_clusters.py  Fixed-scatterer RC evolution
    trajectory.py                  Satellite trajectory simulation
configs/                           Experiment configurations
scripts/                           Experiment and analysis scripts
tests/                             Numerical checks
docs/                              Model, usage, and validation notes
data/                              Scene placement instructions
third_party/                       Dependency license notices
```

## Requirements

The tested environment is Ubuntu 24.04, Python 3.12, Sionna RT 1.2.1, and an NVIDIA RTX 5080. Package versions are listed in `requirements-linux-py312.lock`. Other platforms have not been verified.

```bash
git clone https://github.com/Ollie-92/HybridChannel.git
cd HybridChannel
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-linux-py312.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation .
.venv/bin/python -m pip check
```

## Usage

### Basic example

Run the small ground-plane example:

```bash
.venv/bin/python scripts/run_experiment.py --config configs/E01-smoke.json
```

This example generates its own scene and evaluates three satellite positions. Results are saved to a new directory under `runs/`.

To generate PDP/CFR plots and CSV metrics, use the run directory printed by the command above:

```bash
.venv/bin/python scripts/analyze_run.py runs/<run-directory> --trusted-local-output
```

### City scene

Place the scene XML and its referenced meshes under `data/external/` as described in [data/README.md](data/README.md). Scene assets are not included in this repository.

```bash
# Hybrid RT + RC
.venv/bin/python scripts/run_experiment.py --config configs/E02-nyc-hybrid.json

# Consistent RC-only
.venv/bin/python scripts/run_experiment.py --config configs/E03-nyc-consistent-rc.json
```

Carrier frequency, antenna count, receiver position, trajectory, and RT settings are specified in the configuration files. See the [usage guide](docs/usage.md) for custom scenes and parameter conventions.

## Outputs and Validation

Each run saves channel coefficients, delays, satellite positions, configuration, and logs. The analysis script adds PDP/CFR plots and CSV metrics.

The recorded tests include a three-position ground-plane run, five-position NYC Hybrid and RC-only runs, and 16 exact numerical comparisons against the original implementation. These results were obtained in the environment listed above. See the [validation report](docs/validation_report.md) for details.

The local LSP table contains retained approximations; independent 3GPP/OpenNTN conformance and complete paper reproduction have not been established.

## References and Usage

Sionna RT is used for ray tracing. Dependency attribution is listed in [third_party/README.md](third_party/README.md). Research publication and citation information will be added when available.

A project license has not yet been selected. Third-party components retain their respective licenses. External scene data must be obtained separately.

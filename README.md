# Hybrid RT–RC Satellite Channel Simulator

This repository implements a satellite-to-ground channel model combining Sionna ray tracing (RT) with statistical random clusters (RC).

RT paths describe the scene-specific propagation environment. Random clusters supplement these paths through delay generation, duplicate-cluster removal, and power anchoring. The combined rays are used to generate a complex MIMO channel impulse response along a configurable satellite trajectory.

## Project Structure

All simulation code, experiment settings, and installation files are in [`Hybrid_channel/`](Hybrid_channel/).

## Usage

The simulator provides two experiments:

- **Hybrid RT + RC:** generate a combined channel at each satellite position.
- **Consistent RC:** retain fixed random scatterers and evaluate an RC-only channel along the trajectory.

The current trajectory is a configurable straight line. See [`Hybrid_channel/README.md`](Hybrid_channel/README.md) for installation, scene input, and execution commands.

## Simulation Outputs

The simulator saves channel coefficients, delays, and satellite positions. The analysis script generates PDP/CFR plots and CSV metrics. Results are stored under `Hybrid_channel/runs/`.

City scene XML and mesh files must be supplied separately. A small ground-plane example is included for checking the simulation workflow.

## References

Ray tracing uses Sionna RT. Dependency notices are provided in [`Hybrid_channel/third_party/`](Hybrid_channel/third_party/). Publication and citation information will be added when available.

## Usage Notice

A project license has not yet been selected. Third-party components retain their respective licenses.

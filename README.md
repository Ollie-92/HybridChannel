# Hybrid RT–RC Satellite Channel Simulator

This repository implements a satellite-to-ground channel model combining Sionna ray tracing (RT) with statistical random clusters (RC).

The hybrid channel generation methodology is based on the map-based hybrid model in **3GPP TR 38.901 (v19), Clause 8**. Satellite-specific parameters and implementation assumptions are described in the [model notes](Hybrid_channel/docs/model_and_units.md).

RT paths describe the scene-specific propagation environment. Random clusters supplement these paths through delay generation, duplicate-cluster removal, and power anchoring. The combined rays are used to generate a complex MIMO channel impulse response along a configurable satellite trajectory.

## Channel Generation Procedure

![Map-based hybrid channel generation procedure](Hybrid_channel/docs/figures/hybrid_channel_flowchart.png)

Source: **3GPP TR 38.901 V19.1.0 (Release 19), Clause 8.4, Figure 8.4-1**, “Channel coefficient generation procedure.” [ETSI reference](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf#page=151). The figure shows the reference procedure; see the [implemented flow](Hybrid_channel/docs/flowcharts.md) for this simulator.

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

- 3GPP TR 38.901 V19.1.0, *Study on channel model for frequencies from 0.5 to 100 GHz*, Release 19, October 2025. [ETSI PDF](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf).

Ray tracing uses Sionna RT. Dependency notices are provided in [`Hybrid_channel/third_party/`](Hybrid_channel/third_party/). Publication and citation information will be added when available.

## Usage Notice

A project license has not yet been selected. Third-party components retain their respective licenses.

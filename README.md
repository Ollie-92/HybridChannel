# HybridChannel

Hybrid RT–RC satellite channel simulator based on the map-based hybrid model in **3GPP TR 38.901 (v19), Clause 8**.

Sionna RT computes propagation paths in a 3D scene. Statistical random clusters supplement those paths through delay generation, duplicate-cluster removal, and power anchoring. The simulator produces complex MIMO channel impulse responses for configured satellite and user-terminal positions.

[Quick start](#quick-start) · [Usage](#usage) · [Outputs](#outputs)

## Channel Generation

![Map-based hybrid channel generation procedure](hybrid_simulator/channel_flowchart.png)

Source: **3GPP TR 38.901 V19.1.0, Clause 8.4, Figure 8.4-1**, “Channel coefficient generation procedure.” [ETSI PDF](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf#page=151).

The figure describes the reference channel-generation procedure.

## Quick Start

Open [hybrid_simulator.ipynb](hybrid_simulator/hybrid_simulator.ipynb) to configure and run the simulator. On Linux/Ubuntu with Python 3.12 and a working NVIDIA driver:

```bash
git clone --branch codex/notebook-simulator-integration https://github.com/Ollie-92/HybridChannel.git
cd HybridChannel
python3.12 -m venv .venv-notebook
source .venv-notebook/bin/activate
python -m pip install -r hybrid_simulator/requirements-notebook.txt
python -m jupyterlab hybrid_simulator/hybrid_simulator.ipynb
```

The branch above contains the update in PR #2. The dependency file specifies the package versions. Installation on a fresh machine has not yet been verified.

## Usage

1. Run **Environment** and **Configuration**.
2. Set `mode` to `rt`, `hybrid`, or `3gpp` (OpenNTN), and select downlink, uplink, or both.
3. Configure the endpoint positions and antennas in the same Configuration cell.
4. Run **A: fixed positions** or **B: orbit**, then create the channels and run channel generation. Execute only the selected geometry section; OpenNTN currently uses fixed topology.
5. Run **Inspect outputs** and the plotting cells.

The default uses the included NYC scene and 10 million RT samples per source. Execution time depends on the scene and hardware. Restart the kernel after updating the Python modules.

## Outputs

The notebook prints the output directory, under `hybrid_simulator/results/runs/<run>/<link>/<direction>/fixed` or `orbit_XXXX`. Reusing the same directory overwrites matching output files.

| File | What it contains / when to use it |
|---|---|
| `channel_paths.csv` | Every ray/tap's complex coefficient and delay for each RX/TX port pair and time sample. Use it to inspect individual propagation contributions. |
| `channel_matrix.csv` | The complex channel after summing all path coefficients at the carrier, for each port pair and time sample. Use it to inspect the narrowband MIMO matrix. |
| `cfr.csv` | The complex channel at each frequency offset. Use it to inspect frequency-selective behavior. |
| `pdp.csv` | Path powers summed across all ports at exactly equal delays, without phase interference or peak normalization. This is a combined power table, not a selected-port channel matrix. |
| `inspection_summary.csv` | Mode, frequency, snapshot/sample, port counts and path counts displayed by Inspect outputs. |
| `port_metrics.csv` | Each RX/TX pair's power gains, coherent phase, mean excess delay and RMS delay spread for the selected sample. |
| `source_summary.csv` | Counts and summed path powers by RT, RC or OpenNTN source for the selected sample. |
| `channel.npz` | Complete numerical arrays for loading back into Python; complex values and dimensions are preserved. |
| `summary.json` | Simulation metadata, units, conventions and result dimensions. |
| `CSV_README.txt` | CSV column definitions and units. |
| `ray_pdp_*.png` / `.csv` | The selected port pair's per-ray plot and the exact points used to draw it; saved in the link directory. |

The inspection CSVs are created when **Inspect outputs** runs. For the per-ray plot, select `RX_PORT`, `TX_PORT`, `SNAPSHOT_INDEX` and `TIME_SAMPLE`. Each panel is normalized to its strongest ray (0 dB), and equal-delay rays overlap without summation. This differs from the combined `pdp.csv`. Power gains are in dB, not received power in dBm.

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

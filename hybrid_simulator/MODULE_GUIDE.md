# Module and output guide

Open the root `hybrid_simulator.ipynb` from the repository directory and restart the kernel after code updates.

| Module | Role |
|---|---|
| channel_generator.py | Mode selection and orchestration; exports HybridChannel and generate_channel |
| hybrid_model.py | Hybrid RC generation, power anchoring, subray expansion and channel synthesis |
| config.py | Validation, notebook configuration conversion and the fixed internal VH probe |
| rt.py | Native Sionna scene, antennas, PathSolver and CIR |
| openntn.py | Native OpenNTN adapter |
| rt_postprocess.py | Native RT conversion, thresholding and Hybrid inputs |
| lsp_calculator.py | NTN table lookup and correlated LSP sampling |
| topology.py | Endpoint coordinates, direction and elevation |
| orbit.py | TLE/linear positions and trajectory restrictions |
| temporal.py | Experimental fixed-scatterer segment state |
| ionosphere.py | Optional medium corrections |
| result.py | Shared channel result, NPZ/JSON/CSV persistence |
| csv_export.py | Readable coefficient, matrix, CFR and delay-grouped PDP tables |
| visualization.py | Analysis and rendering helpers |

`hybrid_channel.py` and `hybrid.py` are compatibility import shims. New code should use the descriptive module names above. They do not contain separate model implementations.

Notebook order: environment -> configuration -> fixed/orbit geometry -> generators -> snapshot loop -> inspection -> custom plotting.
The expected Mitsuba variant is set in `environment`, separately from physical parameters. It does not switch an already loaded backend.
The fixed 1x1 iso VH probe comes from `config.py`; endpoint antennas still apply at Hybrid synthesis. Moving these modules and flattening the former backends directory does not change their calculations. Numerical equivalence to the original Hybrid_channel package has not been established.

Every saved snapshot contains:

- `channel_paths.csv`: one row per RX port, TX port, path and time sample. Real/imaginary coefficients, power, source, angles and delays are explicit.
- `channel_matrix.csv`: coherent path sum for each RX/TX port and time sample.
- `pdp.csv`: incoherent power sum over ports and exactly equal delays, with constituent path counts.
- `cfr.csv`: complex frequency response per port, time sample and frequency offset.
- `CSV_README.txt`: units, conventions and missing-value policy.
- `channel.npz`, `summary.json`: complete arrays and metadata remain available.

For a 2x2-port, 41-path, one-sample channel, channel_paths.csv has 164 data rows. A 1+20+20 shared-delay grouping gives three PDP rows without discarding the 41 path records. PDP dB values are summed dimensionless channel power, not dBm or peak-normalized values.

Each snapshot is saved under `results/runs/<run>/<link>/<direction>/<snapshot>/`. The notebook prints its absolute directory. Reusing the same output path overwrites matching files. CSV export does not rerun the channel or modify NPZ values.

OpenNTN now supports regular multi-element arrays. See OPENNTN_MIMO.md for port order, native parity validation and the pinned upstream version.

The notebook per-ray plot selects one RX/TX port pair, defaults to snapshot 0/sample 0, and normalizes each panel to its strongest ray. Equal-delay rays overlap. Its `ray_pdp_*.csv` matches the plotted points; it intentionally differs from the all-port `pdp.csv`. `visualization.py` is retained as an optional helper and is not called by the notebook.

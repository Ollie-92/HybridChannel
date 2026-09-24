# Block–code mapping

Block IDs jointly map the architecture and procedural images supplied by the user. Original locations below refer to the exact **YuChein** SHA-256 snapshot in [provenance.json](provenance.json). Full private absolute locations are outside Git. Status is implementation status; executed test evidence is in [validation_report.md](validation_report.md).

| Block ID | Function | Original file and lines | Function/class | Extracted file | Input | Output | Dependencies | Experiment | Verification status |
|---|---|---|---|---|---|---|---|---|---|
| B01 | Simulation configuration | `YuChein/run_satellite_trajectory_channel.py:185-462` | main | `src/hybridchannel/trajectory.py` | CLI/config JSON | parsed options | argparse | E01/E02/E03 | Implemented; JSON wrapper added |
| B02 | 3D scene input | `YuChein/run_satellite_trajectory_channel.py:63-102` | build_scene | `src/hybridchannel/trajectory.py` | XML/PLY, Hz, dBm, meters | Sionna Scene | Sionna RT, Mitsuba, scene assets | E01/E02/E03 | Implemented; private NYC assets withheld |
| B03 | Satellite trajectory | `YuChein/run_satellite_trajectory_channel.py:105-126` | make_linear_trajectory | `src/hybridchannel/trajectory.py` | m, m/s, deg, seconds, count | times and positions/velocities | NumPy | E01/E02/E03 | Straight line implemented; TLE/ephemeris absent |
| B04 | RT / RC / Hybrid selector | Not found | top-level / absent | Not included | requested mode | dispatch | none | none | Not implemented; existing flag selects RC-only consistency |
| B05 | Per-position geometry | `YuChein/run_satellite_trajectory_channel.py:129-132` | elevation_deg | `src/hybridchannel/trajectory.py` | satellite and receiver XYZ meters | elevation degrees; main computes slant range | NumPy | E01/E02/E03 | Implemented |
| B06 | LOS-only solve | `YuChein/run_satellite_trajectory_channel.py:185-462` | main | `src/hybridchannel/trajectory.py` | Scene and LOS solver options | LOS Paths; skipped steps | PathSolver, lsp_calculator.has_los | E01/E02/E03 | Implemented |
| B07 | Full RT solve | `YuChein/run_satellite_trajectory_channel.py:185-462` | main | `src/hybridchannel/trajectory.py` | Scene, full solver settings, seed | Paths | PathSolver | E01/E02/E03 | Implemented: LOS/reflection/scattering/refraction |
| B08 | Extract deterministic paths | `YuChein/cluster_splitter.py:37-92` | extract_paths_from_rt | `src/hybridchannel/cluster_splitter.py` | full and LOS Paths | L powers/delays/angles | NumPy, Sionna CIR interface | E01/E02/E03 | Implemented; scalar VV powers |
| B09 | RT path filtering | `YuChein/run_satellite_trajectory_channel.py:141-163` | rt_paths_to_hybrid_input | `src/hybridchannel/trajectory.py` | path dictionary, dB threshold | sorted RT records | NumPy | E01/E02/E03 | Implemented; -25 dB default |
| B10 | 3GPP-derived local parameters | `YuChein/lsp_calculator.py:323-358` | _ntn_dur_los | `src/hybridchannel/lsp_calculator.py` | elevation degrees | table parameters | embedded GPP_TABLE | E01/E02/E03/E04 | Legacy approximate table, not an OpenNTN invocation |
| B11 | Sample LSPs | `YuChein/lsp_calculator.py:411-516` | sample_lsps | `src/hybridchannel/lsp_calculator.py` | table + RNG | DS seconds, K dB, 4 spreads degrees | NumPy | E01/E02/E03/E04 | Implemented; 6-dimensional covariance |
| B12 | RT-conditioned RC delays | `YuChein/hybrid_channel.py:49-78` | generate_rc_delays | `src/hybridchannel/hybrid_channel.py` | compute_mu_tau_RC result + RNG | scaled and unscaled delays | local LSP sample | E01/E02/E03 | Implemented; coupled to RT |
| B13 | Delay de-duplication | `YuChein/hybrid_channel.py:84-115` | remove_rc_clusters | `src/hybridchannel/hybrid_channel.py` | candidate and RT delays, p0 | retained RC delays | compute_mu_tau_RC | E01/E02/E03 | Implemented |
| B14 | Power generation and anchoring | `YuChein/hybrid_channel.py:194-213` | anchor_rc_power | `src/hybridchannel/hybrid_channel.py` | virtual RC/RT powers + actual RT powers | anchored RC powers | compute_virtual_powers, normalize_virtual_powers | E01/E02/E03 | Implemented |
| B15 | RC angles | `YuChein/hybrid_channel.py:219-339` | generate_rc_angles | `src/hybridchannel/hybrid_channel.py` | LSPs, RT angles/powers, RNG | 4 angle arrays in radians | lookup_C_phi/theta | E01/E02/E03 | Implemented |
| B16 | Cluster/ray merge and channel state | `YuChein/hybrid_channel.py:608-757` | expand_clusters_to_rays | `src/hybridchannel/hybrid_channel.py` | merged thresholded clusters, M | rays, PDP, source labels | merge_and_truncate; NumPy/SciPy | E01/E02/E03 | Implemented; no independent global state class |
| B17 | Spatial consistency manager | `YuChein/consistent_random_clusters.py:246-421` | evaluate_random_clusters | `src/hybridchannel/consistent_random_clusters.py` | frozen RC state, current geometry/RT anchor | RC-only channel | freeze_random_cluster_state | E03 | Partial: fixed RC scatterers only; no deterministic path tracking |
| B18 | Complex channel assembly | `YuChein/hybrid_channel.py:875-1020` | compute_channel_coefficients | `src/hybridchannel/hybrid_channel.py` | rays/phases/arrays/frequency/velocity | H[U,S,N,M], tau[N,M] | generate_xpr, generate_initial_phases | E01/E02/E03 | CIR implemented; separate added Fourier CFR analysis |
| B19 | Channel analysis | `YuChein/hybrid_channel.py:445-467` | compute_metadata | `src/hybridchannel/hybrid_channel.py` | clusters and powers | realized DS, target DS, counts | PDP accumulation; added analysis script | E01/E02/E03 | Original metadata; CSV/CFR/plot wrapper added |
| B20 | LSP validation / OpenNTN reference | `YuChein/run_pdp_3gpp_single.py:1-99` | top-level / absent | Not included | independent statistical topology | reference PDP plot | OpenNTN DenseUrban + TensorFlow | reference only; E04 local diagnostic | Separate original script, not per-step validation; not executed in this extraction |
| B21 | Save results | `YuChein/run_satellite_trajectory_channel.py:185-462` | main | `src/hybridchannel/trajectory.py` | valid tensors, positions, summaries | NPZ + text; wrapper adds manifests | NumPy, JSON | E01/E02/E03 | Implemented after the loop; analysis exports CSV/figures |

## Substage locations

- `YuChein/hybrid_channel.py:15-43` — `compute_mu_tau_RC`.
- `YuChein/hybrid_channel.py:49-78` — `generate_rc_delays`.
- `YuChein/hybrid_channel.py:84-115` — `remove_rc_clusters`.
- `YuChein/hybrid_channel.py:121-156` — `compute_virtual_powers`.
- `YuChein/hybrid_channel.py:162-188` — `normalize_virtual_powers`.
- `YuChein/hybrid_channel.py:194-213` — `anchor_rc_power`.
- `YuChein/hybrid_channel.py:219-339` — `generate_rc_angles`.
- `YuChein/hybrid_channel.py:345-413` — `merge_and_truncate`.
- `YuChein/hybrid_channel.py:419-439` — `discretize_pdp`.
- `YuChein/hybrid_channel.py:473-596` — `generate_hybrid_pdp`.
- `YuChein/hybrid_channel.py:608-757` — `expand_clusters_to_rays`.
- `YuChein/hybrid_channel.py:781-804` — `generate_xpr`.
- `YuChein/hybrid_channel.py:810-868` — `generate_initial_phases`.
- `YuChein/hybrid_channel.py:875-1020` — `compute_channel_coefficients`.
- `YuChein/hybrid_channel.py:1026-1120` — `generate_full_channel`.
- `YuChein/consistent_random_clusters.py:86-243` — `freeze_random_cluster_state`.
- `YuChein/consistent_random_clusters.py:246-421` — `evaluate_random_clusters`.

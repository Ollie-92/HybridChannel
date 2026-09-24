# Actual execution flow

The two supplied images describe a target architecture. This graph shows only the inspected selected code; dashed reference relationships are not an implemented per-step API.

```mermaid
flowchart TD
  C["B01 configs + run_experiment.py"] --> S["B02 trajectory.build_scene: XML and meshes"]
  S --> T["B03 make_linear_trajectory"]
  T --> L["B05 loop: position and elevation_deg"]
  L --> LO["B06 PathSolver LOS-only; has_los"]
  LO -->|no LOS| NEXT["skip invalid step"]
  LO -->|LOS| RT["B07 PathSolver full RT"]
  RT --> X["B08 cluster_splitter.extract_paths_from_rt"]
  X --> F["B09 trajectory.rt_paths_to_hybrid_input"]
  F --> BR{"existing consistent-RC flag?"}
  BR -->|false| SP["B10/B11 lsp_calculator: table + sample_lsps"]
  SP --> D["B12 hybrid_channel: compute_mu_tau_RC / generate_rc_delays"]
  D --> DD["B13 remove_rc_clusters"]
  DD --> P["B14 virtual normalization + anchor_rc_power"]
  P --> A["B15 generate_rc_angles"]
  A --> M["B16 merge_and_truncate / expand_clusters_to_rays"]
  M --> H["B18 generate_xpr / generate_initial_phases / compute_channel_coefficients"]
  BR -->|true, first valid step| FR["B17 freeze_random_cluster_state calls the same generator"]
  FR --> EV["B17 evaluate_random_clusters: RC ONLY"]
  BR -->|true, subsequent step| EV
  EV --> HRC["B18 compute_channel_coefficients: RC ONLY"]
  H --> STORE["B21 accumulate arrays and summaries"]
  HRC --> STORE
  STORE --> NEXT
  NEXT -->|more positions| L
  NEXT -->|finished| SAVE["B21 NPZ + summary after loop"]
  SAVE --> ANAL["B19 added analyze_run.py: CSV / PDP / Fourier CFR"]
  REF["B20 original run_pdp_3gpp_single.py: independent OpenNTN"] -. separate reference .-> ANAL
  GPP["B20 added validate_lsp_sampling.py: local table diagnostic"] -. no independent conformance proof .-> SP
```

`B04` (general RT/RC/Hybrid selector) has no matching implementation. TLE/ephemeris in B03, complete RT identity tracking in B17 and automatic per-step OpenNTN LSP validation in B20 are not implemented in this path. Saving happens after the loop, not after each position as drawn in the process diagram. The new wrapper and analysis are labeled additions, not prior research methods.

```mermaid
flowchart LR
 trajectory.py --> cluster_splitter.py
 trajectory.py --> hybrid_channel.py
 trajectory.py --> consistent_random_clusters.py
 trajectory.py --> lsp_calculator.py
 hybrid_channel.py --> lsp_calculator.py
 consistent_random_clusters.py --> hybrid_channel.py
 trajectory.py --> SionnaRT["Sionna RT / Mitsuba / Dr.Jit"]
 hybrid_channel.py --> NumPy
 hybrid_channel.py --> SciPy["SciPy only for M other than 1 or 3"]
```

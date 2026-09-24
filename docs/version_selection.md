# Version selection and original behavior

Decision: on 2026-09-23 the owner explicitly selected **YuChein original HybridChannel**, after being shown the root/legacy/v2/v3 differences. This decision did not use file modification times.

The inspected original `run_satellite_trajectory_channel.py` imports its sibling `hybrid_channel`, `consistent_random_clusters`, `cluster_splitter` and `lsp_calculator`; none of these five imports IONORT or an ionosphere module. The selected call path can be extracted without removing a physical model. Its only lazy non-stdlib dependency beyond NumPy is SciPy for alternate ray counts. Original optional `to_sionna_cir` TensorFlow conversion is unused by the driver and was not extracted.

| Candidate | Inspected differences | Decision |
|---|---|---|
| YuChein root | Scalar theta-only Hybrid and existing RC-only consistency | Selected by owner |
| NTN_2026 root | Additional RT identity metadata; optional polarization/RT replay and Faraday hooks; CFR/link metrics | Not combined with selected version |
| NTN_2026/hybrid_legacy_before_polarization | Shared LSP/trajectory/consistency function bodies match YuChein; added `rt_index` fields in merge/ray functions | Useful comparison evidence, not replacement baseline |
| NTN_2026/hybrid_ntn_v2 | Changes in LSP lookup/sampling, angle generation, PDP/ray generation and polarization wrapper | Different research behavior; excluded |
| NTN_2026/hybrid_ntn_v3 | Additional table-profile and ray-delay choices; changed LSP and ray code; polarization wrapper | Different research behavior; excluded |

AST comparisons and full SHA-256s are preserved in the private version audit. In particular YuChein and NTN root `lsp_calculator.py` have identical function bodies (one initial blank line differs), whereas v2/v3 change lookup/sampling functions. YuChein vs the legacy backup differs in merge/ray RT identity fields; no metadata from that branch was injected into the selected baseline.

Root `NTN_2026/hybrid_channel.py:1041` imports `hybrid_polarization` before deciding the scalar/extended branch. That is an actual import coupling in an **unselected** version. YuChein has no such dependency. No Faraday code was copied into the package or disabled by a hidden flag.

Extraction changes are limited to relative package imports, English documentation, the required function subset of shared modules, an NTN-only registry, and portable CLI I/O resolution. Existing NTN function bodies are checked structurally after ignoring documentation/import qualification; baseline numeric tests use identical input/seed and a pristine source copy. The private patch files show every change. New run/analysis/test infrastructure is separately identified.

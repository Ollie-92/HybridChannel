# Experiment catalog

| ID | Kind | Blocks | Script/config | Data/version | Seeds | Expected outputs | Paper mapping |
|---|---|---|---|---|---|---|---|
| E01-smoke | Added integration test | B01–B03, B05–B16, B18/B19/B21 | run_experiment.py; E01-smoke.json | Generated test XML; hash in each run | 42 + documented offsets | 3 CIR snapshots, metadata, logs; optional plot/CFR | Not a paper figure |
| E02-nyc-hybrid | Original main CLI defaults | B01–B03, B05–B16, B18/B19/B21 | run_experiment.py; E02-nyc-hybrid.json | Exact private NYC XML + 2,905 meshes, SHA-256 manifest | 42 + step / RT offsets | 5 requested snapshots; validity explicitly checked | TBD by author |
| E03-nyc-consistent-rc | Original existing optional branch | same plus B17 | run_experiment.py; E03-nyc-consistent-rc.json | Same private scene | RC once at 42, current RT seeds as above | RC-only snapshots with absolute delays | TBD by author |
| E04-lsp-sampling | Added diagnostic of original sampler | B10/B11/B20 | validate_lsp_sampling.py | Embedded legacy GPP_TABLE, source hash in provenance | 42; 10,000 samples at 40 deg | JSON mean/std/covariance/eigenvalues | TBD, diagnostic only |
| V01-copy-parity | Extraction verification | B10–B18 | tests/compare_baseline.py | Explicit synthetic RT fixture + pristine private code copy | 7,42,123,2026; M=1,3,20 | Exact intermediate/final comparison JSON | Verification, not scientific result |

Commands appear in README. The trajectory JSONs list every original main argument so no hidden defaults are needed. The wrapper records input hashes, full configuration, installed environment, current Git commit or null, dirty status, exact package hashes, command and unique output location. Git identity was not invented. Execution outcomes are recorded separately in validation_report.md.

The separate source `YuChein/run_pdp_3gpp_single.py` invokes OpenNTN directly and generates nine random reference realizations. It has no controlled seed and is not called by the selected pipeline. It was read and audited but not advertised as a reproduced conference experiment. `run_hybrid_lsp_compare.py` in the source compares local table values and includes a 3-billion-sample RT configuration, which is distinct from the main trajectory defaults. Neither script supplies an automatic per-step validation block in the selected driver.

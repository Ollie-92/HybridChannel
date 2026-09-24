# Validation report — 2026-09-23

**Result:** the packaged HybridChannel implementation is executable in an independent environment, preserves the checked original numerical behavior, and completes the original main trajectory configurations. This is **not** a claim that every conference figure, an independent OpenNTN benchmark, or the complete supplied target architecture has been reproduced.

Machine-readable results are in [validation_evidence.json](validation_evidence.json); source integrity is in [source_integrity_report.json](source_integrity_report.json). Full run directories, logs, pristine references, source manifests and file-access traces are retained privately outside the proposed Git distribution.

## Environment and isolation

- Ubuntu 24.04.3 LTS, Linux 6.8.0-139-generic, x86-64; Python 3.12.3.
- AMD Ryzen 9 9950X3D, 16 cores / 32 logical CPUs; 249 GiB RAM.
- NVIDIA RTX 5080, 16,303 MiB reported VRAM; driver 575.64.03.
- NumPy 1.26.4, SciPy 1.16.1, Sionna RT 1.2.1, Mitsuba 3.7.1, Dr.Jit 1.2.0, Matplotlib 3.10.5. The exact resolved closure and build tools are in requirements-linux-py312.lock.
- A new venv was created using the system interpreter, without system-site-packages or the original research environment. The package was installed normally as a built distribution, not imported from the original roots or an editable source path.
- A **second** empty venv followed the README's locked install, `pip install --no-deps --no-build-isolation .`, `pip check`, and E01 command. Installation, dependency check and three-step smoke all passed. No TensorFlow, Sionna PHY or OpenNTN is required for these commands.
- Caches, HOME, logs, temporary files, installs and outputs were directed to the new workspace. A validation-only Python audit hook denied file opens/listing/chdir under the original research environment. `strace -f -e trace=%file` recorded no original-environment file-open or mutation calls in the smoke, E02, E03 and clean-README-smoke traces. Guard initialization performed metadata checks of the environment directory itself; this is not a read of original source content.
- Attempted filesystem namespace isolation with bwrap was unavailable: network namespace setup and then UID mapping were rejected by the host. Validation therefore uses the fresh venv, path guard and recorded file accesses; it does **not** claim an OS-mounted inaccessible original filesystem or a physically separate machine. Network isolation was not asserted. Existing system NVIDIA driver/OS libraries are documented dependencies.

## Executed checks

| Check | Command (from repository, venv Python) | Outcome |
|---|---|---|
| Pure numerical parity | `python tests/compare_baseline.py --reference-dir <private-pristine-copy> --output <new-report.json>` | PASS: 16 cases, 10,514 array elements and 15,390 scalar comparisons; every compared intermediate and final output is exactly equal. |
| Retained function bodies | Included in compare_baseline.py | PASS: 40 retained functions/classes match original AST bodies after ignoring docstrings and package import qualification. |
| Power and RC contracts | `python -m unittest discover -s tests -p test_contracts.py` | PASS: two tests covering M=1/3/20 power conservation, finite output, deterministic frozen RC and RC-only source identity. |
| E01 integration smoke | `python scripts/run_experiment.py --config configs/E01-smoke.json` | PASS: 3/3 steps; generated ground scene, RT, extraction, fusion, CIR, NPZ and summary. |
| E02 original main defaults | `python scripts/run_experiment.py --config configs/E02-nyc-hybrid.json` | PASS: 5/5 city Hybrid steps, unmodified original settings. |
| E03 original optional RC branch | `python scripts/run_experiment.py --config configs/E03-nyc-consistent-rc.json` | PASS: 5/5 RC-only steps; absolute delays. |
| Actual-city same-input parity | `python tests/compare_real_rt.py --reference-dir <private-pristine-copy> --run <E02-run> --output <new-report.json>` | PASS: one 3,000,000-sample city RT solution; original and extracted path extraction/filtering, complete Hybrid intermediates, frozen state and three RC evaluations match exactly. 852 array elements / 1,286 scalars checked. |
| Repeated seeded city RT | Included in compare_real_rt.py | First city position has identical shape and exact H/tau equality to the earlier E02 run on this machine. Not a cross-GPU determinism claim. |
| E04 sampling diagnostic | `python scripts/validate_lsp_sampling.py --samples 10000 --seed 42 --elevation-deg 40 --output <new-file.json>` | EXECUTED: finite moments/covariance reported. This is a diagnostic, not a standard-conformance pass. |
| Plot/CSV/CFR export | `python scripts/analyze_run.py <run> --trusted-local-output` | PASS for original E01, E02 and E03; output plot visually inspected, with readable axes and no clipped labels. |
| Second clean install | README installation sequence and E01 | PASS: locked dependencies, package installation, pip check and smoke. |
| Source integrity | Rehash both source trees after work | PASS: 82,815 entries, including 80,309 regular files and 99 symlinks; no content/mtime/permission changes, added/deleted entries, read errors or unstable files. |

The synthetic parity fixture tests seeds 7, 42, 123 and 2026 with M=1,3,20, unequal RX/TX array sizes, sampled LSPs, clusters, ray PDPs, XPR, all phases and final CIRs. Frozen-state comparisons include t=0,0.01,0.2 seconds. Same-input comparisons use **rtol=0, atol=0**. The actual RT check uses the same solved Paths object for both extractors so any solver randomness is separated from extraction behavior.

The initial power-invariant test accumulated many powers in float32 and failed a 2e-7 relative threshold (observed reduction difference approximately 1.79e-6). Only the **test's reduction precision** was changed: it now converts stored complex64 coefficients to complex128 before accumulating powers. The threshold remains rtol=2e-7, atol=0. No channel equation, parameter or stored coefficient precision changed; source parity remained exact.

## Recorded runs and resource use

| Run ID | Valid steps | H shapes / observations |
|---|---:|---|
| E01-smoke_20260923T064239Z_54f3272a | 3 | (2,2,5,3), (2,2,6,3), (2,2,4,3); 13/16/10 rays. |
| E02-nyc-hybrid_20260923T064436Z_a5197821 | 5 | N=5,6,5,5,4 with U=S=4, Mmax=3; 13/16/13/13/10 rays. |
| E03-nyc-consistent-rc_20260923T065100Z_e90bd4b6 | 5 | (4,4,3,3) at every step, 9 RC rays; no deterministic RT rays in output. |
| E01-smoke_20260923T070107Z_8ae19bd5 | 3 | Second clean README environment; passed. |

E02 took 6 min 23.74 s wall time including tracing/wrapper, with maximum reported RSS 1,540,052 KiB. E03 took 6 min 5.56 s and 1,553,324 KiB. These are observations of the timed command, not aggregate GPU memory measurements or guaranteed hardware minima. The smoke driver's measured simulation section took 0.78 s; full cold import/setup takes longer. Peak GPU allocation was not measured. The private scene contains 2,906 input files; all bytes were copied and hash-checked before execution.

The legacy 40-degree LSP matrix has minimum eigenvalue about 0.129988 in this run, so its diagonal-stabilization fallback was not needed for this configuration. This does not validate the matrix against a standard. The model's angle clipping and other preserved assumptions remain documented.

## Git and provenance

No pre-existing Git author identity was available on the tested host or local machine. A request for identity/commit preference was made; no identity is fabricated. Until a real identity is supplied, there are no commits and recorded `git_commit` is null. Exact source/package hashes and configuration/data snapshots identify the checked code. The repository is initialized and the intended public file set can be reviewed separately from ignored/private content.

Original paths are stored in the private source-location record. Public provenance uses root-relative source names plus complete SHA-256 and extraction line ranges. Original line numbers always refer to the pre-work source snapshot, not the translated packaged files. Copy diffs record documentation, package imports, required-subset extraction and CLI I/O changes.

Delivery checks: both the tested host and local review copy have the same explicit 46-file Git candidate set. The tracked-file credential/path/size scan found zero findings; this is not automatic license clearance. Markdown file links, source/destination SHA-256 records, retained upstream license hashes and Git whitespace checks passed. Final text line endings and a trailing blank line were normalized without an AST change; the final package was rebuilt, installed and passed the exact 16-case/40-function parity check again. Earlier run metadata retains the hashes of the files actually used in those runs.

## Not executed / not established

- Complete conference-paper reproduction: paper specification, figure IDs, statistical ensembles and acceptance tolerances were not supplied.
- Independent OpenNTN ensemble/conformance comparison: the original separate reference was inspected and licensed, but not executed or silently replaced with a new seeded experiment.
- The historical 3-billion-sample comparison scripts, alternative v2/v3 models, TLE/ephemeris trajectory, generic channel selector and full Hybrid RT identity tracking.
- CPU-only RT, other operating systems/Python versions, different GPUs/drivers, cross-machine bitwise determinism or peak VRAM profiling.
- Public redistribution rights for the NYC scene; source license selection for the team's code; full binary/wheelhouse redistribution clearance.

No original source was executed to obtain these results. Pristine-copy code was executed only inside the new workspace, with all writes/caches/results redirected there. The final content check compares both full original trees, not just the five extracted modules.

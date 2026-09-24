# Setup and experiment guide

## Supported and tested environment

Test target: **Ubuntu 24.04.3 LTS, x86-64, Python 3.12.3**, NVIDIA RTX 5080 (16 GB), driver **575.64.03**. The CPU is an AMD Ryzen 9 9950X3D (16 cores / 32 logical CPUs); the machine has 249 GiB RAM. These are observed test resources, not minimum requirements. See the validation report for measured runtime/RSS. Other Linux versions, Windows, macOS, CPU-only RT and multi-GPU execution are unverified.

The selected runtime uses NumPy 1.26.4, SciPy 1.16.1, Matplotlib 3.10.5, Sionna RT 1.2.1, Mitsuba 3.7.1 and Dr.Jit 1.2.0. Sionna PHY, TensorFlow and OpenNTN are not required for these experiments. A separate CUDA toolkit was not installed; the tested GPU path uses the prebuilt wheels and the existing NVIDIA driver. CPU RT needs an LLVM backend, but no CPU RT support claim is made here.

Plan approximately 3 GB free disk for an isolated environment, package downloads and small runs; the private scene copy is approximately 60 MB. A GPU with 16 GB was used for the original settings. CPU/RAM/GPU minima have not been established; small sampling-only checks do not require a GPU. GPU memory grows with RT sampling and path limits.

## Clean installation

From your local repository checkout, create the tested Linux environment:

```bash
cd HybridChannel
python3.12 -m venv .venv
mkdir -p .cache/pip tmp
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/tmp"
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m pip install -r requirements-linux-py312.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation .
.venv/bin/python -m pip check
```

Ubuntu needs a working Python 3.12 venv module and driver before these commands. If `ensurepip` is unavailable, the system administrator must supply Ubuntu's `python3.12-venv` package. Do not install into an existing research environment. The wrapper places HOME, caches, temporary files, logs and outputs inside each new run. No source-code editing is required.

The lock contains only the clean environment's dependency closure, including Sionna RT's required visualization dependencies, plus build tools. It is not a dump of the original research environment. The Python package itself is installed in the final command. [License audit](license_audit.md) documents the pinned independent OpenNTN reference version.

## Smoke test and analysis

```bash
.venv/bin/python scripts/run_experiment.py --config configs/E01-smoke.json
# Copy the exact RUN_DIR printed above:
.venv/bin/python scripts/analyze_run.py runs/E01-smoke_<printed-suffix> --trusted-local-output
.venv/bin/python -m unittest discover -s tests -p test_contracts.py
```

E01 creates a clearly labeled synthetic ground-plane XML, uses 5,000 RT samples/source, 3 trajectory positions and 2-by-2 arrays. It exercises loading, LOS/full RT calls, extraction/filtering, local LSP sampling, fusion, CIR generation and output. This is an integration check, not a city or conference result. The tested run has three valid steps and H shapes `(2,2,5,3)`, `(2,2,6,3)`, `(2,2,4,3)`; counts may differ on untested backend versions. An empty or partially skipped trajectory is not reported as a pass by the wrapper.

Each run contains `config.json`, `environment.json`, `data_manifest.json`, `run.log`, `summary.txt`, `channel.npz`; analysis adds `metrics.csv`, `cfr.npz`, `pdp_cfr.png`, `analysis.json`. Repeated execution creates a new timestamp/UUID directory. Analysis refuses to overwrite an existing `analysis/` directory.

## Original research configurations

First obtain the exact NYC XML and all referenced meshes legally and place them according to [data/README.md](../data/README.md). The private local review already has a verified copy on the tested host; it is not part of Git or the portable public candidate.

```bash
.venv/bin/python scripts/run_experiment.py --config configs/E02-nyc-hybrid.json
.venv/bin/python scripts/run_experiment.py --config configs/E03-nyc-consistent-rc.json
# Alternate licensed scene without editing Python (a different experiment):
.venv/bin/python scripts/run_experiment.py --config configs/E02-nyc-hybrid.json --scene /absolute/path/to/scene.xml
# Diagnostic of the retained sampler, not independent OpenNTN validation:
.venv/bin/python scripts/validate_lsp_sampling.py --samples 10000 --seed 42 --elevation-deg 40 --output runs/lsp-diagnostic-001.json
```

E02 preserves the original main CLI defaults: 28 GHz, 600 km scene altitude, nominal 40 degrees elevation, 7,560 m/s, 0.2 s, five positions, 3,000,000 RT samples/source, depth 5, up to 2,000 paths, -25 dB RT pre-filter, 4-by-4 arrays, seed 42. E03 changes only the existing consistent-RC flag. Both are original executable research configurations; associating them with a specific submitted paper is still pending. Other historical scripts sometimes use 3,000,000,000 samples; those separate experiments are not silently substituted. [Experiment catalog](experiments.md) lists scope and current evidence.

## Reproducibility and interpretation

RT LOS solves use `seed`; full RT uses `seed + 10000 + step`; independent Hybrid draws use `seed + step`; consistent RC draws once at t0. NumPy RNG output is pinned to the tested version. GPU floating-point arithmetic and RT path discovery may differ across hardware/backends.

Scene positions/array offsets are in meters, time/delay in seconds, carrier input in GHz, path angles in radians, LSP angular spreads in degrees and K in dB. The coordinate frame is the scene's local XYZ frame; +z is vertical. H is complex64 with `(RX antenna, TX antenna, cluster, ray)` axes and zero-padded invalid rays. Tau has `(cluster, ray)` axes. Hybrid tau is **excess delay**; consistent-RC tau is **absolute delay**. Never compare their absolute phases without accounting for these conventions. Powers are linear channel gains, not an unconditional received-power value in watts. Details and known inconsistencies are in [model_and_units.md](model_and_units.md).

`channel.npz` retains the original ragged object-array representation. Only load trusted locally generated files with `allow_pickle=True`; analysis requires the explicit trusted-output flag. Coherent CFR summation differs from the incoherent ray-power PDP. Sampled LSPs are generation inputs, not guaranteed realized channel LSPs. The retained approximate correlation matrix, nearest-elevation lookup, C_tau multiplication, theta-only assembly and light-speed conventions are documented, not corrected.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named hybridchannel` | Install `.` into the same venv used for execution; do not add the original project to PYTHONPATH. |
| Scene missing | Supply a legally obtained self-contained XML/mesh directory; see data instructions. |
| Missing LLVM/CUDA backend | Verify the tested driver and wheels. CPU RT is unverified; do not silently change backend for a paper comparison. |
| No LOS / skipped steps | Inspect run.log and receiver/scene geometry. The baseline explicitly skips non-LOS steps. |
| A partially valid run exits 2 | The wrapper intentionally distinguishes partial trajectories from successful end-to-end runs. |
| Local-sampler moment mismatch | The original code stabilizes an approximate covariance and clips angular spreads; see known issues. |
| Analysis directory already exists | Use another run or an explicitly separate analysis workflow; existing results are preserved. |
| Missing Git commit in metadata | No commit exists or Git identity is unresolved. Exact package hashes are still recorded; do not invent an author. |

## Rights, citation and remaining work

The owner states that the original research code is held by the team and may be published. The team must choose its LICENSE before release; **no repository-wide license is implied**. Upstream license copies and citations are retained separately in [third_party](../third_party/README.md). NYC meshes/XML remain a distribution blocker. Publication venue, paper title, authors, DOI, artifact DOI, GitHub URL and figure/table numbering are **TBD**, not fabricated. Cite the actual paper once approved, and credit Sionna RT and any OpenNTN reference actually used.

The validation report distinguishes smoke execution, exact copy parity, original-setting runs, independent reference validation and complete paper reproduction. The last two are not established by smoke success. See [release checklist](release_checklist.md) before any publication.

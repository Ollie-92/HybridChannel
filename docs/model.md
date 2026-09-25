# Channel model and units

The primary channel-generation function is `generate_full_channel`, called by the trajectory CLI. This document describes its calculations and parameter conventions.

## Code guide

| Module | Role |
|---|---|
| [trajectory.py](../Hybrid_channel/trajectory.py) | Load the scene, step through satellite positions, run RT, and save channels |
| [cluster_splitter.py](../Hybrid_channel/cluster_splitter.py) | Extract RT path delays, powers, angles, and LOS labels |
| [lsp_calculator.py](../Hybrid_channel/lsp_calculator.py) | Sample correlated large-scale parameters from the local NTN table |
| [hybrid_channel.py](../Hybrid_channel/hybrid_channel.py) | Generate RC clusters, anchor them to RT, and assemble complex channel coefficients |
| [consistent_random_clusters.py](../Hybrid_channel/consistent_random_clusters.py) | Retain fixed RC scatterers and update their channels along the trajectory |
| [__init__.py](../Hybrid_channel/__init__.py) | Expose the package entry point |

Start with `run_simulation.py` for configuration handling, then `trajectory.py` for the simulation loop and `hybrid_channel.py` for channel generation.

## Data flow and calculations

1. `trajectory.build_scene` loads a Sionna XML, uses scalar vertical isotropic TX/RX probes and sets each supported material's scattering coefficient to 0.25 and pattern to Lambertian. The original NYC XML maps its materials to concrete. No material optimization was introduced.
2. Each step runs an LOS-only solver followed by a full solver. Without a valid LOS it skips the step. The extractor takes probe `[rx,0,tx,0,path,time]`, drops delays below zero and amplitudes at/below 1e-15, computes `abs(a)**2`, and reads four RT angles. The native complex RT phase is discarded by this interface.
3. The pre-filter retains powers within 25 dB of the maximum and sorts by delay. Each record has `tau_rt` in seconds, `power_rt` in linear gain, `aoa/aod/zoa/zod` in radians.
4. The local table selects the nearest elevation (10 to 90 degrees). The fixed legacy NTN table is used regardless of carrier band. LSP vector order is `[K, DS, ASD, ASA, ZSD, ZSA]`; K is dB, the other latent values are log10(seconds) or log10(degrees). The sampler applies Cholesky to the stored 6-by-6 matrix after diagonal stabilization if needed, draws six normals, transforms to linear units and clips azimuth/zenith spreads at 104/52 degrees. SF is not sampled by this six-variable path.
5. RT delays become excess delays relative to the earliest retained RT path. With `L` retained RT paths and `N` candidate RC clusters, the code uses `mu=max(r_tau*DS + L/(N+1)*(r_tau*DS-mean(tau_RT)), mean(tau_RT))`. RC exponential draws are sorted and shifted to zero. The LOS polynomial `C_tau` **multiplies** output RC delays; unscaled delays remain the input for power and duplicate removal.
6. The first candidate is removed. Other candidates are removed when their unscaled delay is within `mu*log(1/(1-p0))` of any RT excess delay; `p0=0.2`. This is RT-conditioned generation, not an independent RC channel joined afterwards.
7. Virtual RC and RT powers combine exponential delay decay and separately sampled lognormal shadow terms. The shared virtual sum is normalized; a K-dependent component is added to the first virtual RT path. Actual RC powers are multiplied by `sum(actual_RT_power)/sum(virtual_RT_power)`. Actual RT powers remain the measured inputs. No final unit-sum normalization is added.
8. RC angles use local sampled angular spreads, power ratios, Gaussian/sign perturbations and RT power-weighted angular centers. Radians/degrees are converted at the indicated boundaries. Clusters are merged, thresholded against the strongest combined cluster and sorted by delay. A 1 ns grid accumulates PDP powers.
9. The first LOS cluster contributes one ray. Other RT and RC clusters both expand to M=3 equal-power, same-delay rays by default (`K_B=1`). Offsets are `[0,1.2247,-1.2247]`. Non-3/1 values use SciPy normal quantiles. This is not a full twenty-ray OpenNTN random-coupling implementation. XPR and four initial phases are sampled, but only the theta-theta phase contributes to the scalar coefficient assembly.
10. For array offsets `d_rx,d_tx`, the coefficient uses `sqrt(P)*exp(j*phase_tt)*exp(j*k*r_rx.dot(d_rx))*exp(j*k*r_tx.dot(d_tx))*exp(j*k*r_rx.dot(v)*t)`. Original code uses wavelength `0.3/f_GHz`. The driver supplies satellite velocity to this receiver-direction Doppler term. Its physical interpretation must be reviewed separately; the extraction does not fix it. H is cast to complex64. Missing ray slots have H=0.

## RC consistency branch

`freeze_random_cluster_state` samples once at t0 and retains only RC clusters. Arrival direction, a one-bounce scatterer position/range, shadow term, ray offsets, XPR and phases are stored. Nonpositive excess-delay and ill-conditioned LOS-aligned clusters are dropped with counts. Per-step evaluation uses exact satellite-to-scatterer plus scatterer-to-UT distance, **absolute** delay and analytic carrier phase. UT arrival angles are frozen; departure center follows the satellite-to-UT direction. Current RT delays/powers provide only a power anchor. The evaluator takes no RNG and its dummy RNG rejects accidental draws.

This is deterministic evaluation of a fixed RC realization. It does not supply stable deterministic RT IDs, RT birth/death fading, moving UTs, multiple-UT random fields or a complete spatially consistent Hybrid channel. The nominal `t` argument is not a separate source of randomness or Doppler in this branch.

## Dimensions and conventions

| Object | Shape / unit | Meaning |
|---|---|---|
| RT path input | list of L records | seconds, linear gains, radians |
| LSP correlation | 6 by 6 | original approximate matrix |
| array geometry | U by 3 / S by 3, meters | half-wavelength ULA along scene x |
| `H` | U by S by N by Mmax, complex64 | per-ray CIR coefficients; not summed CFR |
| `tau` | N by Mmax, float64 seconds | excess in Hybrid, absolute in RC consistency |
| trajectory | steps by 3, meters | scene-local XYZ, fixed z |
| `H_by_step` / `tau_by_step` | ragged object arrays | one tensor per valid step only |
| `valid_steps` | integer vector | indices into all trajectory positions |
| generated CFR | U by S by 129, complex | new analysis at offsets -10 to +10 MHz |

## Known issues preserved

Development tests and original numerical comparisons are preserved on the [archive/research-validation branch](https://github.com/Ollie-92/HybridChannel/tree/archive/research-validation), with instructions for that branch's layout. Independent 3GPP/OpenNTN conformance, a numerical reference baseline for this layout, and cross-machine tolerances have not been established. A fixed seed alone does not guarantee identical output across hardware.

- No general channel-mode selector, orbit/TLE manager, per-step independent OpenNTN validation or full Hybrid spatial-consistency manager exists in the selected path.
- Approximate NTN correlation and XPR parameters, band-independent legacy tables and covariance diagonal adjustment can alter the intended target moments. Nearest-elevation lookup is discontinuous.
- The local delay polynomial convention differs from the separately audited OpenNTN implementation (multiplication versus division). This is a research-review issue, not a silent portability fix.
- The driver/model mix 299,792,458 and 300,000,000 m/s conventions. Hybrid delays are normalized while the LOS geometric phase uses absolute distance. These conventions affect phase and must remain identified.
- Default Hybrid clusters are resampled every step. Only the optional RC branch freezes random identity.
- Native RT complex phase/polarization is not preserved through the scalar power interface. XPR/cross phases are generated but unused in theta-only assembly.
- Original summary `mean_channel_power` includes zero padding; it is not a path-loss or received-power estimator. Analysis labels per-ray power sums separately.
- No algorithmic bug fixes were applied. A proposed corrected model must receive a separate version, baseline comparisons and paper-impact review.

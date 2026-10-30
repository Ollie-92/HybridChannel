# Native OpenNTN multi-element arrays

The adapter uses the official AntennaArray -> scenario -> set_topology -> channel_model -> cir_to_ofdm_channel flow from example/channel_response.ipynb (cells 6, 8, 10, 15, 27).

Upstream: https://github.com/ant-uni-bremen/OpenNTN
Pinned TensorFlow revision: cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2. The installed runtime was not upgraded to the current PyTorch-based upstream main.

## Notebook use
Set mode to "3gpp" and topology to "fixed". Configure each endpoint's num_rows, num_cols, vertical_spacing, horizontal_spacing, polarization and orientation_deg in the existing antenna fields. No extra array configuration is required. Spacing is in wavelengths; orientation is in degrees and is passed to native topology in radians. iso maps to native omni. Supported polarizations are V, H and VH. Arbitrary element offsets without a regular antenna specification remain unsupported. This update does not add OpenNTN orbit spatial consistency.

The notebook uses Sionna PlanarArray to define common element positions. Tests compare these positions against rotated native OpenNTN element positions. The native channel applies the antenna response; the adapter does not apply it again.

## Port order
Native OpenNTN uses polarization-major ports: all V elements, then all H elements for VH. Public ChannelResult and CSV files use element-major order: element 0 V/H, element 1 V/H, etc. Element index is column-major (row changes fastest). The same permutation is applied to CIR and CFR in both directions. Single-polarized ordering is unchanged.

channel.npz retains openntn_a, openntn_tau and openntn_cfr in native order, plus output_to_native_rx_port and output_to_native_tx_port. summary.json records dimensions, local native port positions and per-endpoint port maps.

## Executed validation
Run: python -B hybrid_simulator/validate_openntn_mimo.py
Outputs: hybrid_simulator/results/validation/openntn_mimo/validation.csv and validation.json.
12 cases passed at 2 GHz, seed 12345, four time samples and five frequency offsets, on TensorFlow CPU. Maximum absolute differences were zero for raw native CIR/delays/CFR and reordered public CIR/CFR/H. This verifies adapter parity for these cases; it is not an independent validation of the physical model or a guarantee for untested configurations.

Coverage: downlink/uplink; single V and VH regression against native Antenna; 4-element satellite/2-element UE V arrays; 2x2 satellite and 1x2 UE VH arrays; asymmetric mixed H/VH arrays; DenseUrban/Urban/SubUrban; LOS/NLOS; nonzero orientation; unequal 0.7/0.4 wavelength spacings; Doppler on/off.

Each case folder contains config.json, native_reference.npz, channel.npz, summary.json and readable CSV exports. Per-tap angles and LOS flags remain unavailable in aggregated native OpenNTN outputs; they are not invented by the adapter. Multi-element port counts multiply by two for VH.

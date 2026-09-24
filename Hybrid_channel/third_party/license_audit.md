# License and provenance audit

Audit date: **2026-09-23**. This is an evidence-based release review, not a definitive legal opinion. Public visibility, paper citation and a research-use statement are not treated as redistribution grants. The selected runtime's usage is distinguished from independent reference scripts.

## OpenNTN identity and immutable versions

The installed `openntn-0.1.0.dist-info/direct_url.json` records `https://github.com/ant-uni-bremen/OpenNTN.git` at **cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2** (requested revision main). The source reference script imports `OpenNTN.DenseUrban`, `Antenna`, `AntennaArray` and topology utilities. This establishes repository identity; it does not by itself establish that every installed file is unmodified.

The reviewed reference is [that exact OpenNTN commit](https://github.com/ant-uni-bremen/OpenNTN/tree/cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2). A separate current-main check found **f294df0480f31a4148e8fe4af9b17dd4abcdb869**; it was not substituted for the installed version. Current main's setup constrains Sionna to >=1.0,<2.0, whereas the installed historical commit's setup.py has an unbounded `sionna` requirement. A future independent-reference environment must pin its full dependency closure.

Evidence reviewed: LICENSE, README installation/citation/license sections, setup.py, repository tree, and headers of OpenNTN/lsp.py, rays.py, channel_coefficients.py plus the installed utils.py. No separate NOTICE appeared in the inspected OpenNTN repository tree. Model JSON assets are present upstream; none is copied into this release candidate.

| Question | Evidence-based answer |
|---|---|
| Academic use? | The root MIT grant permits use without a research-only restriction, subject to its conditions. Apache-marked material has its own applicable terms. |
| Modification? | MIT permits modification; Apache-2.0 also permits derivatives subject to its terms. This does not erase embedded third-party notices. |
| Public redistribution? | Permitted by the relevant grants when their conditions and scope are satisfied; do not treat all OpenNTN files as MIT-only. |
| Include in this repository? | A pinned external dependency/reference is feasible. Vendoring the entire mixed-notice tree is **not cleared by this audit** without file-by-file scope reconciliation. No OpenNTN source is included. |
| Citation? | README requests citation of the OpenNTN WSA 2025 paper. This academic request is separate from software-license conditions. |

The root [LICENSE, lines 1–21](https://github.com/ant-uni-bremen/OpenNTN/blob/cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2/LICENSE#L1-L21) identifies MIT and copyright (c) 2025 Arbeitsbereich Nachrichtentechnik. Its notice condition is at lines 10–11. In contrast [lsp.py lines 1–7](https://github.com/ant-uni-bremen/OpenNTN/blob/cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2/OpenNTN/lsp.py#L1-L7), [rays.py](https://github.com/ant-uni-bremen/OpenNTN/blob/cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2/OpenNTN/rays.py#L1-L7) and channel_coefficients.py carry NVIDIA copyright and Apache-2.0 SPDX identifiers. The root/file scope is therefore recorded as mixed evidence; no unilateral relicensing conclusion is made.

[Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0#redistribution) addresses providing a license copy, marking changed files, retaining relevant source notices and handling applicable NOTICE content. Sections 2/3 are the copyright/patent grants; section 6 limits trademark rights. These obligations are distinct from paper citation. Root MIT has no explicit mandatory change-log clause, but this project records changes for scientific provenance regardless.

## Components actually reached or independently referenced

| Component/version | Origin and license evidence | Use in this extraction | Rights/conditions and public treatment |
|---|---|---|---|
| HybridChannel research source | Per-file SHA-256 and original line ranges in ../docs/provenance.json; no source LICENSE found | Copied/extracted and minimally modified | Owner confirmed team ownership and permission to publish. Project license choice pending; no blanket open-source qualification claimed. |
| Sionna RT 1.2.1 | [v1.2.1 LICENSE](https://github.com/NVlabs/sionna-rt/blob/v1.2.1/LICENSE), Apache-2.0 notice; README license/citation section | Runtime dependency, no vendored code | Research/modification/redistribution permitted subject to Apache terms. Keep license and source notices; mark modifications if any. No upstream code modifications here. |
| Mitsuba 3.7.1 | [v3.7.1 LICENSE](https://github.com/mitsuba-renderer/mitsuba3/blob/v3.7.1/LICENSE), BSD-style redistribution conditions | Runtime dependency | Retain copyright/conditions/disclaimer for redistributed source/binaries; no endorsement. Third-party components in wheels retain their own terms. |
| Dr.Jit 1.2.0 | [v1.2.0 LICENSE](https://github.com/mitsuba-renderer/drjit/blob/v1.2.0/LICENSE), BSD-style conditions | Runtime dependency | Same source/binary notice obligations; inspect packaged third-party notices before bundling binaries. |
| NumPy 1.26.4 | [v1.26.4 LICENSE.txt](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt), BSD-3-Clause | Runtime dependency | Preserve copyright, conditions and disclaimer; no endorsement. Binary wheels can contain additional library notices. |
| SciPy 1.16.1 | [v1.16.1 LICENSE.txt](https://github.com/scipy/scipy/blob/v1.16.1/LICENSE.txt), BSD-3-Clause | Needed for alternate M; parity test includes M=20 | Same conditions; bundled binary libraries have additional notices. |
| Matplotlib 3.10.5 | [versioned LICENSE](https://github.com/matplotlib/matplotlib/blob/v3.10.5/LICENSE/LICENSE), PSF-based Matplotlib license | Added plotting infrastructure | Preserve license/notice and applicable modification summaries when distributing derivatives. Matplotlib itself is not modified. |
| Sionna PHY 1.2.1 / TensorFlow 2.20.0 | Original environment metadata; Sionna [v1.2.1 LICENSE](https://github.com/NVlabs/sionna/blob/v1.2.1/LICENSE), Apache-2.0 | Independent original OpenNTN reference only; absent from selected runtime | Not shipped. Any future reference environment must audit its actual installed closure; no broad binary redistribution clearance is inferred. |
| OpenNTN 0.1.0 at cbb0e370… | Immutable URLs above; MIT root and Apache-marked files | Independent reference, not imported by extracted core | Scope reconciliation required before vendoring. MIT and Apache texts retained for review. |
| Legacy GPP_TABLE and numeric constants | Team source cites TR 38.811/38.901; no standalone dataset grant or exact provenance chain supplied | Embedded parameter values preserved | Values/approximations are documented. Author should confirm derivation and required standards attribution. No standards PDF, figure or long text is bundled. Standards conformance is unverified. |
| NYC XML + PLY | YuChein/NYC_scene; no LICENSE/authoritative acquisition URL found | Private copied input for research validation | **尚無法確認 / not established** for public redistribution; excluded from Git and portable candidate. Owner must provide source, license and version. |
| Supplied flowchart images | User attachments; no separate publication license supplied | Read as mapping requirements | Original image files not distributed; fresh textual/Mermaid mapping is included. |
| Weights, ephemerides, models | No such runtime inputs on selected call path | None | Not included. IONORT and ionosphere research are out of scope. |

The clean-environment dependency inventory and lock record transitive package versions. Dependencies are installed from upstream distributions; their source or binary wheels are **not** bundled in this public candidate. Upstream licenses for direct scientific dependencies are copied under licenses/, and installed-package license metadata is retained for traceability. Absence of a license field is not interpreted as permission. Redistribution of a complete wheelhouse/container would require a separate full binary-notice audit.

## Project-license decision

No root LICENSE is created. MIT, BSD-3-Clause or Apache-2.0 could be considered for team-owned original work after authorship and compatibility review. This is a suggestion, not a license selection or a declaration that every asset is cleared. In particular, choosing a project license cannot grant rights to the NYC scene or supersede upstream notices.

## Academic attribution

The OpenNTN README identifies its paper as “OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations,” WSA 2025, with authors T. Düe, M. Vakilifard, C. Bockelmann, D. Wübben and A. Dekorsy. Consult the [pinned citation section](https://github.com/ant-uni-bremen/OpenNTN/blob/cbb0e37079edeaf04f78d25b2c8e5b0539ad0df2/README.md) when the independent reference is actually used. Sionna RT's pinned README provides its software citation. No author, venue or DOI is invented for the user's conference paper.

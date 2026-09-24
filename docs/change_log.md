# Extraction change log

These changes apply only to the new independent copy. Original files were neither reformatted nor executed.

1. **Inventory and licenses:** full source file/SHA-256 baseline, immutable OpenNTN evidence, dependency-license copies, private scene publication gate, owner decisions.
2. **Code extraction:** selected HybridChannel call path, removed unused terrestrial registry entries and optional TensorFlow CIR conversion, retained the required Sionna extractor rather than unrelated splitting experiments. Original calculations and RNG order are unchanged.
3. **Portability:** package-relative imports, explicit scene argument, cwd-relative paths in the low-level CLI, an installable Python package and pinned clean environment. No original absolute path is an execution dependency.
4. **Experiment infrastructure:** explicit JSONs, unique run directories, input SHA-256, environment/source/Git snapshots and logs; added analysis/diagnostic scripts with their new status disclosed.
5. **Documentation:** English module/function descriptions, actual execution graph, source-line mapping, known discrepancies and paper/publication gates.
6. **Validation:** exact pristine-copy intermediate/final comparisons, function-body checks, numerical invariants, GPU end-to-end smoke and original-setting private-scene runs. The validation report records actual outcomes and limitations.

The intended local commit sequence follows the six groups above. Actual commits require the owner's real Git identity; none is fabricated and no global Git setting is changed. Private patch files preserve exact source-to-copy differences independently of commit availability.

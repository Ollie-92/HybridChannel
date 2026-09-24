# Data acquisition and placement

The smoke experiment creates its own small test ground plane. It is not a replacement for the paper's urban environment.

E02/E03 expect a legally obtained **exact** scene at `data/external/NYC_scene/NYC_sionna.xml`, with every referenced `meshes/*.ply` in its original relative position. The inspected scene resolves to 2,906 distinct files including the XML. The two source roots contain the same-named scene; the selected data copy is taken from the original research source and verified against its original SHA-256 manifest.

No scene LICENSE, authoritative download URL, asset author or redistribution permission was found in the provided source inventory. Consequently **no download URL is guessed**, and these assets are excluded from Git and the portable public candidate. The owner must supply the asset source, version and applicable permissions or provide an approved acquisition route. Changing to a different scene is a different experiment.

For private local reproduction, copy legally possessed assets into the directory above (ordinary copies, no links). Do not point the runtime at the original research directories. The wrapper requires referenced meshes to stay inside the supplied scene directory and records every input hash. Runtime results and private absolute paths stay in ignored run directories.

No pretrained weights, external learned model, ephemeris, TLE, electron-density dataset or IONORT data is required by the selected HybridChannel path. No such resources are included.

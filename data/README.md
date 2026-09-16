# MaleCNS data

The `male-cns/` directory is created by `scripts/prepare_data.py`. No fabricated
connectome is bundled. Until measured data is present, the UI disables fly mode.

Official dataset: <https://male-cns.janelia.org/download/>

License: **CC BY 4.0**, <https://creativecommons.org/licenses/by/4.0/>.
Credit: FlyEM, HHMI Janelia Research Campus; University of Cambridge / MRC LMB;
Google Research and collaborators. Cite the MaleCNS v1.0 dataset and associated
paper when publishing derivatives.

The official preparation route downloads approximately 1.2 GB in three Feather
tables. Allow several additional GB of RAM while assembling the sparse matrix.
It keeps annotated neurons and all edges between them, then turns contact counts
into signed, normalized weights. This is a modeling choice, not measured efficacy.
Unknown transmitter signs are treated as positive. No learned memories are supplied.

`manifest.json` records local hashes and transformations after a successful load.
Official downloads use the project proxy configuration, retain partial files with
remote version metadata for resumption, and validate size, server MD5 when supplied,
and Feather format before promotion. TLS certificate checks remain enabled.
An optional `--from-dir` imports already prepared `weights.npz` / `brain.npz`
from fly.ai; the importer cannot independently authenticate arbitrary external files.

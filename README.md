# ISW-DETR reproduction

This repository accompanies the manuscript **ISW-DETR: Stage-specific state-contrast control for vision–weight fusion in fine-grained product detection**.

## Files and evaluation

The [fixed reproduction release](https://github.com/Tb-STAOO/ISW-DETR/releases/tag/sr-repro-20261108) contains the inputs for reference evaluation of the paper. [FILES_SHA256.json](FILES_SHA256.json) lists the expected files and checksums. 

1. Obtain the code from this repository or the release asset ISW_DETR_reproduction.zip.
2. Download private-dataset.zip, private-inference.zip, private-controls.zip, public-metadata.zip and the 54 public-checkpoints.zip.part* assets into one directory. These filenames identify the resource groups used by the code.
3. Save FILES_SHA256.json and reassemble_public_checkpoints.py in the same download directory. Run the reassembly command below to verify the parts and reconstruct public-checkpoints.zip.
4. Follow the [original reproduction guide](README_zh.md) for environment setup and evaluation. The C-CCM preparation tool obtains the original images from the [official C-CCM release](https://doi.org/10.5281/zenodo.4642577). The original configuration masses are from [CORSMAL Containers Manipulation](https://doi.org/10.17636/101CORSMAL1).

~~~sh
python reassemble_public_checkpoints.py --directory /path/to/downloads
python tools/import_resources.py --archives /path/to/downloads --output /path/to/assets --groups private-dataset private-inference private-controls public-metadata public-checkpoints
python tools/prepare_cccm.py --metadata-root /path/to/assets/public_metadata --data-root /path/to/cccm --work-dir /path/to/preparation
python tools/reproduce.py --phase all --assets-root /path/to/assets --public-data-root /path/to/cccm --output /path/to/results --device cuda
~~~

The release provides inputs for evaluation from the reference checkpoints. End-to-end retraining also requires the initialization and training resources described in the reproduction guide. Those resources are available from the corresponding author. C-CCM and CORSMAL source materials retain their original attribution and licence terms.

# Reproducing the evaluation

Run the repository's `pipeline.py` for the reported full-split results.
Both it and `pipeline_ablation.py` use the reported eight-feature rank selector
in `pose_solver/selection.py`. Each run records its resolved configuration,
source and input hashes, checkpoints, and environment in `run_manifest.json`.

The protocol is in [`configs/evaluation.json`](../configs/evaluation.json).
Dataset profiles and component settings are under `configs/pipeline/`.
Run the following commands from the repository root after completing the
README setup and dataset preparation. Use a new output directory for each run.

```bash
PYTHONPATH=. python pipeline.py \
  --dataset uavd4l --data_root data/UAVD4L-LoD/inTraj \
  --batchsize 512 --no-crop --out outputs/uavd4l_intraj

PYTHONPATH=. python pipeline.py \
  --dataset uavd4l --data_root data/UAVD4L-LoD/outTraj \
  --batchsize 512 --no-crop --out outputs/uavd4l_outtraj

PYTHONPATH=. python pipeline.py \
  --dataset swiss_epfl --data_root data/Swiss-EPFL/inPlace \
  --batchsize 512 --no-crop --out outputs/swiss_inplace

PYTHONPATH=. python pipeline.py \
  --dataset swiss_epfl --data_root data/Swiss-EPFL/outPlace \
  --batchsize 512 --no-crop --out outputs/swiss_outplace
```

The expected split sizes are 1,604 and 2,192 queries for UAVD4L-LoDv2, and 712 and
379 queries for Swiss-EPFLv2. Use the full split, the matching LoD mesh, the
aggregate pose files, and the split-specific configuration. Dataset files are
provided by their original sources; see [`data/README.md`](../data/README.md).

These query splits retain the original benchmark directory names shown in the
commands above. The main evaluations use LoD3 for UAVD4L-LoDv2 and LoD2 for
Swiss-EPFLv2.

Summarize a completed run from its saved poses and verify query coverage:

```bash
PYTHONPATH=. python scripts/summarize_results.py outputs/uavd4l_intraj
```

Recall is joint: both translation and rotation errors must be below the named
threshold. Each run records the code, configuration, inputs, checkpoints,
hardware, and package versions in its manifest. Keep output directories
separate for each evaluation.

Expected full-split metrics and query counts are recorded in
[`benchmark_results.json`](benchmark_results.json). The protocol identifier
in manifests remains stable so existing run records can still be verified.

Batch size and GPU arithmetic can affect candidate selection. The recorded
main evaluations use H100 NVL for both UAV splits and Swiss out-of-Place, and
A40 for Swiss in-Place. Match the recorded environment when assessing
repeatability. Saved-pose verification checks coverage and internal consistency,
not bitwise equality across hardware.

The benchmarks informed method development. These evaluations do not constitute
an untouched held-out validation of all pipeline choices.

## Ablations

Component and model comparisons use the fixed 100-query subset per UAV split
with seed 42. The LoD-map and pose-selection comparisons use complete splits.
Synthetic prior-noise experiments use 300 queries per split. Run configurations
and commands are documented in [`ABLATIONS.md`](ABLATIONS.md).

## Models and environment

The reference environment is Linux, Python 3.10, PyTorch 2.6, and CUDA 12.4.
Model revisions and installation commands are listed in the root
[`README.md`](../README.md). Model weights are obtained from their original
providers and are not stored in this repository.

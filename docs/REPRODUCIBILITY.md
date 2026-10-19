# Reproducing the evaluation

Use the versioned camera-ready launcher for the reported full-split results.
It validates the bundled source and configuration, records the run settings,
and retains the exact evaluation source alongside each output.

The reported results use the bundled eight-feature rank selector. Direct
`pipeline.py` and `pipeline_ablation.py` runs use the six shared render features
defined in `pose_solver/selection.py`. Use the camera-ready launcher for the
reported results and keep these configurations distinct when comparing runs.

```bash
python scripts/run_camera_ready.py verify

PYTHONPATH=. python scripts/run_camera_ready.py main -- \
  --dataset uavd4l --data_root data/UAVD4L-LoD/inTraj \
  --batchsize 512 --no-crop --out outputs/uavd4l_intraj

PYTHONPATH=. python scripts/run_camera_ready.py main -- \
  --dataset uavd4l --data_root data/UAVD4L-LoD/outTraj \
  --batchsize 512 --no-crop --out outputs/uavd4l_outtraj

PYTHONPATH=. python scripts/run_camera_ready.py main -- \
  --dataset swiss_epfl --data_root data/Swiss-EPFL/inPlace \
  --batchsize 512 --no-crop --out outputs/swiss_inplace

PYTHONPATH=. python scripts/run_camera_ready.py main -- \
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
PYTHONPATH=. python scripts/summarize_camera_ready.py outputs/uavd4l_intraj
```

Recall is joint: both translation and rotation errors must be below the named
threshold. Each run records the code, configuration, inputs, checkpoints,
hardware, and package versions in its manifest. Keep output directories
separate for each evaluation.

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

# Dataset setup

PlanarCityLoc evaluates on UAVD4L-LoDv2 (in-Traj/out-of-Traj) and
Swiss-EPFL-v2 (in-Place/out-of-Place). Obtain the evaluation data from its
original providers and arrange it under `data/` as described in
[`data/README.md`](../data/README.md).

## Download

Download the [LoD-Loc dataset](https://pan.baidu.com/s/1Ue37i0OQpS2jzo-fQ3o4bA?pwd=xxdv)
(code `xxdv`) and link it into the repository:

```bash
bash scripts/setup_data_links.sh /path/to/extracted/data /path/to/checkpoints
sha256sum --check data/SHA256SUMS
```

The script verifies and links the externally acquired meshes, poses, intrinsics,
split files, query images, and a local checkpoint directory.

```text
data/
├── UAVD4L-LoD/
│   ├── inTraj/Query_image/
│   └── outTraj/Query_image/
├── Swiss-EPFL/
│   ├── inPlace/Query_image/
│   └── outPlace/Query_image/
└── MODEL/ply/
    ├── UAVD4L-LoD3.ply
    ├── UAVD4L-LoD1.ply
    └── Swiss-EPFL-LoD2.ply
```

`Query_image_full/` is also accepted for Swiss-EPFL.

## Important

- Use the aggregate `GT_pose.txt` and `GPS_pose.txt` files from the dataset
  package. UAVD4L's
  per-image pose files encode rotations differently and are not compatible.
- The main paper table uses UAVD4L LoD3 and Swiss-EPFL LoD2. UAVD4L LoD1 is
  included only for the LoD ablation.
- Query images, evaluation metadata, meshes, and model checkpoints are not
  stored in this repository.
- UAVD4L priors are sensor-derived; Swiss priors were synthesized by the
  benchmark authors, not recorded as raw GPS. We use the released files unchanged.

See [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the exact paper
protocol and [data sources and attribution](../data/README.md#sources-and-attribution) for original sources,
processing, checksums, and redistribution terms.

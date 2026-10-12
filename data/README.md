# Evaluation data

Query imagery, camera and pose metadata, and 3D map meshes are obtained from
their original dataset providers and are not distributed here. See
[`DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) for source links and attribution.
Use the providers' current terms when obtaining or using these files.

Arrange the downloaded assets in this layout:

```text
data/
├── UAVD4L-LoD/
│   ├── inTraj/{GT_pose.txt,GPS_pose.txt,intrinsic.txt,Query_image/}
│   └── outTraj/{GT_pose.txt,GPS_pose.txt,intrinsic.txt,Query_image/}
├── Swiss-EPFL/
│   ├── inPlace/{GT_pose.txt,GPS_pose.txt,intrinsic.txt,Query_image/}
│   ├── outPlace/{GT_pose.txt,GPS_pose.txt,intrinsic.txt,Query_image/}
│   └── split/{Swiss_inPlace_test.json,Swiss_outPlace_test.json}
└── MODEL/ply/
    ├── UAVD4L-LoD1.ply
    ├── UAVD4L-LoD3.ply
    └── Swiss-EPFL-LoD2.ply
```

`Query_image_full/` can be used instead of `Query_image/` for Swiss-EPFL.
The helper checks the recorded file hashes and links the benchmark files,
the four query-image folders, and a local checkpoint directory:

```bash
bash scripts/setup_data_links.sh /path/to/extracted/data /path/to/checkpoints
```

The external data directory must use the layout above. The helper accepts
matching files already present in the checkout and preserves existing paths.
[`SHA256SUMS`](SHA256SUMS) records the expected file identities.

Use the aggregate `GT_pose.txt` and `GPS_pose.txt` files. UAVD4L's per-image
pose files encode rotations differently and are not interchangeable with the
aggregate files expected by this pipeline.

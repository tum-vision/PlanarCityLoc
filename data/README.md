# Evaluation data

Query imagery, camera and pose metadata, and 3D map meshes are obtained from
their original dataset providers and are not distributed here. See
[Sources and attribution](#sources-and-attribution) for source links and attribution.
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

## Sources and attribution

Evaluation imagery, annotations, metadata, and map geometry remain subject
to their original providers' terms and are not covered by the PlanarCityLoc
code license. The code licenses of dataset repositories do not automatically
cover separately downloaded data bundles.

| Data | Source | Attribution |
|---|---|---|
| UAVD4L query images and sensor data | [UAVD4L](https://github.com/RingoWRW/UAVD4L) and the [LoD-Loc dataset package](https://pan.baidu.com/s/1Ue37i0OQpS2jzo-fQ3o4bA?pwd=xxdv) (code `xxdv`) | Cite UAVD4L and LoD-Loc. |
| UAVD4L LoD1 and LoD3 meshes | LoD-Loc dataset package; derived from UAVD4L | Cite the source datasets. |
| Swiss-EPFL query imagery and evaluation metadata | [CrossLoc](https://doi.org/10.5061/dryad.mgqnk991c) and the [LoD-Loc evaluation package](https://github.com/VictorZoo/LoD-Loc#dataset-preprocessing) | Cite CrossLoc and LoD-Loc. |
| Swiss-EPFL LoD2 mesh | [swissBUILDINGS3D](https://www.swisstopo.admin.ch/en/landscape-model-swissbuildings3d-3-0-beta), converted from LV95/LN02 coordinates | Preserve the attribution **©swisstopo** and follow the [swisstopo terms](https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices). |

Source packages may distribute different file layouts. The expected paths
are listed above, and [`SHA256SUMS`](SHA256SUMS) records the identities of
the evaluation files used by this implementation. Obtain files from their
providers and follow the terms that accompany those files.

The Swiss evaluation priors are benchmark-provided pose matrices. Their names
do not imply they are raw GPS measurements. Use the released aggregate pose
files expected by the dataset readers.

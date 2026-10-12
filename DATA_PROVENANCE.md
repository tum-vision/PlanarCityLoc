# Data sources and attribution

Evaluation imagery, annotations, metadata, and map geometry are not included
in this repository. They remain subject to the terms of their original
providers and are not covered by the PlanarCityLoc code license.

| Data | Source | Attribution |
|---|---|---|
| UAVD4L query images and sensor data | [UAVD4L](https://github.com/RingoWRW/UAVD4L) and the [LoD-Loc dataset package](https://pan.baidu.com/s/1Ue37i0OQpS2jzo-fQ3o4bA?pwd=xxdv) (code `xxdv`) | Cite UAVD4L and LoD-Loc. |
| UAVD4L LoD1 and LoD3 meshes | LoD-Loc dataset package; derived from UAVD4L | Cite the source datasets. |
| Swiss-EPFL query imagery and evaluation metadata | [CrossLoc](https://doi.org/10.5061/dryad.mgqnk991c) and the [LoD-Loc evaluation package](https://github.com/VictorZoo/LoD-Loc#dataset-preprocessing) | Cite CrossLoc and LoD-Loc. |
| Swiss-EPFL LoD2 mesh | [swissBUILDINGS3D](https://www.swisstopo.admin.ch/en/landscape-model-swissbuildings3d-3-0-beta), converted from LV95/LN02 coordinates | Preserve the attribution **©swisstopo** and follow the [swisstopo terms](https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices). |

Source packages may distribute different file layouts. The expected paths are
listed in [`data/README.md`](data/README.md), and `data/SHA256SUMS` records the
identities of the evaluation files used by this implementation. Obtain files
from their providers and follow the terms that accompany those files.

The Swiss evaluation priors are benchmark-provided pose matrices. Their names
do not imply they are raw GPS measurements. Use the released aggregate pose
files expected by the dataset readers.

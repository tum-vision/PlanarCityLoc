# Third-party notices

PlanarCityLoc includes or depends on third-party software. Those components are
not relicensed under PlanarCityLoc's CC BY-NC 4.0 license.

## Vendored source

- **SAM 3** (`thirdparty/sam3/`) — Copyright Meta Platforms, Inc. and
  affiliates. Distributed under the [SAM License](thirdparty/sam3/LICENSE).
  The small cached `building` prompt feature in
  `assets/sam3_building_language_features.b85` remains subject to the same
  upstream license. Full model checkpoints are downloaded separately.
- **EoMT** (`thirdparty/eomt_repo/`) — Copyright Mobile Perception Systems Lab
  at TU/e. Distributed under the [MIT License](thirdparty/eomt_repo/LICENSE).

## Installed dependencies

- **MoGe / MoGe-2** is installed from Microsoft's repository at the commit
  pinned in `README.md`. MoGe is MIT-licensed; its bundled DINOv2 component is
  Apache-2.0-licensed. Refer to that repository for its complete notices.
- **nvdiffrast** is installed from NVIDIA's repository. Refer to that
  repository for its license and notices.
- Python packages installed through `requirements.txt` retain their respective
  licenses.

## Evaluation data

LoD meshes, evaluation metadata, and query images are obtained separately
from their original providers. They originate from the LoD-Loc dataset
release and are not relicensed under PlanarCityLoc's code license.
UAVD4L data originates from the UAVD4L project. Swiss-EPFL queries and metadata
originate from the CrossLoc benchmark; the processed Swiss LoD model is based
on swissBUILDINGS3D data. Source attribution for the Swiss geodata:
`©swisstopo`.

See `DATA_PROVENANCE.md` for source links, attribution, and file identities.
The code licenses of dataset repositories do not automatically cover separately
downloaded data bundles. Follow the providers' data terms.

Model checkpoints and query images are not included. Their use is subject to
the terms published by their providers.

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union
from collections.abc import Sequence

import numpy as np

from utils.io import load_intrinsics, parse_poses, scale_K
from utils.pose import CameraPose
from .base import BaseDataset


class SwissEPFLDataset(BaseDataset):
    """Swiss-EPFL / CrossLoc dataset (LoD-Loc layout).

    Pose text files use COLMAP w2c (same as UAVD4L).  Per-image ``*_pose.txt``
    files store c2w matrices and are inverted when present.
  """

    @staticmethod
    def _first_existing_path(*candidates: Union[str, Path]) -> Path:
        for candidate in candidates:
            p = Path(candidate)
            if p.exists():
                return p
        raise FileNotFoundError(f"None of these paths exist: {candidates}")

    @classmethod
    def _resolve_split_paths(cls, split_dir: Union[str, Path]) -> dict:
        """Resolve LoDLoc paths (full-release and 10-frame demo layouts)."""
        split_dir = Path(split_dir)
        return {
            "images_dir": cls._first_existing_path(
                split_dir / "Query_image_full",
                split_dir / "Query_image",
            ),
            "gt_pose_txt": cls._first_existing_path(
                split_dir / "GT_pose_full.txt",
                split_dir / "GT_pose.txt",
            ),
            "gps_pose_txt": cls._first_existing_path(
                split_dir / "GPS_pose_full.txt",
                split_dir / "GPS_pose.txt",
            ),
            "intrinsic_txt": cls._first_existing_path(
                split_dir / "intrinsic_full.txt",
                split_dir / "intrinsic.txt",
            ),
            "gt_pose_dir": split_dir / "GT_pose",
            "gps_pose_dir": split_dir / "GPS_pose",
        }

    @staticmethod
    def _load_c2w_pose_matrix(path: Union[str, Path]) -> np.ndarray:
        return np.loadtxt(str(path), dtype=np.float64).reshape(4, 4)

    @classmethod
    def _load_poses(cls, split_dir: Union[str, Path], pose_kind: str = "gt") -> dict:
        paths = cls._resolve_split_paths(split_dir)
        txt_key = "gt_pose_txt" if pose_kind == "gt" else "gps_pose_txt"
        pose_dir = paths["gt_pose_dir" if pose_kind == "gt" else "gps_pose_dir"]

        poses = parse_poses(str(paths[txt_key]))
        stem_to_key = {Path(name).stem: name for name in poses}

        if pose_dir.is_dir():
            for pose_path in sorted(pose_dir.glob("*_pose.txt")):
                stem = pose_path.stem.replace("_pose", "")
                key = stem_to_key.get(stem, f"{stem}.jpg")
                poses[key] = np.linalg.inv(cls._load_c2w_pose_matrix(pose_path))

        return poses

    def __init__(
        self,
        data_root: str,
        render_size: tuple[int, int],
        split: str = "inPlace",
        fname: str | Sequence[str] = None,
        n: int = None,
        random_seed: int = None,
    ):
        self.data_root = data_root
        self.rH, self.rW = render_size
        self.split = split

        split_dir = os.path.join(data_root, self.split)
        paths = self._resolve_split_paths(split_dir)
        self.images_dir = str(paths["images_dir"])
        self.intrinsic_path = str(paths["intrinsic_txt"])
        self.gt_pose_path = str(paths["gt_pose_txt"])

        self.gt_poses = self._load_poses(split_dir, pose_kind="gt")
        self.gps_poses = self._load_poses(split_dir, pose_kind="gps")
        self.gps_poses_stem = {os.path.splitext(k)[0]: v for k, v in self.gps_poses.items()}

        split_file = os.path.join(data_root, "split", f"Swiss_{self.split}_test.json")
        with open(split_file, "r") as f:
            split_data = json.load(f)
            _all_frames = split_data["val"][self.split]

        _all_frames = [f for f in _all_frames if f in self.gt_poses]

        if fname:
            fname_filters = [fname] if isinstance(fname, str) else [x for x in fname if x]
            self.frames = [f for f in _all_frames if any(token in f for token in fname_filters)]
            if not self.frames:
                raise ValueError(f"No frames found matching --fname={fname_filters!r}")
        elif random_seed is not None:
            rng = random.Random(random_seed)
            self.frames = sorted(
                rng.sample(_all_frames, min(n, len(_all_frames)) if n else len(_all_frames))
            )
        else:
            self.frames = _all_frames if n is None else _all_frames[:n]

    def resolve_gps_w2c(self, img_name: str):
        stem_name = os.path.splitext(img_name)[0]
        w2c_gps = self.gps_poses.get(img_name)
        if w2c_gps is None:
            w2c_gps = self.gps_poses_stem.get(stem_name)
        if w2c_gps is None:
            raise RuntimeError(f"No GPS prior for {img_name}")
        return w2c_gps

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for img_name in self.frames:
            img_path = os.path.join(self.images_dir, img_name)
            fx, fy, cx, cy, src_W, src_H = load_intrinsics(self.intrinsic_path, img_name)
            K = scale_K(fx, fy, cx, cy, src_W, src_H, self.rW, self.rH)
            w2c = self.gt_poses[img_name]
            gt_pose = CameraPose.from_w2c(w2c)
            w2c_gps = self.resolve_gps_w2c(img_name)

            yield {
                "img_name": img_name,
                "img_path": img_path,
                "K": K,
                "w2c_gt": w2c,
                "gt_pose": gt_pose,
                "w2c_gps": w2c_gps,
                "src_W": src_W,
                "src_H": src_H,
            }

    def get_w2c_gps_list(self) -> List[Any]:
        return [self.resolve_gps_w2c(name) for name in self.frames]

    def get_first_intrinsics(self) -> Dict[str, Any]:
        if not self.frames:
            return None
        fx, fy, cx, cy, src_W, src_H = load_intrinsics(self.intrinsic_path, self.frames[0])
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "src_W": src_W, "src_H": src_H,
        }

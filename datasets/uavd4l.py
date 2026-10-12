import os
import random
from collections.abc import Sequence
from utils.io import parse_poses, load_intrinsics, scale_K
from utils.pose import CameraPose
from .base import BaseDataset

class UAVD4LDataset(BaseDataset):
    def __init__(self, data_root: str, render_size: tuple[int, int], fname: str | Sequence[str] = None, n: int = None, random_seed: int = None):
        self.data_root = data_root
        self.rH, self.rW = render_size
        self.gt_pose_path = os.path.join(data_root, "GT_pose.txt")
        self.intrinsic_path = os.path.join(data_root, "intrinsic.txt")
        if os.path.isdir(os.path.join(data_root, "images")):
            self.images_dir = os.path.join(data_root, "images")
        elif os.path.isdir(os.path.join(data_root, "Query_image")):
            self.images_dir = os.path.join(data_root, "Query_image")
        else:
            raise FileNotFoundError(
                f"No images/ or Query_image/ under {data_root}"
            )
        
        self.gt_poses = parse_poses(self.gt_pose_path)
        self.gps_poses = parse_poses(os.path.join(data_root, "GPS_pose.txt"))
        self.gps_poses_stem = {os.path.splitext(k)[0]: v for k, v in self.gps_poses.items()}

        _all_frames = sorted(f for f in os.listdir(self.images_dir) if f in self.gt_poses)
        
        if fname:
            fname_filters = [fname] if isinstance(fname, str) else [x for x in fname if x]
            self.frames = [f for f in _all_frames if any(token in f for token in fname_filters)]
            if not self.frames:
                raise ValueError(f"No frames found matching --fname={fname_filters!r}")
        elif random_seed is not None:
            rng = random.Random(random_seed)
            self.frames = sorted(rng.sample(_all_frames, min(n, len(_all_frames)) if n else len(_all_frames)))
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

    def __iter__(self):
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

    def get_w2c_gps_list(self):
        return [self.resolve_gps_w2c(name) for name in self.frames]

    def get_first_intrinsics(self):
        if not self.frames:
            return None
        fx, fy, cx, cy, src_W, src_H = load_intrinsics(self.intrinsic_path, self.frames[0])
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "src_W": src_W, "src_H": src_H
        }

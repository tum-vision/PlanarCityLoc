"""
utils/pose.py
=============
Camera pose utilities: CameraPose class, Euler/quaternion/rotation helpers,
camera intrinsics I/O and scaling, and pose dict ↔ tensor conversions.

Merged from: utils/pose.py + utils/lodloc/geometry.py + utils/lodloc/io_utils.py
"""

import csv
import json
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ── CameraPose class ──────────────────────────────────────────────────────────

class CameraPose:
    """6-DoF camera pose with C2W convention."""

    def __init__(self, position: np.ndarray, R_c2w: np.ndarray):
        self.position = np.asarray(position, dtype=np.float64)
        self.R_c2w = np.asarray(R_c2w, dtype=np.float64)

    @classmethod
    def from_quat_pos(cls, qw, qx, qy, qz, x, y, z):
        R_c2w = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        return cls(position=np.array([x, y, z]), R_c2w=R_c2w)

    @classmethod
    def from_w2c(cls, w2c: np.ndarray):
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        R_c2w = R_w2c.T
        position = -R_c2w @ t_w2c
        return cls(position=position, R_c2w=R_c2w)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.R_c2w

    def to_w2c(self) -> np.ndarray:
        R_w2c = self.R_c2w.T
        t_w2c = -R_w2c @ self.position
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c

    def to_c2w(self) -> np.ndarray:
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = self.R_c2w
        c2w[:3, 3] = self.position
        return c2w

    def to_quat_pos(self) -> Tuple[float, float, float, float, float, float, float]:
        q = Rotation.from_matrix(self.R_c2w).as_quat()  # [qx, qy, qz, qw]
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]),
                float(self.position[0]), float(self.position[1]), float(self.position[2]))


def gt_rotation_corrected(
    w2c_gt: np.ndarray,
    w2c_mc: np.ndarray,
    blend: float = 1.0,
) -> np.ndarray:
    """Published GT translation + slerp-blended rotation toward an MC estimate."""
    gt = CameraPose.from_w2c(w2c_gt)
    mc = CameraPose.from_w2c(w2c_mc)
    r_rel = Rotation.from_matrix(gt.R_c2w).inv() * Rotation.from_matrix(mc.R_c2w)
    r_corr = Rotation.from_matrix(gt.R_c2w) * Rotation.from_rotvec(r_rel.as_rotvec() * blend)
    return CameraPose(gt.position, r_corr.as_matrix()).to_w2c()


# ── Pose file / intrinsics I/O ────────────────────────────────────────────────

def load_poses_csv(csv_path: str) -> dict:
    poses = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row['frame_id'])
            poses[fid] = CameraPose.from_quat_pos(
                qw=float(row['qw']), qx=float(row['qx']),
                qy=float(row['qy']), qz=float(row['qz']),
                x=float(row['x']), y=float(row['y']), z=float(row['z']),
            )
    return poses


def load_intrinsics(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        d = json.load(f)
    return np.array([
        [d['fx'], 0, d['cx']],
        [0, d['fy'], d['cy']],
        [0, 0, 1],
    ], dtype=np.float64)


def parse_pose_list(path):
    """Parse a COLMAP-format pose list → (names, c2w_matrices)."""
    all_pose_c2w, all_name = [], []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            if not data:
                continue
            q, t = np.split(np.array(data[1:], float), [4])
            name = data[0]
            R = np.asmatrix(qvec2rotmat(q)).transpose()  # R_cw (c2w)
            Pose_c2w = np.identity(4)
            Pose_c2w[0:3, 0:3] = R
            Pose_c2w[0:3, 3] = -R.dot(t)
            all_name.append(name)
            all_pose_c2w.append(Pose_c2w)
    return all_name, np.array(all_pose_c2w)


def read_intrinsic(intrinsc_path):
    """Read intrinsics file → (K (3,3), W, H)."""
    all_intrinsics = []
    w, h = 0, 0
    with open(intrinsc_path, 'r') as file:
        for line in file:
            data_line = line.strip("\n").split(' ')
            if not data_line or len(data_line) < 6:
                continue
            w, h, fx, fy, cx, cy = list(map(float, data_line[2:]))[:]
            all_intrinsics.append((fx, fy, cx, cy))
    if not all_intrinsics:
        raise ValueError("No intrinsic data found.")
    K = np.array([
        [all_intrinsics[0][0], 0, all_intrinsics[0][2]],
        [0, all_intrinsics[0][1], all_intrinsics[0][3]],
        [0, 0, 1]
    ])
    return K, int(w), int(h)


def scale_intrinsics(K: np.ndarray, old_size, new_size):
    H0, W0 = old_size
    H, W = new_size
    sx, sy = W / W0, H / H0
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx; K2[1, 1] *= sy
    K2[0, 2] *= sx; K2[1, 2] *= sy
    return K2


def scale_intrinsics_torch(K: torch.Tensor, old_size: Tuple[int, int],
                            new_size: Tuple[int, int]) -> torch.Tensor:
    H0, W0 = old_size
    H, W = new_size
    sx = float(W) / float(W0)
    sy = float(H) / float(H0)
    K2 = K.clone().float()
    K2[0, 0] *= sx; K2[1, 1] *= sy
    K2[0, 2] *= sx; K2[1, 2] *= sy
    return K2


# ── Euler / quaternion / rotation helpers ─────────────────────────────────────

def qvec2rotmat(qvec):
    """Quaternion (wxyz) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*qvec[2]**2 - 2*qvec[3]**2,
         2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3],
         2*qvec[3]*qvec[1] + 2*qvec[0]*qvec[2]],
        [2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3],
         1 - 2*qvec[1]**2 - 2*qvec[3]**2,
         2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1]],
        [2*qvec[3]*qvec[1] - 2*qvec[0]*qvec[2],
         2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1],
         1 - 2*qvec[1]**2 - 2*qvec[2]**2]])


def euler_to_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) → quaternion [w, x, y, z]."""
    y, p, r = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    cy, sy = math.cos(y/2), math.sin(y/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cr, sr = math.cos(r/2), math.sin(r/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    yq = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, yq, z], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q) + 1e-12)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))


def rotmat_to_ypr_zyx(Rcw: np.ndarray) -> Tuple[float, float, float]:
    euler = Rotation.from_matrix(Rcw).as_euler('xyz', degrees=True)
    return euler[0], euler[1], euler[2]


def euler_xyz_extrinsic_to_matrix_torch(pry_deg: torch.Tensor) -> torch.Tensor:
    """Pure-torch ZXY extrinsic Euler → (N,3,3) rotation matrices."""
    rad = torch.deg2rad(pry_deg)
    cx = torch.cos(rad[:, 0]); sx = torch.sin(rad[:, 0])
    cy = torch.cos(rad[:, 1]); sy = torch.sin(rad[:, 1])
    cz = torch.cos(rad[:, 2]); sz = torch.sin(rad[:, 2])
    N = pry_deg.shape[0]
    R_out = torch.empty(N, 3, 3, dtype=pry_deg.dtype, device=pry_deg.device)
    R_out[:, 0, 0] = cy*cz;  R_out[:, 0, 1] = sx*sy*cz - cx*sz;  R_out[:, 0, 2] = cx*sy*cz + sx*sz
    R_out[:, 1, 0] = cy*sz;  R_out[:, 1, 1] = sx*sy*sz + cx*cz;  R_out[:, 1, 2] = cx*sy*sz - sx*cz
    R_out[:, 2, 0] = -sy;    R_out[:, 2, 1] = sx*cy;             R_out[:, 2, 2] = cx*cy
    return R_out


def wrap_angle_deg(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


def wrap_angle_deg_torch(angles: torch.Tensor) -> torch.Tensor:
    return (angles + 180) % 360 - 180


def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    ang = np.radians(angles_deg)
    x = np.cos(ang) * weights
    y = np.sin(ang) * weights
    return wrap_angle_deg(math.degrees(math.atan2(y.sum(), x.sum())))


def circular_mean_deg_torch(angles_deg: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    angles_rad = torch.deg2rad(angles_deg)
    s = (torch.sin(angles_rad) * weights).sum(dim=1)
    c = (torch.cos(angles_rad) * weights).sum(dim=1)
    return torch.rad2deg(torch.atan2(s, c))


# ── Pose dict ↔ tensor ↔ 4×4 matrix ──────────────────────────────────────────

def pose_dict_to_tensor(pose_dict: Dict, device: torch.device) -> torch.Tensor:
    return torch.tensor([[
        pose_dict['x'], pose_dict['y'], pose_dict['z'],
        pose_dict['pitch_deg'], pose_dict['roll_deg'], pose_dict['yaw_deg']
    ]], device=device, dtype=torch.float32)


def pose_tensor_to_dict(pose_tensor: torch.Tensor) -> Dict:
    p = pose_tensor.cpu().numpy()
    return dict(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                pitch_deg=float(p[3]), roll_deg=float(p[4]), yaw_deg=float(p[5]))


def pose_dict_from_matrix(matrix_4x4: np.ndarray) -> Dict:
    Rcw = matrix_4x4[:3, :3]
    tcw = matrix_4x4[:3, 3]
    pitch_deg, roll_deg, yaw_deg = rotmat_to_ypr_zyx(Rcw)
    return dict(x=float(tcw[0]), y=float(tcw[1]), z=float(tcw[2]),
                pitch_deg=float(pitch_deg), roll_deg=float(roll_deg),
                yaw_deg=float(wrap_angle_deg(yaw_deg)))


def calculate_pose_errors(pose_est: Dict, pose_gt: Dict) -> Tuple[float, float, float, float, float]:
    error_x = np.abs(pose_est['x'] - pose_gt['x'])
    error_y = np.abs(pose_est['y'] - pose_gt['y'])
    error_z = np.abs(pose_est['z'] - pose_gt['z'])
    err_t = np.linalg.norm([pose_est['x']-pose_gt['x'],
                             pose_est['y']-pose_gt['y'],
                             pose_est['z']-pose_gt['z']])
    err_yaw = np.abs(wrap_angle_deg(pose_est['yaw_deg'] - pose_gt['yaw_deg']))
    return float(error_x), float(error_y), float(error_z), float(err_t), float(err_yaw)


# ── Pose parameter helpers (used by pose_solver) ───────────────────────────────

def pose_to_w2c(x: float, y: float, z: float,
                yaw: float, pitch: float, roll: float) -> np.ndarray:
    R_c2w = Rotation.from_euler('ZXY', [yaw, pitch, roll]).as_matrix()
    position = np.array([x, y, z], dtype=np.float64)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ position
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    return w2c


def decompose_pose(pose: 'CameraPose') -> Tuple[float, float, float, float, float, float]:
    r = Rotation.from_matrix(pose.R_c2w)
    yaw, pitch, roll = r.as_euler('ZXY')
    x, y, z = pose.position
    return float(x), float(y), float(z), float(yaw), float(pitch), float(roll)


def euler_to_camera_pose(x: float, y: float, z: float,
                         yaw: float, pitch: float, roll: float) -> 'CameraPose':
    R_c2w = Rotation.from_euler('ZXY', [yaw, pitch, roll]).as_matrix()
    return CameraPose(position=np.array([x, y, z]), R_c2w=R_c2w)


def camera_pose_to_params(pose: 'CameraPose') -> np.ndarray:
    xyz = pose.position.copy()
    euler = Rotation.from_matrix(pose.R_c2w).as_euler('xyz', degrees=True)
    return np.concatenate([xyz, euler])


def params_to_camera_pose(params: np.ndarray) -> 'CameraPose':
    xyz = params[:3]
    pry = params[3:6]
    R_c2w = Rotation.from_euler('xyz', pry, degrees=True).as_matrix()
    return CameraPose(position=xyz.copy(), R_c2w=R_c2w)


def perturb_params(params: np.ndarray, delta_xyz: np.ndarray,
                   delta_yaw_deg: float = 0.0) -> np.ndarray:
    p = params.copy()
    p[:3] += delta_xyz
    p[5] += delta_yaw_deg
    return p


def pose_error(est: CameraPose, gt: CameraPose) -> Tuple[float, float]:
    """Translation error (m) and geodesic rotation error (deg)."""
    t_err_m = float(np.linalg.norm(est.position - gt.position))
    R_err = est.R_c2w.T @ gt.R_c2w
    cos_ang = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return t_err_m, float(np.degrees(np.arccos(cos_ang)))

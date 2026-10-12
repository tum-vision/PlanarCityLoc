"""
inference.py  —  MoGe-2 finetuned inference + visualization
------------------------------------------------------------
Runs the finetuned MoGe-2 model on one or more query images from the
UAVD4L dataset (or any folder of images) and saves a side-by-side panel:

    [ Input image | Predicted normals | Building mask | Edge map ]

Optionally overlays GT raycasted normals if --scene and --gt-poses are given.

Usage
-----
    # Single image from outTraj:
    python inference.py --ckpt checkpoints/step36975-val_ang10.45.ckpt \\
                        --img path/to/image.JPG --out results/

    # All val frames with GT overlay:
    python inference.py --ckpt checkpoints/step36975-val_ang10.45.ckpt \\
                        --scene outTraj --n 20 --out results/ --with-gt

    # Grid of N frames side by side (saves one big mosaic):
    python inference.py --ckpt checkpoints/step36975-val_ang10.45.ckpt \\
                        --scene outTraj --n 9 --mosaic --out results/

Dependencies: torch, moge (pip), trimesh, open3d, Pillow, matplotlib, numpy
"""
import argparse
import copy
import math
import os
import sys
from pathlib import Path

import lightning as L
import matplotlib
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from moge.model.v2 import MoGeModel
from moge.utils.geometry_torch import normalized_view_plane_uv
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path resolution ────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
DATA_BASE = (HERE / '..' / 'data' / 'UAVD4L-LoD').resolve()
LOD3_PLY  = (HERE / '..' / 'data' / 'UAVD4L-LoD3.ply').resolve()


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry utils (inlined)
# ══════════════════════════════════════════════════════════════════════════════
def qvec2rotmat(qvec):
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


def _stem(n): return os.path.splitext(os.path.basename(n))[0]


def parse_pose_file(path):
    poses = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8: continue
            q = np.array(list(map(float, p[1:5])))
            t = np.array(list(map(float, p[5:8])))
            R_w2c = qvec2rotmat(q); R_c2w = R_w2c.T
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_c2w; T[:3, 3] = -R_c2w @ t
            poses[_stem(p[0])] = T
    return poses


def parse_intrinsic_file(path):
    intrs = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8: continue
            intrs[_stem(p[0])] = np.array(list(map(float, p[4:8])))
    return intrs


def build_raycast_scene(verts, faces):
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts.astype(np.float32))
    tm.triangle.indices = o3d.core.Tensor(faces.astype(np.uint32))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(tm)
    return sc


def raycast_normals(scene, T_c2w, intrinsic, H, W):
    fx, fy, cx, cy = float(intrinsic[0]), float(intrinsic[1]), \
                     float(intrinsic[2]), float(intrinsic[3])
    u, v   = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dirs_c = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u)], -1).reshape(-1, 3)
    R      = T_c2w[:3, :3].astype(np.float32)
    dirs_w = (R @ dirs_c.T).T
    dirs_w /= np.linalg.norm(dirs_w, axis=1, keepdims=True) + 1e-8
    orig   = np.tile(T_c2w[:3, 3].astype(np.float32), (H*W, 1))
    result = scene.cast_rays(o3d.core.Tensor(np.concatenate([orig, dirs_w], 1)))
    t_hit  = result['t_hit'].numpy().reshape(H, W)
    nw     = result['primitive_normals'].numpy().reshape(H, W, 3)
    hit    = (t_hit < 1e6) & (t_hit > 0)
    nc     = np.zeros((H, W, 3), dtype=np.float32)
    nc[hit] = (R.T @ nw[hit].T).T
    norms  = np.linalg.norm(nc, axis=2, keepdims=True)
    return np.where(norms > 0.1, nc / (norms + 1e-8), 0.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════
def load_base_model(device: str = 'cuda'):
    """Load original pretrained MoGe-2 (no fine-tuning). Used as baseline."""
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
    model.add_module('edge_head', copy.deepcopy(model.mask_head))
    return model.to(device).eval()


def load_model_from_heads(heads_dir: str, device: str = 'cuda'):
    """Load finetuned MoGe-2 from a directory of extracted head .pt files.

    Expects: ``heads_dir/normal_head.pt``, ``mask_head.pt``, ``edge_head.pt``.
    This avoids loading the full Lightning checkpoint (~1.4 GB) into RAM —
    only the small head weights (~18 MB each) are loaded on top of the base
    pretrained ViT backbone.
    """
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
    model.add_module('edge_head', copy.deepcopy(model.mask_head))

    for head_name, attr in [('normal_head', 'normal_head'),
                             ('mask_head',   'mask_head'),
                             ('edge_head',   'edge_head')]:
        pt_path = os.path.join(heads_dir, f'{head_name}.pt')
        if os.path.isfile(pt_path):
            head_sd = torch.load(pt_path, map_location='cpu', weights_only=True)
            getattr(model, attr).load_state_dict(head_sd, strict=True)
        else:
            print(f"  [warn] {pt_path} not found — keeping pretrained weights")

    return model.to(device).eval()


def load_model(ckpt_path: str, device: str = 'cuda'):
    """Load finetuned MoGeModule from a Lightning checkpoint."""

    # Reconstruct module skeleton (same as train.py)
    class _MoGeFwd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moge = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
            self.moge.add_module('edge_head', copy.deepcopy(self.moge.mask_head))

    model = _MoGeFwd()

    # Load Lightning checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt.get('state_dict', ckpt)
    # Strip Lightning 'moge.' prefix added by MoGeModule.save_hyperparameters
    sd_clean = {}
    for k, v in sd.items():
        new_k = k[len('moge.'):] if k.startswith('moge.') else k
        sd_clean[new_k] = v
    missing, unexpected = model.moge.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")

    model = model.to(device).eval()
    return model.moge


def load_model_base_normals(ckpt_path: str, device: str = 'cuda'):
    """Load fine-tuned mask_head from ``ckpt_path`` but restore the base
    moge-2 ``normal_head`` weights.

    Since the backbone (encoder + neck) was frozen during fine-tuning, the
    encoder weights are identical between base and fine-tuned models.
    Only the normal_head is swapped back to avoid overfitting to outTraj normals.
    """

    # 1. Load the base moge-2 normal_head weights
    base = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
    base_normal_sd = {k: v.clone() for k, v in base.normal_head.state_dict().items()}
    del base

    # 2. Load fine-tuned model (mask_head, edge_head fine-tuned; encoder frozen)
    class _MoGeFwd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moge = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
            self.moge.add_module('edge_head', copy.deepcopy(self.moge.mask_head))

    model = _MoGeFwd()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt.get('state_dict', ckpt)
    sd_clean = {}
    for k, v in sd.items():
        new_k = k[len('moge.'):] if k.startswith('moge.') else k
        sd_clean[new_k] = v
    missing, unexpected = model.moge.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")

    # 3. Overwrite fine-tuned normal_head with base weights
    model.moge.normal_head.load_state_dict(base_normal_sd, strict=True)
    print("  [base_normal_head] Replaced fine-tuned normal_head with base moge-2 weights.")

    return model.moge.to(device).eval()


def load_model_base_normals(ckpt_path: str, device: str = 'cuda'):
    """Load fine-tuned mask_head from ``ckpt_path`` but restore the base
    moge-2 ``normal_head`` weights.

    Since the backbone (encoder + neck) was frozen during fine-tuning, the
    encoder weights are identical between base and fine-tuned models.
    Only the normal_head is swapped back to avoid overfitting to outTraj normals.
    """

    # 1. Load the base moge-2 normal_head weights
    base = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
    base_normal_sd = {k: v.clone() for k, v in base.normal_head.state_dict().items()}
    del base

    # 2. Load fine-tuned model (mask_head, edge_head fine-tuned; encoder frozen)
    class _MoGeFwd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moge = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
            self.moge.add_module('edge_head', copy.deepcopy(self.moge.mask_head))

    model = _MoGeFwd()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt.get('state_dict', ckpt)
    sd_clean = {}
    for k, v in sd.items():
        new_k = k[len('moge.'):] if k.startswith('moge.') else k
        sd_clean[new_k] = v
    missing, unexpected = model.moge.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")

    # 3. Overwrite fine-tuned normal_head with base weights
    model.moge.normal_head.load_state_dict(base_normal_sd, strict=True)
    print("  [base_normal_head] Replaced fine-tuned normal_head with base moge-2 weights.")

    return model.moge.to(device).eval()


@torch.no_grad()
def run_inference(moge, img_pil: Image.Image, enc_res: int = 640, device: str = 'cuda'):
    """
    Run finetuned model on a PIL image.

    Returns
    -------
    pred_normal : torch.Tensor  (H_out, W_out, 3)  camera-frame unit normals
    pred_mask   : torch.Tensor  (H_out, W_out)      building probability [0,1]
    pred_edge   : torch.Tensor  (H_out, W_out)      edge probability [0,1] or None
    pred_depth  : torch.Tensor  (H_out, W_out)      camera-space Z depth or None
    native_hw   : (int, int)  head output resolution
    """

    W0, H0 = img_pil.size
    ar     = W0 / H0
    sc     = enc_res / max(W0, H0)
    We, He = int(round(W0 * sc)), int(round(H0 * sc))
    img_t  = torch.from_numpy(
        np.array(img_pil.resize((We, He), Image.BILINEAR), dtype=np.uint8)
    ).permute(2, 0, 1).float() / 255.0
    imgs   = img_t.unsqueeze(0).to(device)

    _, max_t = moge.num_tokens_range
    bh = round((max_t / ar) ** 0.5)
    bw = round((max_t * ar) ** 0.5)
    H_nat, W_nat = bh * 16, bw * 16

    # Encoder + neck (frozen)
    dtype = moge.dtype
    x     = imgs.to(dtype)
    enc, _ = moge.encoder(x, bh, bw, return_class_token=True)
    feats  = [enc] + [None] * 4
    for lvl in range(5):
        uv = normalized_view_plane_uv(bw * 2**lvl, bh * 2**lvl, ar, dtype, device)
        uv = uv.permute(2, 0, 1).unsqueeze(0)
        feats[lvl] = torch.cat([feats[lvl], uv], 1) if feats[lvl] is not None else uv
    neck = moge.neck(feats)

    # Heads
    H, W  = bh * 16, bw * 16   # feature_sizes[-1] == bh*16, bw*16
    n_raw = moge.normal_head(neck)[-1]
    n_up  = F.interpolate(n_raw, (H, W), mode='bilinear', align_corners=False, antialias=False)
    pred_n = F.normalize(n_up.permute(0, 2, 3, 1), dim=-1)[0]

    m_raw  = moge.mask_head(neck)[-1]
    m_up   = F.interpolate(m_raw, (H, W), mode='bilinear', align_corners=False, antialias=False)
    pred_m = m_up[0, 0].sigmoid()

    pred_e = None
    if hasattr(moge, 'edge_head') and moge.edge_head is not None:
        e_raw  = moge.edge_head(neck)[-1]
        e_up   = F.interpolate(e_raw, (H, W), mode='bilinear', align_corners=False, antialias=False)
        pred_e = e_up[0, 0].sigmoid()

    pred_d = None
    if hasattr(moge, 'points_head') and moge.points_head is not None:
        p_raw = moge.points_head(neck)[-1]
        p_up  = F.interpolate(p_raw, (H, W), mode='bilinear', align_corners=False, antialias=False)
        p_map = p_up.permute(0, 2, 3, 1)[0].float()
        rm = getattr(moge, 'remap_output', 'exp')
        if rm == 'exp':
            xy, z = p_map[..., :2], p_map[..., 2:3]
            z = torch.exp(z)
            p_map = torch.cat([xy * z, z], dim=-1)
        elif rm == 'sinh_exp':
            xy, z = p_map[..., :2], p_map[..., 2:3]
            p_map = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        elif rm == 'sinh':
            p_map = torch.sinh(p_map)
        pred_d = p_map[..., 2]

    return pred_n, pred_m, pred_e, pred_d, (H_nat, W_nat)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization helpers
# ══════════════════════════════════════════════════════════════════════════════
def normal_to_rgb(n: np.ndarray) -> np.ndarray:
    """(H,W,3) float normals → (H,W,3) uint8 RGB."""
    return np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def save_panel(img_pil, pred_n, pred_m, pred_e,
               gt_n=None, gt_hit=None,
               save_path=None, title=''):
    """
    Save a side-by-side comparison panel.
    Columns: Input | [GT normal] | Pred normal | Pred mask | [Pred edge]
    """
    H, W = pred_n.shape[:2]
    img_np = np.array(img_pil.resize((W, H), Image.BILINEAR))

    cols = [img_np]
    labels = ['Input']

    if gt_n is not None:
        gt_rgb  = normal_to_rgb(gt_n)
        cols.append(gt_rgb)
        labels.append('GT normal')
        if gt_hit is not None:
            gt_mask_rgb = (np.stack([gt_hit]*3, -1) * 255).astype(np.uint8)
            cols.append(gt_mask_rgb)
            labels.append('GT mask')

    cols.append(normal_to_rgb(pred_n))
    labels.append('Pred normal')

    pred_m_rgb = (np.stack([pred_m]*3, -1) * 255).astype(np.uint8)
    cols.append(pred_m_rgb)
    labels.append('Pred mask')

    if pred_e is not None:
        pred_e_rgb = (np.stack([pred_e]*3, -1) * 255).astype(np.uint8)
        cols.append(pred_e_rgb)
        labels.append('Pred edge')

    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for ax, col, lbl in zip(axes, cols, labels):
        ax.imshow(col)
        ax.set_title(lbl, fontsize=9)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.close(fig)


def save_mosaic(panels_rgb: list, n_cols: int = 3, save_path: str = 'mosaic.png',
                labels: list = None):
    """Save a N×M mosaic of HxW uint8 arrays."""
    n     = len(panels_rgb)
    n_rows = math.ceil(n / n_cols)
    H, W  = panels_rgb[0].shape[:2]
    canvas = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.uint8)
    for i, p in enumerate(panels_rgb):
        r, c = divmod(i, n_cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = p[:, :, :3]
    fig, ax = plt.subplots(1, 1, figsize=(n_cols * 4, n_rows * 4))
    ax.imshow(canvas); ax.axis('off')
    if labels:
        # annotate top-left of each cell
        for i, lbl in enumerate(labels):
            r, c = divmod(i, n_cols)
            ax.text(c*W + 4, r*H + 14, lbl, color='white', fontsize=7,
                    backgroundcolor=(0, 0, 0, 0.5))
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Mosaic → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='MoGe-2 finetuned inference')
    parser.add_argument('--ckpt',    required=True,
                        help='Path to Lightning .ckpt (e.g. checkpoints/step36975-val_ang10.45.ckpt)')
    parser.add_argument('--img',     default=None,
                        help='Path to a single input image')
    parser.add_argument('--scene',   default='outTraj',
                        help='Scene name under data/UAVD4L-LoD/ (used when --img not given)')
    parser.add_argument('--n',       type=int, default=6,
                        help='Number of frames to visualize from --scene')
    parser.add_argument('--enc-res', type=int, default=640)
    parser.add_argument('--out',     default='results',
                        help='Output directory for saved panels')
    parser.add_argument('--with-gt', action='store_true',
                        help='Overlay GT raycasted normals (requires LoD3 mesh + poses)')
    parser.add_argument('--mosaic',  action='store_true',
                        help='Save a single mosaic image instead of one panel per frame')
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    moge = load_model(args.ckpt, device=args.device)

    # ── GT data (optional) ─────────────────────────────────────────────────────
    scene_dir = DATA_BASE / args.scene
    gt_scene = gt_poses = gt_intrs = None
    if args.with_gt:
        print("Loading LoD3 mesh for GT raycasting ...")
        mesh   = trimesh.load(str(LOD3_PLY), force='mesh', process=False)
        verts  = np.array(mesh.vertices, dtype=np.float32)
        faces  = np.array(mesh.faces,  dtype=np.uint32)
        gt_scene = build_raycast_scene(verts, faces)
        gt_poses = parse_pose_file(str(scene_dir / 'GT_pose.txt'))
        gt_intrs = parse_intrinsic_file(str(scene_dir / 'intrinsic.txt'))
        print(f"  {verts.shape[0]:,} verts, {faces.shape[0]:,} faces")

    # ── Build image list ───────────────────────────────────────────────────────
    if args.img:
        img_paths = [Path(args.img)]
    else:
        img_dir   = scene_dir / 'Query_image'
        all_stems = sorted(set(gt_poses or {}).keys()) if gt_poses else []
        if not all_stems:
            all_stems = sorted([p.stem for p in img_dir.glob('*.JPG')])
        # Take every-N-th to get representative spread
        step     = max(1, len(all_stems) // args.n)
        stems    = all_stems[::step][:args.n]
        img_paths = [img_dir / f'{s}.JPG' for s in stems]

    print(f"Running inference on {len(img_paths)} images ...")
    mosaic_panels = []

    for img_path in img_paths:
        img_pil = Image.open(img_path).convert('RGB')
        W0, H0  = img_pil.size
        stem    = img_path.stem

        pred_n, pred_m, pred_e, pred_d, (H_nat, W_nat) = run_inference(
            moge, img_pil, enc_res=args.enc_res, device=args.device)
        
        pred_n = pred_n.cpu().float().numpy()
        pred_m = pred_m.cpu().float().numpy()
        if pred_e is not None:
            pred_e = pred_e.cpu().float().numpy()
        if pred_d is not None:
            pred_d = pred_d.cpu().float().numpy()

        # ── GT overlay ─────────────────────────────────────────────────────────
        gt_n = gt_hit = None
        if gt_scene is not None and stem in (gt_poses or {}):
            intr    = gt_intrs[stem]
            intr_rc = np.array([intr[0]*W_nat/W0, intr[1]*H_nat/H0,
                                 intr[2]*W_nat/W0, intr[3]*H_nat/H0], dtype=np.float32)
            gt_n  = raycast_normals(gt_scene, gt_poses[stem], intr_rc, H_nat, W_nat)
            gt_hit = np.linalg.norm(gt_n, axis=2) > 0.1

        if args.mosaic:
            # Build a compact row: [input | pred_normal | pred_mask]
            H, W = pred_n.shape[:2]
            inp_r = np.array(img_pil.resize((W, H), Image.BILINEAR))
            row   = np.concatenate([inp_r, normal_to_rgb(pred_n),
                                    (np.stack([pred_m]*3,-1)*255).astype(np.uint8)], axis=1)
            if pred_e is not None:
                row = np.concatenate([row, (np.stack([pred_e]*3,-1)*255).astype(np.uint8)], axis=1)
            mosaic_panels.append(row)
        else:
            save_panel(
                img_pil, pred_n, pred_m, pred_e,
                gt_n=gt_n, gt_hit=gt_hit,
                save_path=str(out_dir / f'{stem}_panel.png'),
                title=stem,
            )

    if args.mosaic and mosaic_panels:
        mosaic_path = str(out_dir / 'mosaic.png')
        n_cols = min(3, len(mosaic_panels))
        save_mosaic(mosaic_panels, n_cols=n_cols, save_path=mosaic_path,
                    labels=[p.stem for p in img_paths])

    print(f"\nDone. Results in {out_dir}/")


if __name__ == '__main__':
    main()

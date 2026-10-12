#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.abspath('.'))
from moge2.inference import load_model_base_normals
from moge.utils.geometry_torch import normalized_view_plane_uv

@torch.no_grad()
def run_batched_inference(moge, imgs_pil, enc_res=640, device='cuda'):
    if not imgs_pil: return []
    
    W0, H0 = imgs_pil[0].size
    ar = W0 / H0
    sc = enc_res / max(W0, H0)
    We, He = int(round(W0 * sc)), int(round(H0 * sc))
    
    img_tensors = []
    for img_pil in imgs_pil:
        img_np = np.array(img_pil.resize((We, He), Image.BILINEAR), dtype=np.uint8)
        img_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0)
        
    imgs = torch.stack(img_tensors).to(device)
    B = imgs.shape[0]

    _, max_t = moge.num_tokens_range
    bh = round((max_t / ar) ** 0.5)
    bw = round((max_t * ar) ** 0.5)
    
    dtype = moge.dtype
    x = imgs.to(dtype)
    enc, _ = moge.encoder(x, bh, bw, return_class_token=True)
    feats = [enc] + [None] * 4
    for lvl in range(5):
        uv = normalized_view_plane_uv(bw * 2**lvl, bh * 2**lvl, ar, dtype, device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        feats[lvl] = torch.cat([feats[lvl], uv], 1) if feats[lvl] is not None else uv
    neck = moge.neck(feats)

    H, W = bh * 16, bw * 16
    m_raw = moge.mask_head(neck)[-1]
    m_up = F.interpolate(m_raw, (H, W), mode='bilinear', align_corners=False, antialias=False)
    pred_m = m_up.squeeze(1).sigmoid() # (B, H, W)
    
    return pred_m.cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate and save MoGe-2 masks")
    parser.add_argument('--ckpt', default='checkpoints/moge2/best.ckpt', help='Path to checkpoint')
    parser.add_argument('--enc_res', type=int, default=640, help='Encoder resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument(
        '--dataset-dir', action='append', type=Path, required=True,
        help='Training split directory containing images/; repeat for multiple splits',
    )
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.ckpt} on {device}...")
    model = load_model_base_normals(args.ckpt, device=device)
    
    for ds_path in args.dataset_dir:
        img_dir = ds_path / 'images'
        out_dir = ds_path / 'masks_pred' / 'moge2'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist. Skipping.")
            continue
            
        img_paths = sorted(glob.glob(str(img_dir / '*.*')))
        img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.png'))]
        
        # Filter out already processed paths
        todo_paths = []
        for p in img_paths:
            if not (out_dir / f"{Path(p).stem}.png").exists():
                todo_paths.append(p)
                
        print(f"\nProcessing {ds_path} -> {len(todo_paths)} images (skipped {len(img_paths) - len(todo_paths)})")
        
        pbar = tqdm(range(0, len(todo_paths), args.batch_size), desc=f"Out: file://{out_dir.absolute()}")
        for i in pbar:
            batch_paths = todo_paths[i:i+args.batch_size]
            imgs_pil = [Image.open(p).convert('RGB') for p in batch_paths]
            W0, H0 = imgs_pil[0].size
            
            pred_m_np = run_batched_inference(model, imgs_pil, enc_res=args.enc_res, device=device)
            
            for j, p in enumerate(batch_paths):
                stem = Path(p).stem
                out_path = out_dir / f"{stem}.png"
                mask_res = cv2.resize(pred_m_np[j], (W0, H0), interpolation=cv2.INTER_LINEAR)
                mask_img = (mask_res * 255).astype(np.uint8)
                cv2.imwrite(str(out_path), mask_img)

if __name__ == '__main__':
    main()

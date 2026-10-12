#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

import sys
# Make sure we can import from utils
sys.path.insert(0, os.path.abspath('.'))
from utils.sam3fast import SAM3BuildingSegmentorFast

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate and save SAM3 masks")
    parser.add_argument(
        '--pair', action='append', nargs=2, required=True,
        metavar=('INPUT_DIR', 'OUTPUT_DIR'),
        help='Query-image input and mask output directories; repeat as needed',
    )
    parser.add_argument('--checkpoint', default='checkpoints/sam3/sam3_building.pt')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading SAM3 model on {device}...")
    seg = SAM3BuildingSegmentorFast(checkpoint_path=args.checkpoint, device=device)
    
    for img_dir_str, out_dir_str in args.pair:
        img_dir = Path(img_dir_str)
        out_dir = Path(out_dir_str)

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
                
        print(f"\nProcessing {img_dir_str} -> {len(todo_paths)} images (skipped {len(img_paths) - len(todo_paths)})")
        
        pbar = tqdm(todo_paths, desc=f"Out: {out_dir}")
        for p in pbar:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue
                
            mask, _ = seg.predict_with_confidence(img_bgr)
            
            stem = Path(p).stem
            out_path = out_dir / f"{stem}.png"
            mask_img = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(out_path), mask_img)

if __name__ == '__main__':
    main()

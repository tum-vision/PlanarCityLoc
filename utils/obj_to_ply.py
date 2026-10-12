#!/usr/bin/env python3
"""Convert a LoD mesh (.obj or any trimesh format) to .ply without processing."""

from pathlib import Path

import argparse
import trimesh
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a mesh to PLY (no processing).")
    parser.add_argument("src", type=Path, help="Input mesh, e.g. MODEL/UAVD4L-LoD.obj")
    parser.add_argument("dst", type=Path, help="Output .ply path")
    args = parser.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.src, force="mesh", process=False)
    mesh.export(args.dst)
    logger.info("Wrote {} ({} → {} bytes)", args.dst, args.src.stat().st_size, args.dst.stat().st_size)


if __name__ == "__main__":
    main()

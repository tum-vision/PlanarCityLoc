#!/usr/bin/env python3
"""Smoke-test PlanarCityLoc on the first N UAVD4L in-Traj frames."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--out", default="outputs/smoke_uavd4l_intraj")
    args = parser.parse_args()

    os.chdir(ROOT)
    cmd = [
        sys.executable, "-u", "pipeline.py",
        "--dataset", "uavd4l",
        "--data_root", "data/UAVD4L-LoD/inTraj",
        "--n", str(args.n),
        "--batchsize", str(args.batchsize),
        "--no-crop",
        "--out", args.out,
    ]
    logger.info("Running {}", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

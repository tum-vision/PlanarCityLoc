"""Write auditable evaluation evidence without silently overwriting a run."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from pose_solver.selection import PROTOCOL, PROTOCOL_FILES


def current_protocol():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / PROTOCOL_FILES[PROTOCOL]).read_text())
    if spec['protocol'] != PROTOCOL:
        raise ValueError('Manifest protocol does not match the active selector')
    return spec


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean_json(record), allow_nan=False) + "\n")


def write_manifest(out, args, cfg, dataset, mesh_path, checkpoints=()):
    root = Path(__file__).resolve().parents[1]
    code = [root / "pipeline.py", root / "pipeline_ablation.py"]
    for folder in ("utils", "pose_solver", "datasets", "moge2", "configs"):
        code.extend(p for p in (root / folder).rglob("*")
                    if p.suffix in {".py", ".yaml", ".json"})
    data_root = Path(args.data_root)
    inputs = [p for p in data_root.rglob("*.txt") if "pose" in p.name.lower() or "intrinsic" in p.name.lower()]
    inputs.extend((data_root / "split").glob("*.json"))
    inputs.extend([Path(mesh_path), *(Path(p) for p in checkpoints)])
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]}
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unavailable"
    evidence = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "host": platform.node(),
        "gpu": gpu, "packages": packages, "arguments": vars(args),
        "resolved_config": asdict(cfg), "frames": dataset.frames,
        "frame_list_sha256": hashlib.sha256(("\n".join(dataset.frames) + "\n").encode()).hexdigest(),
        "code_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(set(code))},
        "input_sha256": {str(p): sha256(p) for p in sorted(set(inputs)) if p.is_file()},
        "protocol": current_protocol(),
        "timing": "wall_inference_s includes per-image I/O, preprocessing, inference, all solver stages, transfers, synchronization and logging; excludes model/map loading, warmup, GT evaluation and visualization",
    }
    with Path(out, "run_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(clean_json(evidence), stream, indent=2, allow_nan=False)
        stream.write("\n")

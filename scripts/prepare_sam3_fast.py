#!/usr/bin/env python3
"""Build the paper's compact, building-prompt SAM 3 checkpoint.

Access to the gated ``facebook/sam3`` repository on Hugging Face is required
unless ``--full-checkpoint`` points to an already downloaded ``sam3.pt``.
"""

import argparse
import base64
import hashlib
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent.parent
THIRDPARTY = ROOT / "thirdparty"
EXPECTED_SHA256 = "61c5d101e3572a10e2a688e4082218ab39d50182ed3fc96c2ac00e8250158088"
EXPECTED_CONTENT_SHA256 = "eaf8b0f6d956fcda7f47acf0c4f0cbc37a84fb5387c1aa0c50fd1a7e06c44148"
EXPECTED_MODEL_KEYS = 861


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(checkpoint: dict) -> str:
    """Hash tensor names, shapes, dtypes, and bytes independent of torch.save."""
    digest = hashlib.sha256()
    digest.update(checkpoint["prompt"].encode() + b"\0")
    for section in ("model", "cached_text"):
        for key in sorted(checkpoint[section]):
            value = checkpoint[section][key].detach().cpu().contiguous()
            digest.update(
                section.encode() + b"\0" + key.encode() + b"\0"
                + str(value.dtype).encode() + b"\0"
                + str(tuple(value.shape)).encode() + b"\0"
            )
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        help="optional local sam3.pt; otherwise download facebook/sam3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints/sam3/sam3_building.pt",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(THIRDPARTY))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    full_checkpoint = args.full_checkpoint or Path(hf_hub_download(
        repo_id="facebook/sam3", filename="sam3.pt",
    ))
    model = build_sam3_image_model(
        checkpoint_path=str(full_checkpoint),
        load_from_HF=False,
        device=args.device,
        enable_inst_interactivity=False,
    ).eval()
    processor = Sam3Processor(model)

    # The language embedding is image-independent; this is the exact paper
    # prompt and dummy input used to prepare the released artifact.
    dummy = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
    with torch.no_grad(), torch.amp.autocast("cuda"):
        state = processor.set_image(dummy)
        processor.set_text_prompt(state=state, prompt="building")

    cached = {
        "language_features": state["backbone_out"]["language_features"].cpu().half(),
        "language_mask": state["backbone_out"]["language_mask"].cpu(),
        "language_embeds": state["backbone_out"]["language_embeds"].cpu().half(),
    }

    # This small bundled tensor removes GPU-architecture variation observed in
    # the projected language features. The underlying text embedding and mask
    # are deterministic and are still derived from the official checkpoint.
    encoded = "".join(
        (ROOT / "assets/sam3_building_language_features.b85")
        .read_text(encoding="ascii").splitlines()
    )
    feature_bytes = zlib.decompress(base64.b85decode(encoded))
    cached["language_features"] = torch.frombuffer(
        bytearray(feature_bytes), dtype=torch.float16,
    ).clone().reshape(32, 1, 256)

    # Preserve the exact official detector tensors, including the SAM2
    # convolution adapters that are not registered by the full-model builder.
    # Only the language backbone is removed because its outputs are cached.
    raw = torch.load(full_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    raw_state = raw.get("model", raw)
    compact_state = {}
    for key, value in raw_state.items():
        if not key.startswith("detector."):
            continue
        key = key[len("detector."):]
        if key.startswith("backbone.language_backbone."):
            continue
        compact_state[key] = value.half().cpu() if value.is_floating_point() else value.cpu()
    if len(compact_state) != EXPECTED_MODEL_KEYS:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_KEYS} compact model tensors, got "
            f"{len(compact_state)}; verify that this is the paper SAM 3 checkpoint."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": compact_state, "cached_text": cached, "prompt": "building"}
    functional_hash = content_sha256(checkpoint)
    if functional_hash != EXPECTED_CONTENT_SHA256:
        raise RuntimeError(
            "Generated checkpoint tensor-content hash does not match the paper: "
            f"{functional_hash}"
        )
    torch.save(checkpoint, args.output)
    actual = sha256(args.output)
    print(f"Wrote {args.output}")
    print(f"SHA-256: {actual}")
    print(f"Tensor-content SHA-256: {functional_hash} (paper exact)")
    if actual != EXPECTED_SHA256:
        print(
            "Note: serialized-file hashes can differ across PyTorch versions. "
            "The exact paper artifact hash is " + EXPECTED_SHA256
        )


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_THIRDPARTY = Path(__file__).resolve().parent.parent / "thirdparty"
if str(_THIRDPARTY) not in sys.path:
    sys.path.insert(0, str(_THIRDPARTY))

class SAM3BuildingSegmentorFast:
    """Optimised SAM3 for building-only segmentation.
    
    Optimisations vs SAM3BuildingSegmentor:
      - Text encoder removed (353.7M params); cached embeddings injected directly
      - Tracker weights stripped (11.7M params)
      - FP16 weights (~50% checkpoint size)
      - Memory-mapped + direct-GPU checkpoint loading
      - AMP inference (float16 compute)
      - Optional torch.compile for vision backbone
    """

    def __init__(self, checkpoint_path: str = "checkpoints/sam3/sam3_building.pt",
                 device: str = 'cuda', min_score: float = 0.0,
                 use_compile: bool = False):
        self.device = torch.device(device)
        self.min_score = min_score

        # Memory-mapped load directly to target device (avoids CPU copy)
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=True, mmap=True)
        model_state = ckpt['model']
        self.cached_text = ckpt['cached_text']  # already on device via map_location

        # Build model without text encoder
        from sam3.model_builder import (
            _create_vision_backbone,
            _create_sam3_transformer, _create_dot_product_scoring,
            _create_segmentation_head, _create_geometry_encoder,
            _create_sam3_model,
        )
        from sam3.model.vl_combiner import SAM3VLBackbone

        vision_encoder = _create_vision_backbone(
            enable_inst_interactivity=False,
        )
        backbone = SAM3VLBackbone(visual=vision_encoder, text=None, scalp=1)
        transformer = _create_sam3_transformer()
        dot_prod_scoring = _create_dot_product_scoring()
        segmentation_head = _create_segmentation_head()
        geometry_encoder = _create_geometry_encoder()

        self.model = _create_sam3_model(
            backbone, transformer, geometry_encoder, segmentation_head,
            dot_prod_scoring, None, True,
        )

        # Load weights with assign=True (avoids double-memory from copy)
        fp32_state = {k: (v.float() if v.is_floating_point() else v)
                      for k, v in model_state.items()}
        self.model.load_state_dict(fp32_state, strict=False, assign=True)
        self.model.to(self.device)
        self.model.eval()

        if use_compile:
            self.model.backbone.vision_backbone = torch.compile(
                self.model.backbone.vision_backbone
            )

        # Reuse Sam3Processor for transforms + grounding logic
        from sam3.model.sam3_image_processor import Sam3Processor
        self._proc = Sam3Processor(self.model, confidence_threshold=self.min_score)

        print(f"SAM3BuildingSegmentorFast: loaded {checkpoint_path} "
              f"(text encoder removed, FP16 weights)")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        mask, _ = self.predict_with_confidence(image_bgr)
        return mask

    @torch.no_grad()
    def predict_with_confidence(self, image_bgr: np.ndarray) -> tuple:
        H, W = image_bgr.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        # Image encoding with AMP
        with torch.amp.autocast('cuda'):
            state = self._proc.set_image(img_pil)

        # Inject cached text features (skip text encoder entirely)
        state["backbone_out"]["language_features"] = self.cached_text["language_features"]
        state["backbone_out"]["language_mask"] = self.cached_text["language_mask"]
        state["backbone_out"]["language_embeds"] = self.cached_text["language_embeds"]

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Grounding with AMP
        with torch.amp.autocast('cuda'):
            state = self._proc._forward_grounding(state)

        masks, scores = state["masks"], state["scores"]

        combined = np.zeros((H, W), dtype=bool)
        confidence = np.zeros((H, W), dtype=np.float32)
        for m, s in zip(masks, scores):
            s_val = float(s)
            if s_val < self.min_score:
                continue
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            if m.ndim == 3:
                m = m.squeeze()
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST)
            m_bool = m.astype(bool)
            combined |= m_bool
            confidence = np.where(m_bool & (s_val > confidence),
                                  s_val, confidence)

        return combined, confidence

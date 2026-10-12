import sys
import os
from pathlib import Path
import yaml
import importlib
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from huggingface_hub import hf_hub_download

EOMT_REPO_PATH = Path(__file__).parent.parent / 'thirdparty' / 'eomt_repo'
if str(EOMT_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(EOMT_REPO_PATH))

class EoMTBuildingSegmentor:
    def __init__(self, config_name="eomt_large_640.yaml", device="cuda"):
        self.device = device
        config_path = EOMT_REPO_PATH / "configs" / "dinov2" / "coco" / "panoptic" / config_name
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
        )
        
        # Load encoder
        encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
        encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
        encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
        encoder = encoder_cls(img_size=640, **encoder_cfg.get("init_args", {}))

        # Load network
        network_cfg = config["model"]["init_args"]["network"]
        network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
        network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
        network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
        network = network_cls(
            masked_attn_enabled=False,
            num_classes=133,
            encoder=encoder,
            **network_kwargs,
        )

        # Load Lightning module
        lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
        lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
        model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
        if "stuff_classes" in config["data"].get("init_args", {}):
            model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

        self.model = (
            lit_cls(
                img_size=(640, 640),
                num_classes=133,
                network=network,
                **model_kwargs,
            )
            .eval()
            .to(device)
        )
        
        name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
        try:
            state_dict_path = hf_hub_download(
                repo_id=f"tue-mps/{name}",
                filename="pytorch_model.bin",
                revision="dcd130bed9b1ebda7041fd660fddb16f905b9c3b" if name == "coco_panoptic_eomt_large_640" else None,
            )
            self.checkpoint_path = state_dict_path
            state_dict = torch.load(
                state_dict_path, map_location=device, weights_only=True
            )
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            parameters = set(dict(self.model.named_parameters()))
            missing_parameters = sorted(parameters.intersection(incompatible.missing_keys))
            if missing_parameters:
                raise RuntimeError(f"Checkpoint is missing model parameters: {missing_parameters}")
            print(f"EoMT checkpoint buffers/extra keys: {incompatible}")
            print(f"EoMTBuildingSegmentor: loaded {name} from huggingface hub")
        except Exception as e:
            raise RuntimeError(f"Failed to load EoMT checkpoint; refusing uninitialized-model evaluation: {e}") from e

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        img_tensor = img_tensor.to(torch.uint8).unsqueeze(0).to(self.device)
        
        img_sizes = [img_tensor.shape[2:]]
        with torch.amp.autocast('cuda'):
            transformed_imgs = self.model.resize_and_pad_imgs_instance_panoptic(img_tensor)
            mask_logits_per_layer, class_logits_per_layer = self.model(transformed_imgs)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], self.model.img_size, mode="bilinear"
            )
            mask_logits = self.model.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds = self.model.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits_per_layer[-1],
                self.model.stuff_classes,
                self.model.mask_thresh,
                self.model.overlap_thresh,
            )[0]
        
        sem_pred = preds[..., 0].cpu().numpy()
        # In HuggingFace's facebook/mask2former-swin-large-coco-panoptic:
        # 129 -> 'building-other-merged'
        # 101 -> 'roof'
        # 91  -> 'house'
        building_mask = (sem_pred == 129) | (sem_pred == 101) | (sem_pred == 91)
        return building_mask

import os
import sys
import copy
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file

import folder_paths
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import box_ops
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from AILab_ImageMaskTools import pil2tensor, tensor2pil

# SAM model definitions (6 models)
SAM_MODELS = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "model_type": "vit_h",
        "filename": "sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "model_type": "vit_l",
        "filename": "sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "filename": "sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
        "model_type": "vit_h",
        "filename": "sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
        "model_type": "vit_l",
        "filename": "sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
        "model_type": "vit_b",
        "filename": "sam_hq_vit_b.pth"
    }
}

# GroundingDINO model definitions (2 models)
DINO_MODELS = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "config_filename": "GroundingDINO_SwinT_OGC.cfg.py",
        "model_filename": "groundingdino_swint_ogc.pth"
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
        "config_filename": "GroundingDINO_SwinB.cfg.py",
        "model_filename": "groundingdino_swinb_cogcoor.pth"
    }
}

# Define explicit model directories
SAM_MODELS_DIR = os.path.join(os.path.dirname(folder_paths.__file__), "..", "ComfyUI", "models", "sams")
DINO_MODELS_DIR = os.path.join(os.path.dirname(folder_paths.__file__), "..", "ComfyUI", "models", "grounding-dino")

def get_or_download_model_file(filename, url, dirname):
    # Set the directory based on model type
    if dirname.lower() == "sam":
        folder = os.path.abspath(SAM_MODELS_DIR)
    elif dirname.lower() == "grounding-dino":
        folder = os.path.abspath(DINO_MODELS_DIR)
    else:
        folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)
    local_path = os.path.join(folder, filename)
    # Check if file exists in the intended directory
    if os.path.exists(local_path):
        return local_path
    # If not found, check if folder_paths knows about it
    fp_path = folder_paths.get_full_path(dirname, filename)
    if fp_path and os.path.exists(fp_path):
        return fp_path
    # Download if not present
    print(f"Downloading {filename} from {url} ...")
    download_url_to_file(url, local_path)
    return local_path

def process_mask(mask_image: Image.Image, invert_output: bool = False, 
                mask_blur: int = 0, mask_offset: int = 0) -> Image.Image:
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filter_type(size))
    return mask_image

def apply_background_color(image: Image.Image, mask_image: Image.Image, 
                         background: str = "Alpha",
                         background_color: str = "#222222") -> Image.Image:
    rgba_image = image.copy().convert('RGBA')
    rgba_image.putalpha(mask_image.convert('L'))
    if background == "Color":
        def hex_to_rgba(hex_color):
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return (r, g, b, 255)
        rgba = hex_to_rgba(background_color)
        bg_image = Image.new('RGBA', image.size, rgba)
        composite_image = Image.alpha_composite(bg_image, rgba_image)
        return composite_image.convert('RGB')
    return rgba_image

def get_groundingdino_model(device):
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
    return processor, model

def get_boxes(processor, model, img_pil, prompt, threshold):
    inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[img_pil.size[::-1]]
    )
    return results[0]["boxes"]

class SegmentV2:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "prompt": "Enter the object or scene you want to segment. Use tag-style or natural language for more detailed prompts.",
            "threshold": "Adjust mask detection strength (higher = more strict)",
            "mask_blur": "Apply Gaussian blur to mask edges (0 = disabled)",
            "mask_offset": "Expand/Shrink mask boundary (positive = expand, negative = shrink)",
            "invert_output": "Invert the mask output",
            "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": "Choose background type"}),
            "background_color": "Choose background color (Alpha = transparent)",
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Object to segment", "tooltip": tooltips["prompt"]}),
                "sam_model": (list(SAM_MODELS.keys()),),
                "dino_model": (list(DINO_MODELS.keys()),),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.01, "tooltip": tooltips["threshold"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLOR", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_v2"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.dino_model_cache = {}
        self.sam_model_cache = {}

    def segment_v2(self, image, prompt, sam_model, dino_model, threshold=0.30,
                   mask_blur=0, mask_offset=0, background="Alpha", 
                   background_color="#222222", invert_output=False):
        img_pil = tensor2pil(image[0]) if image.ndim == 4 else tensor2pil(image)
        img_np = np.array(img_pil.convert("RGB"))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load GroundingDINO config and weights
        dino_info = DINO_MODELS[dino_model]
        config_path = get_or_download_model_file(dino_info["config_filename"], dino_info["config_url"], "grounding-dino")
        weights_path = get_or_download_model_file(dino_info["model_filename"], dino_info["model_url"], "grounding-dino")

        # Load and cache GroundingDINO model
        dino_key = (config_path, weights_path, device)
        if dino_key not in self.dino_model_cache:
            args = SLConfig.fromfile(config_path)
            model = build_model(args)
            checkpoint = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            model.eval()
            model.to(device)
            self.dino_model_cache[dino_key] = model
        dino = self.dino_model_cache[dino_key]

        # Preprocess image for DINO
        from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
        transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(img_pil.convert("RGB"), None)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Prepare text prompt
        text_prompt = prompt if prompt.endswith(".") else prompt + "."

        # Forward pass
        with torch.no_grad():
            outputs = dino(image_tensor, captions=[text_prompt])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # Filter boxes by threshold
        filt_mask = logits.max(dim=1)[0] > threshold
        boxes_filt = boxes[filt_mask]
        if boxes_filt.shape[0] == 0:
            width, height = img_pil.size
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
            empty_mask_rgb = empty_mask.reshape((-1, 1, height, width)).movedim(1, -1).expand(-1, -1, -1, 3)
            result_image = apply_background_color(img_pil, Image.fromarray((empty_mask[0].numpy() * 255).astype(np.uint8)), background, background_color)
            return (pil2tensor(result_image), empty_mask, empty_mask_rgb)

        # Convert boxes to xyxy
        H, W = img_pil.size[1], img_pil.size[0]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt)
        boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H], dtype=torch.float32, device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy.cpu().numpy()

        # Download/check SAM weights
        sam_info = SAM_MODELS[sam_model]
        sam_ckpt_path = get_or_download_model_file(sam_info["filename"], sam_info["model_url"], "SAM")

        # Load SAM model (cache to avoid reloading)
        sam_key = (sam_info["model_type"], sam_ckpt_path, device)
        if sam_key not in self.sam_model_cache:
            sam = sam_model_registry[sam_info["model_type"]](checkpoint=sam_ckpt_path)
            sam.to(device)
            self.sam_model_cache[sam_key] = SamPredictor(sam)
        predictor = self.sam_model_cache[sam_key]

        # Use SAM to get masks for each box
        predictor.set_image(img_np)
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_tensor, img_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        # Process mask following the original implementation
        print(f"Mask shape before processing: {masks.shape}")
        # Combine all masks into one
        combined_mask = torch.max(masks, dim=0)[0]  # Take maximum across all masks
        mask = combined_mask.float().cpu().numpy()
        print(f"Mask shape after processing: {mask.shape}")
        # Squeeze out the extra dimension to get a 2D array
        mask = mask.squeeze(0)
        print(f"Final mask shape: {mask.shape}")
        mask = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask, mode="L")

        mask_image = process_mask(mask_pil, invert_output, mask_blur, mask_offset)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        if background == "Color":
            result_image = result_image.convert("RGB")
        else:
            result_image = result_image.convert("RGBA")
        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_image_vis = mask_tensor.reshape((-1, 1, mask_image.height, mask_image.width)).movedim(1, -1).expand(-1, -1, -1, 3)
        return (pil2tensor(result_image), mask_tensor, mask_image_vis)

NODE_CLASS_MAPPINGS = {
    "SegmentV2": SegmentV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegmentV2": "Segmentation V2 (RMBG)",
}


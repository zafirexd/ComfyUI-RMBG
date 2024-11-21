import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize
import numpy as np
import folder_paths
from transformers import AutoModelForImageSegmentation
from PIL import ImageFilter
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

AVAILABLE_MODELS = {
    "RMBG-2.0": "briaai/RMBG-2.0"
}

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class AILAB_RMBG:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0")
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "sensitivity": "Adjust mask detection strength",
            "process_res": "Processing resolution (higher = more VRAM)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "background": "Choose background color (Alpha = transparent background)",
            "invert_output": "Invert both image and mask output",
        }
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_version": (list(AVAILABLE_MODELS.keys()),),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "background": (["Alpha", "black", "white", "green", "blue", "red"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "🧪AILab/🧽RMBG"

    def check_model_cache(self, model_version):
        model_files_path = os.path.join(self.cache_dir)
        
        if not os.path.exists(self.cache_dir):
            return False, "Model directory not found"
        
        required_files = [
            'config.json',
            'model.safetensors',
            'birefnet.py',
            'BiRefNet_config.py'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_files_path, f))]
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache is complete"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.current_model_version = None
            torch.cuda.empty_cache()
            print("Model cleared from memory")

    def download_model_files(self, model_version):
        model_id = AVAILABLE_MODELS[model_version]
        required_files = {
            'config.json': 'config.json',
            'model.safetensors': 'model.safetensors',
            'birefnet.py': 'birefnet.py',
            'BiRefNet_config.py': 'BiRefNet_config.py'
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            for filename, save_name in required_files.items():
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=self.cache_dir,
                    local_dir_use_symlinks=False
                )
                
                if os.path.dirname(downloaded_path) != self.cache_dir:
                    target_path = os.path.join(self.cache_dir, save_name)
                    shutil.move(downloaded_path, target_path)
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def remove_background(self, image, model_version, sensitivity=1.0, process_res=1024, 
                         mask_blur=0, mask_offset=0, invert_output=False, background="Alpha"):
        try:
            cache_status, message = self.check_model_cache(model_version)
            
            if not cache_status:
                print(f"Model cache status: {message}")
                print("Downloading required model files...")
                download_status, download_message = self.download_model_files(model_version)
                if not download_status:
                    raise RuntimeError(download_message)
                print("Download completed.")
            
            if self.current_model_version != model_version or self.model is None:
                if self.model is not None:
                    self.clear_model()
                
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
                torch.set_float32_matmul_precision('high')
                self.model.to(device)
                self.model.eval()
                self.current_model_version = model_version
                print(f"Loaded model version: {model_version}")

            processed_images = []
            processed_masks = []
            
            bg_colors = {
                "Alpha": None,
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "red": (255, 0, 0)
            }
            
            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            for img in image:
                orig_image = tensor2pil(img)
                w, h = orig_image.size
                
                input_tensor = transform_image(orig_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    result = self.model(input_tensor)[-1].sigmoid().cpu()
                    result = result[0].squeeze()
                    
                    result = result * (1 + (1 - sensitivity))
                    result = torch.clamp(result, 0, 1)
                    
                    result = F.interpolate(result.unsqueeze(0).unsqueeze(0), 
                                         size=(h, w), 
                                         mode='bilinear').squeeze()
                    
                    mask_pil = tensor2pil(result)
                    
                    if invert_output:
                        mask_np = np.array(mask_pil)
                        mask_np = 255 - mask_np
                        mask_pil = Image.fromarray(mask_np)
                    
                    if mask_blur > 0:
                        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                    
                    if mask_offset != 0:
                        if mask_offset > 0:
                            for _ in range(mask_offset):
                                mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
                        else:
                            for _ in range(-mask_offset):
                                mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
                    
                    rgba_image = orig_image.copy().convert('RGBA')
                    rgba_image.putalpha(mask_pil)
                    
                    if background != "Alpha":
                        bg_color = bg_colors[background]
                        bg_image = Image.new('RGBA', orig_image.size, (*bg_color, 255))
                        composite_image = Image.alpha_composite(bg_image, rgba_image)
                        processed_images.append(pil2tensor(composite_image))
                    else:
                        processed_images.append(pil2tensor(rgba_image))
                    
                    processed_masks.append(pil2tensor(mask_pil))

            torch.cuda.empty_cache()
            
            new_ims = torch.cat(processed_images, dim=0)
            new_masks = torch.cat(processed_masks, dim=0)
            
            return (new_ims, new_masks)
        
        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in RMBG processing: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "AILAB_RMBG": AILAB_RMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILAB_RMBG": "RMBG (Remove Background)"
}
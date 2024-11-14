import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import folder_paths
from transformers import AutoModelForImageSegmentation
from PIL import ImageFilter

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
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_version": (list(AVAILABLE_MODELS.keys()),),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "process_res": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 128}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "🧪AILab/🧽RMBG"

    def remove_background(self, image, model_version, sensitivity=0.5, process_res=1024, mask_blur=0, mask_offset=0):
        try:
            cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0")
            
            if self.current_model_version != model_version or self.model is None:
                model_path = AVAILABLE_MODELS[model_version]
                model_files_path = os.path.join(cache_dir, model_version.replace("/", "--"))
                
                if not os.path.exists(model_files_path):
                    print(f"Downloading {model_version} model... This may take a while.")
                
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    revision="main",
                    local_files_only=False
                )
                torch.set_float32_matmul_precision('high')
                self.model.to(device)
                self.model.eval()
                self.current_model_version = model_version
                print(f"Loaded model version: {model_version}")

            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        except Exception as e:
            raise RuntimeError(f"Error loading RMBG model: {str(e)}")

        processed_images = []
        processed_masks = []
        
        for img in image:
            orig_image = tensor2pil(img)
            input_tensor = transform_image(orig_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()
                pred = (pred > sensitivity).float()
                
                mask = transforms.ToPILImage()(pred)
                mask = mask.resize(orig_image.size)
                
                if mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                
                if mask_offset != 0:
                    from PIL import ImageMorph
                    if mask_offset > 0:
                        pattern = [[1,1,1],[1,1,1],[1,1,1]]
                        for _ in range(mask_offset):
                            mask = mask.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-mask_offset):
                            mask = mask.filter(ImageFilter.MinFilter(3))
                
                new_im = orig_image.copy()
                new_im.putalpha(mask)
                
                new_im_tensor = pil2tensor(new_im)
                mask_tensor = pil2tensor(mask)
                
                processed_images.append(new_im_tensor)
                processed_masks.append(mask_tensor)
        
        torch.cuda.empty_cache()
        
        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)
        
        return (new_ims, new_masks)
    
NODE_CLASS_MAPPINGS = {
    "AILAB_RMBG": AILAB_RMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILAB_RMBG": "🧽 RMBG (Remove Background)"
}
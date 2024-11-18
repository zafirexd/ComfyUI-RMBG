# ComfyUI-RMBG

A ComfyUI node for removing image backgrounds using RMBG-2.0.

![RMBG_3](https://github.com/user-attachments/assets/f3ffa3c4-5a21-4c0c-a078-b4ffe681c4c4)

## Features

RMBG-2.0 is built on the innovative BiRefNet (Bilateral Reference Network) architecture, offering:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image

## Installation

1. Clone this repository to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
```

2. RMBG Model Download:

- The model will be automatically downloaded to `ComfyUI/models/RMBG/` when first time using the custom node.

## Usage
![RMBG](https://github.com/user-attachments/assets/cd0eb92e-8f2e-4ae4-95f1-899a6d83cab6)

### Optional Settings :bulb: Tips
| Optional Settings    | :memo: Description                                                           | :bulb: Tips                                                                                   |
|----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Sensitivity**      | Adjusts the strength of mask detection. Higher values result in stricter detection. | Default value is 0.5. Adjust based on image complexity; more complex images may require higher sensitivity. |
| **Processing Resolution** | Controls the processing resolution of the input image, affecting detail and memory usage. | Choose a value between 256 and 2048, with a default of 1024. Higher resolutions provide better detail but increase memory consumption. |
| **Mask Blur**        | Controls the amount of blur applied to the mask edges, reducing jaggedness. | Default value is 0. Try setting it between 1 and 5 for smoother edge effects.                    |
| **Mask Offset**      | Allows for expanding or shrinking the mask boundary. Positive values expand the boundary, while negative values shrink it. | Default value is 0. Adjust based on the specific image, typically fine-tuning between -10 and 10. |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage
1. Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2. Connect an image to the input
3. Get two outputs:
   - IMAGE: Processed image with transparent background
   - MASK: Binary mask of the foreground

### Parameters
- `sensitivity`: Controls the background removal sensitivity (0.0-1.0)
- `process_res`: Processing resolution (512-2048, step 128)
- `mask_blur`: Blur amount for the mask (0-64)
- `mask_offset`: Adjust mask edges (-20 to 20)

## About RMBG-2.0

RMBG-2.0 is developed by BRIA AI and uses the BiRefNet architecture which includes:

- **Localization Module (LM)**: Generates semantic maps for primary image areas
- **Restoration Module (RM)**: Performs precise boundary restoration using:
  - Original Reference: Provides general background context
  - Gradient Reference: Focuses on edges and fine details

The model is trained on a diverse dataset of over 15,000 high-quality images, ensuring:
- Balanced representation across different image types
- High accuracy in various scenarios
- Robust performance with complex backgrounds

## Requirements
- ComfyUI
- Python 3.10+
- Required packages (automatically installed):
  - torch>=2.0.0
  - torchvision>=0.15.0
  - Pillow>=9.0.0
  - numpy>=1.22.0
  - transformers>=4.30.0
  - safetensors>=0.3.0

## Credits
- RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
- Created by: [1038 Lab](https://github.com/1038lab)

## License
MIT License

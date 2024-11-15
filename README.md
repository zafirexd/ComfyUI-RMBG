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

2. Download RMBG model:
- Download from [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
- Place the [RMBG-2.0 model](https://huggingface.co/briaai/RMBG-2.0/tree/main) files in `ComfyUI/models/RMBG/RMBG-2.0/` directory

## Usage

### Basic Usage
1. Load `🧽 RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
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

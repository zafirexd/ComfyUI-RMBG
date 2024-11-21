# AILAB RMBG Update Log

## Version 1.1.0

### New Features
- Added background color options
  - Alpha (transparent background)
  - Black, White, Green, Blue, Red
    
![rmbg1 1](https://github.com/user-attachments/assets/4f7d073c-f9cc-4bdb-875c-ba51decc9d5a)

- Improved mask processing
  - Better detail preservation
  - Enhanced edge quality
  - More accurate segmentation
    
![rmbg version compare](https://github.com/user-attachments/assets/8339aa8e-46db-4f11-aa7b-0a710f0a1711)

- Added video batch processing
  - Support for video file background removal
  - Maintains original video framerate and resolution
  - Multiple output format support (with Alpha channel)
  - Efficient batch processing for video frames

https://github.com/user-attachments/assets/259220d3-c148-4030-93d6-c17dd5bccee1

- Added model cache management
  - Cache status checking
  - Model memory cleanup
  - Better error handling

### Parameter Updates
- Renamed 'invert_mask' to 'invert_output' for clarity
- Added sensitivity adjustment for mask strength
- Updated tooltips for better clarity

### Technical Improvements
- Optimized image processing pipeline
- Added proper model cache verification
- Improved memory management
- Better error handling and recovery
- Enhanced batch processing performance for videos

### Dependencies
- Added timm>=0.6.12,<1.0.0 for model support
- Updated requirements.txt with version constraints

### Bug Fixes
- Fixed mask detail preservation issues
- Improved mask edge quality
- Fixed memory leaks in model handling

### Usage Notes
- The 'Alpha' background option provides transparent background
- Sensitivity parameter now controls mask strength
- Model cache is checked before each operation
- Memory is automatically cleaned when switching models
- Video processing supports various formats and maintains quality

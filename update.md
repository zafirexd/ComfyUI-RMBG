# AILAB RMBG Update Log

## Version 1.1.0

### New Features
- Added background color options
  - Alpha (transparent background)
  - Black
  - White
  - Green
  - Blue
  - Red

- Improved mask processing
  - Better detail preservation
  - Enhanced edge quality
  - More accurate segmentation

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
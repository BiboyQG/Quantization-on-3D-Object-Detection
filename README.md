# Q-LiDAR: Dynamic and Static Precision Quantization for 3D Object Detection

This repository contains the implementation of efficient quantization techniques for 3D object detection, specifically focusing on the CenterPoint algorithm. Our approach combines dynamic and static post-training quantization methods to significantly reduce computational overhead while maintaining detection accuracy.

## Key Features

- **Hybrid Quantization Framework**
  - Dynamic post-training quantization (PTQ) for adaptive precision
  - Static PTQ for consistent performance
  - Progressive quantization strategy with 16-bit activation preservation
  - Customized quantization for Sparse 3D convolutions

- **Performance Improvements**
  - 35% reduction in inference time
  - Only 1% accuracy trade-off
  - Optimized memory usage through precision reduction

## Technical Highlights

### Progressive Quantization Strategy
Our approach maintains 16-bit activations while progressively quantizing other operators, achieving an optimal balance between precision and computational efficiency. We've developed a custom quantization strategy specifically designed for Sparse 3D convolutions.

### Sensitivity Analysis
The repository includes tools for quantization sensitivity analysis, helping identify efficiency-critical variables. This enables:
- Precise model tuning
- Minimal accuracy impact
- Enhanced interpretability of the quantization process

### SmoothQuant Integration
We incorporate SmoothQuant techniques to address extreme outlier issues commonly encountered in direct PTQ approaches, effectively recovering accuracy losses.

## Getting Started

### Installation
Follow OpenPCDet's official steps to setup the environment. Then:
```bash
git clone https://github.com/BiboyQG/Quantization-on-3D-Object-Detection.git
cd Quantization-on-3D-Object-Detection
```

## Results

Our quantization approach achieves:
- 35% reduction in inference time
- Minimal accuracy loss (~1%)
- Improved memory efficiency
- Better handling of outlier cases through SmoothQuant integration

## Contributors

- [Banghao Chi](https://biboyqg.github.io/)
- Hongbo Zheng
- Advisor: [Minjia Zhang](https://minjiazhang.github.io/)

## License

[License information will be added]

## Acknowledgments

This work was conducted at the University of Illinois Urbana-Champaign under the guidance of Prof. Minjia Zhang.

Computer Vision: Image Enhancement & Bit-Plane Analysis

This repository contains implementations and deep-dive analyses for fundamental digital image processing techniques. The focus is on understanding how visual information is encoded at the bit level and how statistical redistributions can enhance image quality.
Key Learning Objectives:

    Bit-Plane Slicing: Decomposing an 8-bit image into binary planes to distinguish between structural information and noise.

    Image Reconstruction & Sensitivity: Measuring the impact of individual bits on image fidelity using MSE, PSNR, and Entropy.

    Histogram Equalization (HE): Implementing Global HE from scratch to understand Cumulative Distribution Functions (CDF).

    Adaptive Histogram Equalization (AHE): Comparing Tiling and Sliding Window approaches to handle local contrast variations and avoid over-amplification
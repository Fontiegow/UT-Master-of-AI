# Breast Cancer Classification using KNN

This project focuses on a deep dive into the **K-Nearest Neighbors (KNN)** algorithm using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to understand how preprocessing, feature selection, noise, and distance metrics affect model performance and interpretability.

## Project Objectives
1. [cite_start]**Understanding Preprocessing**: Analyze the critical impact of feature scaling (Standardization) on distance-based models.
2. [cite_start]**K-Value Optimization**: Use Cross-Validation and Grid Search to find the optimal $K$ and study the bias-variance tradeoff[cite: 29].
3. [cite_start]**Feature Engineering**: Perform correlation analysis to reduce dimensionality and improve model speed/accuracy[cite: 37, 41, 43].
4. [cite_start]**Robustness Analysis**: Test model sensitivity by injecting artificial noise into labels and features[cite: 61, 75].
5. [cite_start]**Decision Boundaries**: Visualize how scaling and $K$ change the classification landscape[cite: 84, 90].
6. [cite_start]**Distance Metrics**: Compare Euclidean, Manhattan, and other metrics (Minkowski/Chebyshev)[cite: 92, 93, 94, 102].

## Dataset Information
- [cite_start]**Name**: Breast Cancer Wisconsin (Diagnostic)[cite: 15].
- [cite_start]**Features**: 30 numeric attributes (Mean, SE, and "Worst" measurements of cell nuclei)[cite: 15, 18, 19, 20].
- [cite_start]**Classes**: Malignant (M) and Benign (B)[cite: 15].
- [cite_start]**Source**: [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)[cite: 16].

## Implementation Strategy
The project is implemented in Python, avoiding high-level libraries for the core KNN logic where possible to ensure a deep understanding of the underlying mathematics.

---
[cite_start]*Developed as part of the Machine Learning Course - University of Tehran (Fall 2024)*[cite: 1, 12].
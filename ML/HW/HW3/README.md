# 🚗 Car Price Prediction with MLP (From Scratch) Q1

This project implements a **Multilayer Perceptron (MLP) regression model from scratch (NumPy only)** to predict car prices using the *Cars Dataset 2025*.  
No machine learning libraries (e.g. sklearn, PyTorch, TensorFlow) are used for the model itself.

The goal is to understand the **full ML pipeline**: data preprocessing, encoding, normalization, model design, training, debugging, and evaluation.

---

## Problem Statement

Given a dataset of car attributes (brand, model, year, mileage, engine specs, etc.),  
predict the **car price** as a continuous value using a neural network.

This is a **supervised regression problem**.

---

## Project Structure

The project is divided into four main parts:

1. Data preprocessing
2. Feature encoding & scaling
3. MLP implementation from scratch
4. Training, evaluation, and debugging

---

## Part 1 – Data Preprocessing

Steps:

* Separate features `X` and target `y = Price`
* Handle missing values using:
  * mode for categorical
  * mean for numerical

**Why:**  
Neural networks cannot handle NaNs and require fully numeric input.

---

## Part 2 – Encoding & Scaling

### Target Encoding (for categorical features)

Instead of one-hot encoding, **target encoding** was used:

Each category → replaced by the mean price of that category in training data.

This:

* Keeps dimensionality small
* Works better for high-cardinality features (brands, models)

### Standardization

All features are standardized:

X_std = (X - μ) / σ


Target `y` is also standardized.

**Why:**  
Without scaling, gradients exploded and training diverged.

---

## Part 3 – MLP From Scratch

Implemented manually:

* Architecture: Input → 64 → 32 → 1
* Activations: Leaky ReLU
* Loss: Mean Squared Error (MSE)
* Optimization: Gradient Descent
* Initialization: Xavier

Everything is done using **pure NumPy**:

* Forward pass
* Backpropagation
* Weight updates

---

## Part 4 – Training & Evaluation

Dataset split:

* 70% train
* 15% validation
* 15% test

Metrics:

* Validation MSE
* Test MSE

---

## Major Problems Encountered (and Fixed)

### 1. Loss Exploding to 10¹⁰⁰+

**Cause:**

* Features not scaled
* Learning rate too high
* Bad weight initialization

**Fix:**

* Standardization
* Xavier initialization
* Smaller learning rate
* Leaky ReLU instead of ReLU

---

### 2. Loss Constant at ~1.0 (No Learning)

**Cause:**

* Vanishing gradients
* Dead neurons (ReLU)

**Fix:**

* Switched to Leaky ReLU
* Verified gradients numerically

---

### 3. Data Leakage

**Cause:**

* Encoding before train/val split

**Fix:**

* Always split first
* Fit encoders only on training set

---

## Final Outcome

The final model trains stably and produces finite MSE on validation and test sets.

The goal was not maximum accuracy, but to:

> **Build a scientifically correct ML pipeline from scratch and understand every failure mode.**

This project demonstrates:

* Real neural network math
* Real ML debugging
* Real experimental methodology

Not just calling `.fit()`.


---
# Unsupervised Car Recommendation System Q2

This repository contains a from-scratch implementation of a car recommendation engine using unsupervised clustering techniques.  
The project focuses on segmenting the automotive market based on price and performance metrics **without using high-level ML libraries** like `scikit-learn`.

---

## 🚀 Step-by-Step Implementation

1. **Data Preprocessing**:  
   Extracted numeric values from raw strings (e.g., "$25,000", "340 km/h", "70-85 hp").  
   Handled interval data by calculating the mean of the range.

2. **Manual Standardization**:  
   Implemented Z-score normalization using `NumPy` to ensure all features (Price, Horsepower, etc.) contribute equally to distance calculations.

3. **DBSCAN Implementation**:  
   * Found the optimal `Eps` using the **K-Distance Graph**.  
   * Implemented the clustering logic (Core, Border, and Noise points) from scratch.

4. **OPTICS Clustering**:  
   Created an ordering of points to identify density-based structures and visualized them via **Reachability Plots**.

5. **Hierarchical Clustering**:  
   Compared **Single, Complete, and Average Linkage** methods and visualized the taxonomy using Dendrograms.

6. **Recommendation Logic**:  
   Developed a system that maps user input to the nearest cluster centroid and ranks the **top 5 closest vehicles**.

---

## 🛠 Challenges & Solutions

| Problem | Technical Fix |
| --- | --- |
| **High Dimensionality**: One-Hot encoding categorical data created 2,900+ features, making distance-based clustering impossible ("Curse of Dimensionality"). | Switched to **Label Encoding** and focused on high-variance numerical features (Price & Power) to maintain cluster density. |
| **Non-Numeric Data**: Raw data contained units ("hp", "Nm") and price ranges ("$10k-$12k") which crashed standard math operations. | Built a **Regex-based parser** to extract floats and average out intervals during the loading phase. |
| **Parameter Sensitivity**: DBSCAN is highly sensitive to the `Eps` value; setting it manually resulted in either 1 giant cluster or 100% noise. | Implemented a **K-Distance Plot** logic. By identifying the "knee" (elbow) of the sorted neighbor distances, we found the mathematically optimal `Eps` (0.3). |
| **Efficiency**: Calculating distance matrices in pure Python is slow. | Optimized calculations using **NumPy broadcasting** to vectorize Euclidean distance operations, reducing execution time from minutes to milliseconds. |

---

## 📊 Key Results

* **Optimal Clustering**: DBSCAN achieved a **Silhouette Score of 0.6981**, indicating highly distinct and meaningful car segments.  
* **Insights**: The system successfully separated the market into two primary tiers: "Economy/Utility" and "Luxury/High-Performance," while correctly identifying 25 unique exotic vehicles as "Noise."

---

## 💻 Tech Stack

* **Language**: Python 3.x  
* **Libraries**: Numpy, Pandas (Data Manipulation), Matplotlib (Visualization)  
* **Algorithm Logic**: Manual implementations of DBSCAN, OPTICS, and Agglomerative Clustering.

---

### How to Use

1. Clone the repository.  
2. Run the notebook/script.  
3. Input your desired car specs (Price, HP) to get **5 tailored recommendations**.

---

### Pro-Tip for GitHub

Include the **DBSCAN Visualization** and the **Reachability Plot** images in your `assets` folder and link them in the README.  
This makes the project look much more polished and professional.


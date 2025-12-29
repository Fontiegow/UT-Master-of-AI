# Breast Cancer Classification using KNN

This project focuses on a deep dive into the **K-Nearest Neighbors (KNN)** algorithm using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to understand how preprocessing, feature selection, noise, and distance metrics affect model performance and interpretability.

## Project Objectives
1. **Understanding Preprocessing**: Analyze the critical impact of feature scaling (Standardization) on distance-based models.
2. **K-Value Optimization**: Use Cross-Validation and Grid Search to find the optimal $K$ and study the bias-variance tradeoff[cite: 29].
3. **Feature Engineering**: Perform correlation analysis to reduce dimensionality and improve model speed/accuracy[cite: 37, 41, 43].
4. **Robustness Analysis**: Test model sensitivity by injecting artificial noise into labels and features[cite: 61, 75].
5. **Decision Boundaries**: Visualize how scaling and $K$ change the classification landscape[cite: 84, 90].
6. **Distance Metrics**: Compare Euclidean, Manhattan, and other metrics (Minkowski/Chebyshev)[cite: 92, 93, 94, 102].

## Dataset Information
- **Name**: Breast Cancer Wisconsin (Diagnostic)[cite: 15].
- **Features**: 30 numeric attributes (Mean, SE, and "Worst" measurements of cell nuclei)[cite: 15, 18, 19, 20].
- **Classes**: Malignant (M) and Benign (B)[cite: 15].
- **Source**: [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)[cite: 16].

## Implementation Strategy
The project is implemented in Python, avoiding high-level libraries for the core KNN logic where possible to ensure a deep understanding of the underlying mathematics.

--- *Developed as part of the Machine Learning Course - University of Tehran (Fall 2024)*[cite: 1, 12].

---
## 1. Data Preparation and Custom Train-Test Split

In this stage, we handle the initial data pipeline. Since the goal is a deep understanding of the mechanics, we avoid high-level libraries and implement the data splitting logic from scratch using `NumPy`.

### Data Cleaning Process:
* **Irrelevant Features**: Dropped the `id` and any empty columns.
* **Label Encoding**: Mapped the categorical `diagnosis` column to numerical values ($M = 1$, $B = 0$).
* **Feature-Target Separation**: Isolated the 30 descriptive features ($X$) from the target label ($y$).

### Custom Split Logic
To evaluate the model's performance on unseen data, we implement a manual `train_test_split`. This ensures we understand how data shuffling prevents ordering bias.

```python
def custom_train_test_split(X, y, test_size=0.2, seed=42):
    """
    Splits the dataset into training and testing sets manually.
    
    1. Determines the number of test samples based on 'test_size'.
    2. Shuffles the indices randomly using a 'seed' for reproducibility.
    3. Slices the shuffled indices to create distinct training and testing pools.
    """
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # Shuffle indices to ensure the split is representative
    np.random.seed(seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices into two groups
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Index the original arrays to produce the final sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Execution
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, seed=42)

## 1.2. Custom KNN Implementation (No Scaling)

In this phase, we implement the K-Nearest Neighbors (KNN) algorithm **from scratch** using `NumPy`.
The objective is to establish a baseline performance on raw (unscaled) data and observe how feature magnitudes affect distance-based reasoning.

---

### Core Implementation Functions

Below are the fundamental building blocks of our custom KNN model.

---

### 1. Euclidean Distance

Calculates the straight-line distance between two points in a multi-dimensional space.

```python
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

2. Neighbor Selection (get_neighbors)

This function acts as the search engine of the model.
It computes distances between a test sample and all training samples, sorts them, and selects the top K nearest neighbors.

def get_neighbors(X_train, y_train, x_test_row, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_test_row, X_train[i])
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])  # Sort by distance
    return [item[1] for item in distances[:k]]  # Return labels of top K

3. Majority Voting (predict_classification)

After identifying the K nearest neighbors, the predicted class is determined by majority voting.

def predict_classification(neighbors_labels):
    counts = np.bincount(neighbors_labels)
    return np.argmax(counts)

4. K-Fold Cross-Validation (custom_grid_search_cv)

To select the optimal value of K without leaking information from the test set,
we implement 5-fold cross-validation.

For each value of:

K ∈ [1, 16]

the model is trained on 4 folds and validated on the remaining fold.

def custom_grid_search_cv(X, y, k_values, n_folds=5):
    fold_size = len(X) // n_folds
    scores = {}

    for k in k_values:
        fold_accuracies = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size

            X_val = X[start:end]
            y_val = y[start:end]

            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)

            correct = 0
            for i in range(len(X_val)):
                neighbors = get_neighbors(X_train, y_train, X_val[i], k)
                prediction = predict_classification(neighbors)
                if prediction == y_val[i]:
                    correct += 1

            accuracy = correct / len(X_val)
            fold_accuracies.append(accuracy)

        scores[k] = np.mean(fold_accuracies)

    return scores

Notes

No feature scaling is applied in this implementation

Distance sensitivity is intentional and analyzed in later phases

All logic is implemented manually to expose algorithmic behavior



## 3. Implementing KNN with Feature Selection

Highly-correlated features can unduly influence the distance calculation in KNN. By selecting a representative subset of features, we aim to reduce redundancy and potentially improve generalization (Accuracy).

### Method Steps:
1.  **Correlation Analysis:** Analyze the correlation within the three feature groups (Mean, SE, Worst).
2.  **Feature Selection:** Select a final, reduced set of features that are less correlated with each other.
3.  **Scaling and Grid Search:** Apply the custom `CustomStandardScaler` to the reduced feature set and repeat the Grid Search (5-Fold CV) to find the new optimal $K$.
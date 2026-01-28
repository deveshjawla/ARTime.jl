# Utils API Reference

This document provides detailed API reference for the Utils module, which contains utility functions for evaluating anomaly detection performance.

## Table of Contents

- [Module Overview](#module-overview)
- [Functions](#functions)

## Module Overview

The Utils module provides utility functions for computing performance metrics for binary classification tasks, particularly anomaly detection. It includes functions for computing confusion matrices, F1 scores, and balanced accuracy.

### Dependencies

```julia
using Statistics
```

### Usage Pattern

```julia
using .ARTime  # Include utils

# After running anomaly detection
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]  # Ground truth
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]  # Detector output

# Calculate metrics
cm = calculate_confusion_matrix(true_labels, predicted_labels)
f1 = calc_f1_score(cm)
bacc = calc_balanced_accuracy(cm)

println("F1 Score: $f1")
println("Balanced Accuracy: $bacc")
```

## Functions

### calculate_confusion_matrix

```julia
calculate_confusion_matrix(y_true::AbstractVector, y_pred::AbstractVector) -> NamedTuple
```

Calculate the confusion matrix for binary classification results.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | `AbstractVector` | Ground truth labels (0 = normal, 1 = anomaly). |
| `y_pred` | `AbstractVector` | Predicted labels (0 = normal, 1 = anomaly). |

#### Returns

- `NamedTuple`: Contains the following fields:
  - `tp::Int`: True Positive count
  - `tn::Int`: True Negative count
  - `fp::Int`: False Positive count
  - `fn::Int`: False Negative count
  - `matrix::Matrix{Int}`: 2x2 confusion matrix

#### Description

This function computes a confusion matrix that summarizes the performance of a binary classifier. The matrix shows the counts of:

- **True Positives (TP)**: Anomalies correctly detected
- **True Negatives (TN)**: Normal samples correctly identified
- **False Positives (FP)**: Normal samples incorrectly flagged as anomalies
- **False Negatives (FN)**: Anomalies missed by the detector

##### Confusion Matrix Layout

```
                Predicted
              0      1
Actual  0  [TN]   [FP]
        1  [FN]   [TP]
```

The returned matrix has this layout:
```julia
matrix = [tn  fp
           fn  tp]
```

##### Interpretation

**High TP + TN**: Good overall accuracy
**High FP**: Too sensitive (many false alarms)
**High FN**: Not sensitive enough (missing anomalies)

#### Example

```julia
# Ground truth and predictions
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]

# Calculate confusion matrix
cm = calculate_confusion_matrix(true_labels, predicted_labels)

println("True Positives: $(cm.tp)")   # 2
println("True Negatives: $(cm.tn)")   # 4
println("False Positives: $(cm.fp)")  # 1
println("False Negatives: $(cm.fn)")  # 1

println("Confusion Matrix:")
println(cm.matrix)
# Output:
# 4  1
# 1  2
```

#### Notes

- Assumes binary classification with labels 0 (negative) and 1 (positive)
- Throws `DimensionMismatch` if vectors have different lengths
- The matrix representation follows the convention: rows = actual, columns = predicted
- Used by [`calc_f1_score`](@ref) and [`calc_balanced_accuracy`](@ref)

---

### calc_f1_score

```julia
calc_f1_score(stats::NamedTuple) -> Float64
```

Calculate F1 score from confusion matrix statistics.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `stats` | `NamedTuple` | Confusion matrix statistics with fields: `tp`, `tn`, `fp`, `fn`. |

#### Returns

- `Float64`: F1 score in range [0, 1]. Returns 0.0 if denominator is 0.

#### Description

The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.

##### Formula

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

Where:
- **TP**: True Positives (correctly detected anomalies)
- **FP**: False Positives (normal samples flagged as anomalies)
- **FN**: False Negatives (anomalies missed by detector)

##### Interpretation

**F1 = 1.0**: Perfect performance (no errors)
**F1 = 0.0**: Worst performance (no correct detections)
**F1 > 0.7**: Good performance
**F1 < 0.5**: Poor performance

##### Why F1 Score?

1. **Harmonic Mean**: Penalizes extreme values more than arithmetic mean
2. **Balances Precision and Recall**: Doesn't favor one over the other
3. **Robust to Imbalance**: Works well even with few anomalies

##### Relationship to Other Metrics

```
Precision = TP / (TP + FP)      # How many detected are actually anomalies?
Recall = TP / (TP + FN)          # How many anomalies were detected?
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

#### Example

```julia
# Get confusion matrix
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]
cm = calculate_confusion_matrix(true_labels, predicted_labels)

# Calculate F1 score
f1 = calc_f1_score(cm)
println("F1 Score: $f1")
# Output: F1 Score: 0.6667

# Interpretation
if f1 > 0.7
    println("Good performance")
elseif f1 > 0.5
    println("Moderate performance")
else
    println("Poor performance")
end
```

#### Notes

- Returns 0.0 if denominator is 0 (no true positives)
- Higher values indicate better performance
- Commonly used in anomaly detection evaluation
- Used by NAB (Numenta Anomaly Benchmark) for scoring

---

### calc_balanced_accuracy

```julia
calc_balanced_accuracy(stats::NamedTuple) -> Float64
```

Calculate balanced accuracy from confusion matrix statistics.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `stats` | `NamedTuple` | Confusion matrix statistics with fields: `tp`, `tn`, `fp`, `fn`. |

#### Returns

- `Float64`: Balanced accuracy in range [0, 1]. Returns 0.5 if no samples in a class.

#### Description

Balanced accuracy is the arithmetic mean of sensitivity and specificity. It provides a fair metric for imbalanced datasets, which are common in anomaly detection (few anomalies, many normal samples).

##### Formula

```
Balanced Accuracy = (Sensitivity + Specificity) / 2

Where:
Sensitivity (Recall) = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

##### Components

**Sensitivity (True Positive Rate)**:
```
Sensitivity = TP / (TP + FN)
```
- Measures ability to detect anomalies
- Also called Recall or True Positive Rate (TPR)
- Range: [0, 1]
- Higher values indicate better anomaly detection

**Specificity (True Negative Rate)**:
```
Specificity = TN / (TN + FP)
```
- Measures ability to identify normal samples
- Also called True Negative Rate (TNR)
- Range: [0, 1]
- Higher values indicate fewer false alarms

##### Why Balanced Accuracy?

1. **Handles Imbalance**: Works well when anomalies are rare
2. **Fair Metric**: Doesn't favor majority class
3. **Interpretable**: Easy to understand (average of two rates)
4. **Robust**: Less sensitive to class distribution than accuracy

##### Comparison to Regular Accuracy

```
Regular Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

For imbalanced data (e.g., 1% anomalies):
- Regular accuracy can be 99% by always predicting "normal"
- Balanced accuracy requires good performance on both classes

#### Example

```julia
# Get confusion matrix
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]
cm = calculate_confusion_matrix(true_labels, predicted_labels)

# Calculate balanced accuracy
bacc = calc_balanced_accuracy(cm)
println("Balanced Accuracy: $bacc")
# Output: Balanced Accuracy: 0.75

# Interpretation
if bacc > 0.8
    println("Excellent performance")
elseif bacc > 0.7
    println("Good performance")
elseif bacc > 0.6
    println("Moderate performance")
else
    println("Poor performance")
end

# Compare with regular accuracy
regular_accuracy = (cm.tp + cm.tn) / sum([cm.tp, cm.tn, cm.fp, cm.fn])
println("Regular Accuracy: $regular_accuracy")
# Output: Regular Accuracy: 0.75 (same in this balanced example)
```

#### Notes

- Returns 0.5 if a class has no samples (neutral value)
- Higher values indicate better performance
- Particularly useful for anomaly detection where anomalies are rare
- Used by NAB (Numenta Anomaly Benchmark) for evaluation
- Balances the trade-off between detecting anomalies and avoiding false alarms

---

### performance_stats_multiclass

```julia
performance_stats_multiclass(true_labels, predicted_labels) -> Tuple
```

Calculate all performance metrics for binary classification results.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `true_labels` | `AbstractVector` | Ground truth labels (0 = normal, 1 = anomaly). |
| `predicted_labels` | `AbstractVector` | Predicted labels (0 = normal, 1 = anomaly). |

#### Returns

- `Tuple`: Contains:
  - `bacc::Float64`: Balanced accuracy
  - `f1::Float64`: F1 score
  - `cm::NamedTuple`: Confusion matrix statistics

#### Description

This is a convenience function that computes all performance metrics in one call. It internally calls [`calculate_confusion_matrix`](@ref), [`calc_f1_score`](@ref), and [`calc_balanced_accuracy`](@ref).

#### Example

```julia
# Ground truth and predictions
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]

# Calculate all metrics
bacc, f1, cm = performance_stats_multiclass(true_labels, predicted_labels)

println("Balanced Accuracy: $bacc")
println("F1 Score: $f1")
println("Confusion Matrix:")
println(cm.matrix)
```

#### Notes

- Converts labels to `Vector{Int}` for consistency
- Provides a convenient interface for computing all metrics
- Returns metrics in a tuple for easy unpacking

"""
# Utils Module

This module provides utility functions for evaluating anomaly detection performance.
It includes functions for computing confusion matrices, F1 scores, and
balanced accuracy metrics.

## Metrics Overview

### Confusion Matrix

A confusion matrix summarizes the performance of a classification algorithm:

```
				Predicted
			  0      1
Actual  0  [TN]   [FP]
		1  [FN]   [TP]
```

Where:
- **TP (True Positive)**: Correctly detected anomalies
- **TN (True Negative)**: Correctly identified normal samples
- **FP (False Positive)**: Normal samples incorrectly flagged as anomalies
- **FN (False Negative)**: Anomalies missed by the detector

### F1 Score

The F1 score is the harmonic mean of precision and recall:

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

- Range: [0, 1]
- Higher values indicate better performance
- Balances precision (avoiding false positives) and recall (catching anomalies)

### Balanced Accuracy

Balanced accuracy is the arithmetic mean of sensitivity and specificity:

```
Balanced Accuracy = (Sensitivity + Specificity) / 2

Where:
Sensitivity (Recall) = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

- Range: [0, 1]
- Higher values indicate better performance
- Useful for imbalanced datasets (common in anomaly detection)

## Usage Example

julia
using ARTime
using .ARTime  # Include utils

# After running anomaly detection
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]  # Ground truth
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]  # Detector output

# Calculate metrics
cm = calculate_confusion_matrix(true_labels, predicted_labels)
f1 = calc_f1_score(cm)
bacc = calc_balanced_accuracy(cm)

	println("Confusion Matrix:")
	println(cm.matrix)
	println("F1 Score: ", f1)
	println("Balanced Accuracy: ", bacc)

# Or use the convenience function
bacc, f1, cm = performance_stats_multiclass(true_labels, predicted_labels)


## Notes

- All functions assume binary classification (0 = normal, 1 = anomaly)
- Metrics are computed from the test period (after probationary period)
- These metrics are commonly used in the NAB (Numenta Anomaly Benchmark) evaluation
"""
function performance_stats_multiclass(true_labels, predicted_labels)
	true_labels = convert(Vector{Int}, vec(true_labels))
	predicted_labels = convert(Vector{Int}, vec(predicted_labels))
	cm = calculate_confusion_matrix(true_labels, predicted_labels)
	bacc = calc_balanced_accuracy(cm)
	f1 = calc_f1_score(cm)

	return bacc, f1, cm
end

"""
	calculate_confusion_matrix(y_true, y_pred) -> NamedTuple

Calculate the confusion matrix for binary classification results.

## Arguments

- `y_true::AbstractVector`: Ground truth labels (0 = normal, 1 = anomaly).
- `y_pred::AbstractVector`: Predicted labels (0 = normal, 1 = anomaly).

## Returns

- `NamedTuple`: Contains the following fields:
  - `tp::Int`: True Positive count
  - `tn::Int`: True Negative count
  - `fp::Int`: False Positive count
  - `fn::Int`: False Negative count
  - `matrix::Matrix{Int}`: 2x2 confusion matrix

## Description

This function computes a confusion matrix that summarizes the performance of
a binary classifier. The matrix shows the counts of:

- **True Positives (TP)**: Anomalies correctly detected
- **True Negatives (TN)**: Normal samples correctly identified
- **False Positives (FP)**: Normal samples incorrectly flagged as anomalies
- **False Negatives (FN)**: Anomalies missed by the detector

### Confusion Matrix Layout

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

### Interpretation

**High TP + TN**: Good overall accuracy
**High FP**: Too sensitive (many false alarms)
**High FN**: Not sensitive enough (missing anomalies)

## Example

```julia
# Ground truth and predictions
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]

# Calculate confusion matrix
cm = calculate_confusion_matrix(true_labels, predicted_labels)

	println("True Positives: ", cm.tp)   # 2
	println("True Negatives: ", cm.tn)   # 4
	println("False Positives: ", cm.fp)  # 1
	println("False Negatives: ", cm.fn)  # 1

println("Confusion Matrix:")
println(cm.matrix)
# Output:
# 4  1
# 1  2

# Use with other metrics
f1 = calc_f1_score(cm)
bacc = calc_balanced_accuracy(cm)
```

## Notes

- Assumes binary classification with labels 0 (negative) and 1 (positive)
- Throws `DimensionMismatch` if vectors have different lengths
- The matrix representation follows the convention: rows = actual, columns = predicted
- Used by [`calc_f1_score`](@ref) and [`calc_balanced_accuracy`](@ref)
"""
function calculate_confusion_matrix(y_true::AbstractVector, y_pred::AbstractVector)
	if length(y_true) != length(y_pred)
		throw(DimensionMismatch("Vectors must be the same length"))
	end

	tp = 0  # True Positive
	tn = 0  # True Negative
	fp = 0  # False Positive
	fn = 0  # False Negative

	@inbounds for (t, p) in zip(y_true, y_pred)
		if t == 1 && p == 1
			tp += 1
		elseif t == 0 && p == 0
			tn += 1
		elseif t == 0 && p == 1
			fp += 1
		elseif t == 1 && p == 0
			fn += 1
		end
	end

	# Return raw counts and a formal matrix (Rows=Actual, Cols=Predicted)
	# [TN  FP]
	# [FN  TP]
	matrix = [tn fp; fn tp]

	return (tp = tp, tn = tn, fp = fp, fn = fn, matrix = matrix)
end

"""
	calc_f1_score(stats) -> Float64

Calculate F1 score from confusion matrix statistics.

## Arguments

- `stats::NamedTuple`: Confusion matrix statistics with fields:
  - `tp::Int`: True Positive count
  - `tn::Int`: True Negative count
  - `fp::Int`: False Positive count
  - `fn::Int`: False Negative count

## Returns

- `Float64`: F1 score in range [0, 1]. Returns 0.0 if denominator is 0.

## Description

The F1 score is the harmonic mean of precision and recall, providing
a single metric that balances both concerns.

### Formula

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

Where:
- **TP**: True Positives (correctly detected anomalies)
- **FP**: False Positives (normal samples flagged as anomalies)
- **FN**: False Negatives (anomalies missed by detector)

### Interpretation

**F1 = 1.0**: Perfect performance (no errors)
**F1 = 0.0**: Worst performance (no correct detections)
**F1 > 0.7**: Good performance
**F1 < 0.5**: Poor performance

### Why F1 Score?

1. **Harmonic Mean**: Penalizes extreme values more than arithmetic mean
2. **Balances Precision and Recall**: Doesn't favor one over the other
3. **Robust to Imbalance**: Works well even with few anomalies

### Relationship to Other Metrics

```
Precision = TP / (TP + FP)      # How many detected are actually anomalies?
Recall = TP / (TP + FN)          # How many anomalies were detected?
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## Example

```julia
# Get confusion matrix
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]
cm = calculate_confusion_matrix(true_labels, predicted_labels)

	# Calculate F1 score
	f1 = calc_f1_score(cm)
	println("F1 Score: ", f1)
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

## Notes

- Returns 0.0 if denominator is 0 (no true positives)
- Higher values indicate better performance
- Commonly used in anomaly detection evaluation
- Used by NAB (Numenta Anomaly Benchmark) for scoring
"""
function calc_f1_score(stats)
	# F1 = 2TP / (2TP + FP + FN)
	denominator = (2 * stats.tp) + stats.fp + stats.fn

	if denominator == 0
		return 0.0
	end

	return (2 * stats.tp) / denominator
end

"""
	calc_balanced_accuracy(stats) -> Float64

Calculate balanced accuracy from confusion matrix statistics.

## Arguments

- `stats::NamedTuple`: Confusion matrix statistics with fields:
  - `tp::Int`: True Positive count
  - `tn::Int`: True Negative count
  - `fp::Int`: False Positive count
  - `fn::Int`: False Negative count

## Returns

- `Float64`: Balanced accuracy in range [0, 1]. Returns 0.5 if no samples in a class.

## Description

Balanced accuracy is the arithmetic mean of sensitivity and specificity.
It provides a fair metric for imbalanced datasets, which are common
in anomaly detection (few anomalies, many normal samples).

### Formula

```
Balanced Accuracy = (Sensitivity + Specificity) / 2

Where:
Sensitivity (Recall) = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

### Components

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

### Why Balanced Accuracy?

1. **Handles Imbalance**: Works well when anomalies are rare
2. **Fair Metric**: Doesn't favor majority class
3. **Interpretable**: Easy to understand (average of two rates)
4. **Robust**: Less sensitive to class distribution than accuracy

### Comparison to Regular Accuracy

```
Regular Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

For imbalanced data (e.g., 1% anomalies):
- Regular accuracy can be 99% by always predicting "normal"
- Balanced accuracy requires good performance on both classes

## Example

```julia
# Get confusion matrix
true_labels = [0, 0, 1, 0, 1, 0, 0, 1]
predicted_labels = [0, 0, 1, 1, 1, 0, 0, 0]
cm = calculate_confusion_matrix(true_labels, predicted_labels)

	# Calculate balanced accuracy
	bacc = calc_balanced_accuracy(cm)
	println("Balanced Accuracy: ", bacc)
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
	println("Regular Accuracy: ", regular_accuracy)
	# Output: Regular Accuracy: 0.75 (same in this balanced example)
```

## Notes

- Returns 0.5 if a class has no samples (neutral value)
- Higher values indicate better performance
- Particularly useful for anomaly detection where anomalies are rare
- Used by NAB (Numenta Anomaly Benchmark) for evaluation
- Balances the trade-off between detecting anomalies and avoiding false alarms
"""
function calc_balanced_accuracy(stats)
	# Sensitivity (Recall) = TP / (TP + FN)
	actual_positives = stats.tp + stats.fn
	sensitivity = actual_positives == 0 ? 0.5 : stats.tp / actual_positives

	# Specificity = TN / (TN + FP)
	actual_negatives = stats.tn + stats.fp
	specificity = actual_negatives == 0 ? 0.5 : stats.tn / actual_negatives

	return (sensitivity + specificity) / 2
end
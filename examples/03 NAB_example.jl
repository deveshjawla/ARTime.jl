# Include the ARTime module
include("../src/ARTime.jl")
include("../src/EvalMetrics.jl")
using .ARTime
using CSV, DataFrames

#TODO - Assume that the first anomaly detected is the changepoint and then second anomaly after that is the changepoint returning the process to normal = For this, we ask the user, what kind of process are they monitoring? If they expect point anomalies such as fraud detection, then we use the point AD, if it is process monitoring then we consider the point anomalies as changepoints. Also depending on the problem, we can adjust the vigilance parameters etc?!

"""
# NAB (Numenta Anomaly Benchmark) Example

This script demonstrates how to use ARTime for anomaly detection on
real-world data from the NAB benchmark.

## Overview

The NAB benchmark is a standard dataset for evaluating anomaly detection
algorithms on real-world time series data. This example uses the
"art_daily_flatmiddle" dataset, which contains:

- **Daily art server metrics**
- **Real-world anomalies**: Known anomaly periods with labels
- **Flat middle pattern**: Characteristic pattern with occasional anomalies

## Usage

Run the example:
```bash
julia --project=. examples/NAB_example.jl
```

## Process

**1. Load Data**:
```julia
df = CSV.read("Data/ARTime_art_daily_flatmiddle", DataFrame)
ts = Vector(df.value)
```

**2. Initialize Detector**:
```julia
tsmin = minimum(ts)
tsmax = maximum(ts)
tslength = lastindex(ts)
tsd = ARTime.TimeSeriesDetector()
ARTime.init(ts, tsd)
```

**3. Detect Anomalies**:
```julia
anomalyscores = map(x -> ARTime.process_sample!(x, tsd), ts)
```

## Output

The script produces:
- `anomalyscores`: Vector of anomaly scores for each time step
  - Scores > 0 indicate detected anomalies
  - Scores are in range [0, 1]

## Next Steps

After running this example, you can:

1. **Visualize Results**:
```julia
using Plots
plot(ts, label = "Signal")
scatter!(findall(x -> x > 0, anomalyscores),
	ts[findall(x -> x > 0, anomalyscores)],
	label = "Detected Anomalies", color = :red)
```

2. **Evaluate Performance**:
```julia
# Load ground truth labels (if available)
true_labels = load_nab_labels("art_daily_flatmiddle")

# Calculate metrics
cm = calculate_confusion_matrix(true_labels, anomalyscores .> 0)
f1 = calc_f1_score(cm)
bacc = calc_balanced_accuracy(cm)
```

3. **Save Results**:
```julia
# Save anomaly scores
CSV.write("anomaly_scores", DataFrame(score = anomalyscores))

# Save detected anomalies
detected_indices = findall(x -> x > 0, anomalyscores)
CSV.write("detected_anomalies",
	DataFrame(index = detected_indices,
				score = anomalyscores[detected_indices]))
```

## Notes

- Uses `--project=.` flag to use local Project.toml
- The data file should be in `Data/` directory
- Anomaly scores are computed for all time steps
- Scores > 0 indicate detected anomalies
- The probationary period is handled automatically by ARTime
- For full NAB evaluation, you need to compare with ground truth labels

## NAB Benchmark

The NAB benchmark includes:
- Multiple real-world datasets
- Labeled anomaly periods
- Standard evaluation metrics
- Comparison with other algorithms

For more information about NAB:
https://github.com/numenta/NAB
"""

# Load NAB data
df = CSV.read("Data/ARTime_art_daily_flatmiddle", DataFrame)
ts = Vector(df.value)

# Get data statistics
tsmin = minimum(ts)
tsmax = maximum(ts)
tslength = lastindex(ts)

# Initialize ARTime detector
tsd = ARTime.TimeSeriesDetector()

# Configure detector with data bounds
jline = ARTime.init(ts, tsd)

# Process all samples and detect anomalies
anomalyscores = map(x -> ARTime.process_sample!(x, tsd), ts)

# Output results
println("Processed ", tslength, " samples")
println("Anomaly scores computed")
println("Number of detected anomalies: ", sum(anomalyscores .> 0))
println("Max anomaly score: ", maximum(anomalyscores))

# ==============================================================================
# EVALUATION WITH THRESHOLD TUNING
# ==============================================================================

"""
	evaluate_with_threshold(anomalyscores, true_labels; threshold = 0.0) -> NamedTuple

Evaluate anomaly detection performance using a threshold on anomaly scores.

## Arguments

- `anomalyscores::Vector{Float64}`: Anomaly confidence scores from ARTime.
- `true_labels::Vector{Int}`: Ground truth labels (0 = normal, 1 = anomaly).
- `threshold::Float64`: Threshold for converting scores to binary predictions.
  Scores >= threshold are classified as anomalies (default: 0.0, meaning any score > 0).

## Returns

- `NamedTuple`: Contains:
  - `threshold::Float64`: Threshold used
  - `cm::NamedTuple`: Confusion matrix statistics
  - `f1::Float64`: F1 score
  - `bacc::Float64`: Balanced accuracy
  - `n_detected::Int`: Number of detected anomalies

## Description

This function converts continuous anomaly scores to binary predictions using a threshold
and computes performance metrics against ground truth labels.

## Threshold Selection Strategies

**1. Default (threshold = 0.0)**:
- Any score > 0 is classified as anomaly
- This is ARTime's built-in threshold
- Good for initial evaluation

**2. Fixed Threshold (e.g., threshold = 0.5)**:
- Only scores >= 0.5 are classified as anomalies
- More conservative (fewer false positives)
- May miss some true anomalies

**3. Percentile-Based**:
- Use top X% of scores as anomalies
- Example: threshold = quantile(anomalyscores, 0.95)
- Ensures consistent detection rate

**4. Grid Search**:
- Try multiple thresholds and pick best F1
- See [`tune_threshold`](@ref) function

## Example

```julia
# Use default threshold (any score > 0)
result = evaluate_with_threshold(anomalyscores, true_labels)

# Use fixed threshold
result = evaluate_with_threshold(anomalyscores, true_labels, threshold = 0.5)

# Access results
println("Threshold: ", result.threshold)
println("F1 Score: ", result.f1)
println("Balanced Accuracy: ", result.bacc)
println("Confusion Matrix:")
println(result.cm.matrix)
```

## Notes

- Ground truth labels should be 0 (normal) or 1 (anomaly)
- Only evaluates after probationary period
- Higher threshold = fewer detections (more conservative)
- Lower threshold = more detections (more sensitive)
"""
function evaluate_with_threshold(anomalyscores, true_labels; threshold = 0.0)
	# Convert scores to binary predictions
	predicted_labels = Int.(anomalyscores .>= threshold)

	# Calculate metrics
	cm = calculate_confusion_matrix(true_labels, predicted_labels)
	f1 = calc_f1_score(cm)
	bacc = calc_balanced_accuracy(cm)
	n_detected = sum(predicted_labels)

	return (threshold = threshold, cm = cm, f1 = f1, bacc = bacc, n_detected = n_detected)
end

"""
	tune_threshold(anomalyscores, true_labels; n_thresholds = 20, min_threshold = 0.0, max_threshold = 1.0) -> NamedTuple

Find optimal threshold for anomaly detection by grid search.

## Arguments

- `anomalyscores::Vector{Float64}`: Anomaly confidence scores from ARTime.
- `true_labels::Vector{Int}`: Ground truth labels (0 = normal, 1 = anomaly).
- `n_thresholds::Int`: Number of thresholds to try (default: 20).
- `min_threshold::Float64`: Minimum threshold to try (default: 0.0).
- `max_threshold::Float64`: Maximum threshold to try (default: 1.0).

## Returns

- `NamedTuple`: Contains:
  - `best_threshold::Float64`: Threshold with highest F1 score
  - `best_f1::Float64`: Best F1 score achieved
  - `best_bacc::Float64`: Balanced accuracy at best threshold
  - `best_cm::NamedTuple`: Confusion matrix at best threshold
  - `all_results::Vector{NamedTuple}`: Results for all thresholds tried

## Description

This function performs a grid search over threshold values to find the one that
maximizes F1 score. This is useful for tuning the detector to achieve
optimal performance on a specific dataset.

## Grid Search Process

1. **Generate Thresholds**: Create `n_thresholds` evenly spaced values
2. **Evaluate Each Threshold**: Compute F1 and balanced accuracy
3. **Select Best**: Choose threshold with highest F1 score
4. **Return Results**: Best threshold and all results for analysis

## Example

```julia
# Find optimal threshold
results = tune_threshold(anomalyscores, true_labels)

println("Best threshold: ", results.best_threshold)
println("Best F1 Score: ", results.best_f1)
println("Best Balanced Accuracy: ", results.best_bacc)

# Analyze all thresholds
for result in results.all_results
	println("Threshold: ", result.threshold, ", F1: ", result.f1, ", BACC: ", result.bacc)
end
```

## Notes

- Grid search is simple but effective for threshold tuning
- For large datasets, consider using percentile-based thresholds
- The best threshold may vary between datasets
- F1 score is used as the optimization metric
"""
function tune_threshold(anomalyscores, true_labels; n_thresholds = 20, min_threshold = 0.0, max_threshold = 1.0)
	# Generate threshold values
	thresholds = range(min_threshold, stop = max_threshold, length = n_thresholds)

	# Evaluate each threshold
	all_results = []
	for threshold in thresholds
		result = evaluate_with_threshold(anomalyscores, true_labels, threshold = threshold)
		push!(all_results, result)
	end

	# Find best threshold (highest F1)
	best_idx = argmax([r.f1 + r.bacc for r in all_results])
	best_result = all_results[best_idx]

	return (
		best_threshold = best_result.threshold,
		best_f1 = best_result.f1,
		best_bacc = best_result.bacc,
		best_cm = best_result.cm,
		all_results = all_results,
	)
end

df = CSV.read("Data/ARTime_art_daily_flatmiddle", DataFrame)
true_labels = df.label
probationary_period = Int(floor(0.15 * tslength))
test_period_start = probationary_period + 1

changepoints = abs.(diff(true_labels))

result_default = evaluate_with_threshold(anomalyscores[(test_period_start+1):end],
	changepoints[test_period_start:end],
	threshold = 0.5)
println("Threshold: ", result_default.threshold)
println("F1 Score: ", round(result_default.f1, digits = 4))
println("Balanced Accuracy: ", round(result_default.bacc, digits = 4))
println("Confusion Matrix:")
println(result_default.cm.matrix)
println("Detected anomalies: ", result_default.n_detected)
println("True anomalies: ", sum(true_labels[test_period_start:end]))

# Example 3: Tune threshold using grid search
println("\n--- Threshold Tuning (Grid Search) ---")
tuning_results = tune_threshold(anomalyscores[test_period_start:end],
	true_labels[test_period_start:end],
	n_thresholds = 20,
	min_threshold = 0.0,
	max_threshold = 1.0)
println("Best threshold: ", round(tuning_results.best_threshold, digits = 4))
println("Best F1 Score: ", round(tuning_results.best_f1, digits = 4))
println("Best Balanced Accuracy: ", round(tuning_results.best_bacc, digits = 4))
println("Confusion Matrix at best threshold:")
println(tuning_results.best_cm.matrix)

# Example 4: Percentile-based threshold
println("\n--- Example 4: Percentile-Based Threshold ---")
percentile_95 = quantile(anomalyscores[test_period_start:end], 0.95)
result_percentile = evaluate_with_threshold(anomalyscores[test_period_start:end],
	true_labels_demo[test_period_start:end],
	threshold = percentile_95)
println("95th percentile threshold: ", round(percentile_95, digits = 4))
println("F1 Score: ", round(result_percentile.f1, digits = 4))
println("Balanced Accuracy: ", round(result_percentile.bacc, digits = 4))
println("Confusion Matrix:")
println(result_percentile.cm.matrix)
println("Detected anomalies: ", result_percentile.n_detected)

println("\n" * "="^70)
println("THRESHOLD TUNING SUMMARY")
println("="^70)
println("\nThreshold Selection Strategies:")
println("1. Default (score > 0): Uses ARTime's built-in threshold")
println("2. Fixed threshold: Set a specific value (e.g., 0.5)")
println("3. Percentile-based: Use top X% of scores (e.g., 95th percentile)")
println("4. Grid search: Try multiple thresholds and pick best F1")
println("\nRecommendations:")
println("- Start with default threshold (score > 0)")
println("- If too many false positives: Increase threshold")
println("- If missing anomalies: Decrease threshold")
println("- Use grid search to find optimal threshold for your dataset")
println("- Consider percentile-based thresholds for consistent detection rates")
println("\nFor NAB evaluation:")
println("- Load ground truth labels from NAB JSON files")
println("- Use NAB's scoring script for official evaluation")
println("- See: https://github.com/numenta/NAB/tree/master/scoring")
println("="^70)
# Example of Anomaly Detection using Multivariate Time Series

"""
# Multivariate Anomaly Detection Example

This script demonstrates how to use ARTime for anomaly detection on
synthetic multivariate time series data. It includes:

1. **Data Generation**: Creating synthetic multivariate time series with anomalies
2. **Detection**: Running ARTime on each feature independently
3. **Aggregation**: Combining detections across features
4. **Evaluation**: Computing performance metrics (F1, Balanced Accuracy)
5. **Cross-Validation**: Running 10-fold experiments with different seeds
6. **Visualization**: Plotting results for each feature

## Overview

The example generates a synthetic multivariate time series with:
- Multiple features (e.g., 3 different signals)
- Base signals: Different sine waves with noise
- Anomalies: Random spikes in random features
- Configurable: Number of points, features, anomalies, and magnitude

## Multivariate Approach

**Current Implementation**:
- Each feature is processed independently with its own ARTime instance
- Anomalies are aggregated across features (OR logic)
- No communication between features during training

**Limitations**:
- Features don't share information during training
- May confuse the model when anomalies occur in different features
- TODO: Implement multi-threading for parallel feature processing

## Usage

Run the 10-fold experiment:
```bash
julia --project=. examples/multivariate_anomaly_detection.jl
```

Run with visualization (uncomment the last line):
```bash
julia --project=. examples/multivariate_anomaly_detection.jl
# Then uncomment: run_example_with_plot(42)
```

## Functions

- `generate_multivariate_data`: Creates synthetic multivariate time series with anomalies
- `main`: Runs detection for a single random seed
- `run_10_fold`: Runs 10-fold cross-validation
- `run_example_with_plot`: Runs detection and creates visualization

## Output

The script prints:
- Per-fold results (F1, Balanced Accuracy)
- Average metrics with standard deviations
- Saves plots (one per feature) if visualization enabled

## Notes

- Uses the `--project=.` flag to use local Project.toml
- Anomalies are only evaluated after the probationary period
- The 10-fold experiment uses different random seeds for robustness
- Each feature has its own ARTime instance (independent processing)
"""

using Plots
using Random
using Statistics

# Include the ARTime module and utils
include("../src/ARTime.jl")
include("../src/EvalMetrics.jl")
using .ARTime

#TODO = Implement multi threading for the features (train on all the time steps of one feature on a single thread). And multi processing for the various time series (train on all the features of the time series on the same process)
#	= For multivariate case, we are confusing the model when we tell that at time step i, there is an anomaly, but we do not specify which feature. Since there is not communication between the features (yet), it will be confuding the model.

"""
	generate_multivariate_data(n_points = 1000; n_features = 3, anomaly_points = 5, anomaly_magnitude = 3.0)

Generate synthetic multivariate time series data with injected anomalies.

## Arguments

- `n_points::Int`: Number of data points to generate (default: 1000).
- `n_features::Int`: Number of features (time series) to generate (default: 3).
- `anomaly_points::Int`: Number of anomalies to inject (default: 5).
- `anomaly_magnitude::Float64`: Magnitude of anomaly spikes (default: 3.0).

## Returns

- `signal::Matrix{Float64}`: Generated multivariate time series with anomalies.
  Dimensions: [n_points, n_features].
- `anomaly_positions::Vector{Int}`: Indices where anomalies were injected.

## Description

This function creates a synthetic multivariate time series that simulates
real-world multi-sensor data with periodic patterns and occasional anomalies.

### Signal Generation

**Base Signals**:
```julia
t = range(0, stop = 10π, length = n_points)
signal1 = sin.(t) .+ 0.1 .* randn(n_points)
signal2 = cos.(t) .+ 0.1 .* randn(n_points)
signal3 = sin.(2t) .+ 0.1 .* randn(n_points)
```

- Creates different periodic patterns for each feature
- Feature 1: Sine wave
- Feature 2: Cosine wave
- Feature 3: Double-frequency sine wave
- Adds Gaussian noise (σ = 0.1) for realism

**Combine Features**:
```julia
signal = hcat(signal1, signal2, signal3)
```
Creates a matrix where each column is a feature.

### Anomaly Injection

```julia
anomaly_positions = rand(1:n_points, anomaly_points)
for pos in anomaly_positions
	feature_idx = rand(1:n_features)
	signal[pos, feature_idx] += anomaly_magnitude * (rand() > 0.5 ? 1 : -1)
end
```

- Randomly selects positions for anomalies
- Randomly selects which feature gets the anomaly
- Adds positive or negative spikes (random direction)
- Only one feature is affected per anomaly

### Characteristics

**Normal Data**:
- Range: Approximately [-1.1, 1.1] for each feature
- Pattern: Different periodic patterns per feature
- Noise: Small Gaussian variations
- Correlation: Features are independent (different patterns)

**Anomalous Data**:
- Range: Approximately [-4.1, 4.1] (including spikes)
- Pattern: Sudden, large deviations in one feature
- Duration: Single point spikes
- Localization: Only one feature affected per anomaly

## Example

```julia
# Generate default data (3 features)
signal, anomaly_positions = generate_multivariate_data()

	println("Generated ", size(signal), " matrix")
	println("Anomalies at positions: ", anomaly_positions)

# Generate custom data
signal, anomaly_positions = generate_multivariate_data(
	n_points = 500,
	n_features = 5,
	anomaly_points = 10,
	anomaly_magnitude = 5.0
)

# Visualize each feature
for feature_idx in 1:size(signal, 2)
	plot(signal[:, feature_idx], label = "Feature ", feature_idx)
	scatter!(anomaly_positions, signal[anomaly_positions, feature_idx],
		label = "Anomalies", color = :red, markersize = 5)
end
xlabel!("Time")
ylabel!("Value")
title!("Multivariate Time Series with Anomalies")
```

## Notes

- Anomalies are randomly positioned (may overlap)
- Each anomaly affects only one feature
- Features have different periodic patterns (independent)
- Noise level (0.1) is small relative to anomaly magnitude (3.0)
- This creates a clear distinction between normal and anomalous data
- Used for testing and demonstration purposes
"""
# Generate synthetic multivariate time series with anomalies
function generate_multivariate_data(n_points = 1000; n_features = 3, anomaly_points = 5, anomaly_magnitude = 3.0)
	# Base signals: sine waves with noise
	t = range(0, stop = 10π, length = n_points)
	signal1 = sin.(t) .+ 0.1 .* randn(n_points)
	signal2 = cos.(t) .+ 0.1 .* randn(n_points)
	signal3 = sin.(2t) .+ 0.1 .* randn(n_points)

	# Combine into a multivariate signal
	signal = hcat(signal1, signal2, signal3)

	# Add anomalies at random positions
	anomaly_positions = rand(round(Int, 0.15*n_points):n_points, anomaly_points)
	for pos in anomaly_positions
		# Randomly select a feature to add anomaly
		feature_idx = rand(1:n_features)
		signal[pos, feature_idx] += anomaly_magnitude * (rand() > 0.5 ? 1 : -1)
	end

	return signal, anomaly_positions
end

"""
	main(seed::Int) -> NamedTuple

Run multivariate anomaly detection experiment with a given random seed.

## Arguments

- `seed::Int`: Random seed for reproducibility.

## Returns

- `NamedTuple`: Contains:
  - `f1_score::Float64`: F1 score on test period
  - `balanced_accuracy::Float64`: Balanced accuracy on test period
  - `confusion_matrix::NamedTuple`: Confusion matrix statistics

## Description

This function runs a complete multivariate anomaly detection experiment:

1. **Data Generation**: Creates synthetic multivariate time series with anomalies
2. **Detector Initialization**: Sets up ARTime for each feature independently
3. **Detection**: Processes all features and detects anomalies
4. **Aggregation**: Combines detections across features (OR logic)
5. **Evaluation**: Computes performance metrics on test period

### Steps

**1. Generate Data**:
```julia
Random.seed!(seed)
signal, anomaly_positions = generate_multivariate_data()
```

**2. Initialize Detectors** (one per feature):
```julia
dmin = minimum(signal, dims = 1)
dmax = maximum(signal, dims = 1)
dlength = size(signal, 1)

for feature_idx in 1:size(signal, 2)
	ts = ARTime.TimeSeries()
	ARTime.init(dmin[feature_idx], dmax[feature_idx], dlength, ts)
end
```

**3. Detect Anomalies** (independent per feature):
```julia
anomalies = zeros(size(signal))
for feature_idx in 1:size(signal, 2)
	for (i, A) in enumerate(signal[:, feature_idx])
		anomalies[i, feature_idx] += ARTime.process_sample!(A, ts)
	end
end
```

**4. Aggregate Detections** (OR logic):
```julia
# Anomaly detected if ANY feature detects it
for i in test_period_start:dlength
	if any(anomalies[i, :] .> 0)
		predicted_labels[i] = 1
	end
end
```

**5. Evaluate Performance**:
- Only evaluate after probationary period
- Create true labels from anomaly positions
- Create predicted labels from aggregated detections
- Compute confusion matrix, F1, and balanced accuracy

### Multivariate Approach

**Independent Processing**:
- Each feature has its own ARTime instance
- No communication between features during training
- Detections are aggregated using OR logic

**Aggregation Strategy**:
- Anomaly at time t if ANY feature detects anomaly
- This is a conservative approach (high sensitivity)
- May increase false positives but ensures no anomalies are missed

### Probationary Period

The first `probationary_period` samples are used for training:
- No anomalies are reported during this period
- Each feature learns its own normal patterns
- Evaluation starts at `test_period_start = probationary_period + 1`

## Example

```julia
# Run experiment with seed 42
result = main(42)

	println("F1 Score: ", result.f1_score)
	println("Balanced Accuracy: ", result.balanced_accuracy)
	println("Confusion Matrix:")
	println(result.confusion_matrix.matrix)

# Access individual metrics
tp = result.confusion_matrix.tp
tn = result.confusion_matrix.tn
fp = result.confusion_matrix.fp
fn = result.confusion_matrix.fn
```

## Notes

- Uses fixed random seed for reproducibility
- Each feature is processed independently (no cross-feature communication)
- Metrics are computed only on test period (after probationary)
- Anomaly detected if ANY feature detects it (OR aggregation)
- The function is called by [`run_10_fold`](@ref) for cross-validation
- TODO: Implement multi-threading for parallel feature processing
"""
# Main function to demonstrate multivariate anomaly detection with a given seed
function main(seed::Int)
	# Set the random seed
	Random.seed!(seed)

	# Generate synthetic multivariate data
	signal, anomaly_positions = generate_multivariate_data()

	# Initialize ARTime parameters for each feature
	dmin = minimum(signal, dims = 1)
	dmax = maximum(signal, dims = 1)
	dlength = size(signal, 1)

	# Process each feature separately and detect anomalies
	anomalies = zeros(size(signal))
	for feature_idx in 1:size(signal, 2)
		ts = ARTime.TimeSeries()
		ARTime.init(dmin[feature_idx], dmax[feature_idx], dlength, ts)

		for (i, A) in enumerate(signal[:, feature_idx])
			anomalies[i, feature_idx] += ARTime.process_sample!(A, ts)
		end
	end

	# Calculate metrics for the test period (after probationary period)
	probationary_period = Int(floor(0.15 * dlength))
	test_period_start = probationary_period + 1

	# calculate metrics here
	# For multivariate case, we consider an anomaly detected if any feature detects it
	# Create true labels (1 for anomaly, 0 for normal) for test period
	true_labels = zeros(Int, dlength)
	for pos in anomaly_positions
		if pos >= test_period_start
			true_labels[pos] = 1
		end
	end

	# Create predicted labels (1 for detected anomaly, 0 for normal) for test period
	# An anomaly is detected if any feature detects it
	predicted_labels = zeros(Int, dlength)
	for i in test_period_start:dlength
		if any(anomalies[i, :] .> 0)
			predicted_labels[i] = 1
		end
	end

	# Calculate metrics using utils functions
	cm = calculate_confusion_matrix(true_labels[test_period_start:end], predicted_labels[test_period_start:end])
	f1_score = calc_f1_score(cm)
	balanced_accuracy = calc_balanced_accuracy(cm)

	return (f1_score = f1_score, balanced_accuracy = balanced_accuracy, confusion_matrix = cm)
end

"""
	run_10_fold() -> NamedTuple

Run 10-fold cross-validation experiment for multivariate anomaly detection.

## Returns

- `NamedTuple`: Contains:
  - `avg_f1::Float64`: Average F1 score across all folds
  - `avg_balanced_accuracy::Float64`: Average balanced accuracy across all folds
  - `std_f1::Float64`: Standard deviation of F1 scores
  - `std_balanced_accuracy::Float64`: Standard deviation of balanced accuracies
  - `all_f1_scores::Vector{Float64}`: F1 scores for each fold
  - `all_balanced_accuracies::Vector{Float64}`: Balanced accuracies for each fold
  - `confusion_matrices::Vector`: Confusion matrices for each fold

## Description

This function performs 10-fold cross-validation to evaluate ARTime's
performance on multivariate data across different random seeds.

### Process

**1. Define Seeds**:
```julia
seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
```
Uses 10 different random seeds for reproducibility.

**2. Run Experiments**:
```julia
for seed in seeds
	result = main(seed)
	# Store metrics
end
```
Runs [`main`](@ref) for each seed and collects results.

**3. Compute Statistics**:
```julia
avg_f1 = mean(f1_scores)
avg_balanced_accuracy = mean(balanced_accuracies)
std_f1 = std(f1_scores)
std_balanced_accuracy = std(balanced_accuracies)
```
Computes mean and standard deviation of metrics.

**4. Print Results**:
- Per-fold results (F1, Balanced Accuracy)
- Average metrics with standard deviations

### Output Format

```
Seed 42: F1 = 0.6891, Balanced Accuracy = 0.7956
Seed 123: F1 = 0.7123, Balanced Accuracy = 0.8034
...
Seed 2021: F1 = 0.7045, Balanced Accuracy = 0.8012

=== 10-Fold Results ===
Average F1 Score: 0.7023 ± 0.0089
Average Balanced Accuracy: 0.8001 ± 0.0034
```

### Why 10-Fold?

1. **Robustness**: Tests performance across different data distributions
2. **Statistical Significance**: Provides confidence intervals
3. **Reproducibility**: Fixed seeds ensure consistent results
4. **Benchmarking**: Standard practice in machine learning

### Multivariate Considerations

- Each fold processes multiple features independently
- Detections are aggregated across features (OR logic)
- Performance may vary based on which features have anomalies
- Standard deviation indicates consistency across different random seeds

## Example

```julia
# Run 10-fold experiment
results = run_10_fold()

	# Access average metrics
	println("Average F1: ", results.avg_f1)
	println("Average Balanced Accuracy: ", results.avg_balanced_accuracy)
	
	# Access variability
	println("F1 Std Dev: ", results.std_f1)
	println("Balanced Accuracy Std Dev: ", results.std_balanced_accuracy)
	
	# Access per-fold results
	println("All F1 Scores: ", results.all_f1_scores)
	println("All Balanced Accuracies: ", results.all_balanced_accuracies)
	
	# Analyze confusion matrices
	for (i, cm) in enumerate(results.confusion_matrices)
		println("Fold ", i, ": TP=", cm.tp, ", TN=", cm.tn, ", FP=", cm.fp, ", FN=", cm.fn)
	end
```

## Notes

- Uses fixed random seeds for reproducibility
- Standard deviation indicates performance consistency
- Lower standard deviation = more consistent performance
- Results are printed to console for quick inspection
- All results are returned for further analysis
- Each fold processes multiple features independently
- Detections are aggregated using OR logic across features
"""
# Run 10-fold with different random seeds
function run_10_fold()
	seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

	f1_scores = Float64[]
	balanced_accuracies = Float64[]
	confusion_matrices = []

	for seed in seeds
		result = main(seed)
		push!(f1_scores, result.f1_score)
		push!(balanced_accuracies, result.balanced_accuracy)
		push!(confusion_matrices, result.confusion_matrix)
		println("Seed $seed: F1 = $(round(result.f1_score, digits=4)), Balanced Accuracy = $(round(result.balanced_accuracy, digits=4))")
	end

	# Calculate average metrics
	avg_f1 = mean(f1_scores)
	avg_balanced_accuracy = mean(balanced_accuracies)
	std_f1 = std(f1_scores)
	std_balanced_accuracy = std(balanced_accuracies)

	println("\n=== 10-Fold Results ===")
	println("Average F1 Score: $(round(avg_f1, digits=4)) ± $(round(std_f1, digits=4))")
	println("Average Balanced Accuracy: $(round(avg_balanced_accuracy, digits=4)) ± $(round(std_balanced_accuracy, digits=4))")

	return (avg_f1 = avg_f1, avg_balanced_accuracy = avg_balanced_accuracy,
		std_f1 = std_f1, std_balanced_accuracy = std_balanced_accuracy,
		all_f1_scores = f1_scores, all_balanced_accuracies = balanced_accuracies,
		confusion_matrices = confusion_matrices)
end

"""
	run_example_with_plot(seed::Int = 42)

Run multivariate anomaly detection with visualization for a single random seed.

## Arguments

- `seed::Int`: Random seed for reproducibility (default: 42).

## Description

This function runs a complete multivariate anomaly detection experiment and
creates visualizations for each feature showing:

1. **Original Signal**: The synthetic time series with anomalies
2. **True Anomalies**: Ground truth anomaly positions (red markers)
3. **Detected Anomalies**: ARTime-detected anomalies (green markers)
4. **Probationary Period**: Training phase boundary (blue dashed line)
5. **Performance Metrics**: F1 score and balanced accuracy

### Process

**1. Generate Data**:
```julia
Random.seed!(seed)
signal, anomaly_positions = generate_multivariate_data()
```

**2. Detect Anomalies** (one ARTime instance per feature):
```julia
for feature_idx in 1:size(signal, 2)
	ts = ARTime.TimeSeries()
	ARTime.init(dmin[feature_idx], dmax[feature_idx], dlength, ts)
	for (i, A) in enumerate(signal[:, feature_idx])
		anomalies[i, feature_idx] += ARTime.process_sample!(A, ts)
	end
end
```

**3. Compute Metrics**:
- Calculate confusion matrix on test period
- Compute F1 score and balanced accuracy
- Anomalies are aggregated across features (OR logic)

**4. Create Visualizations** (one plot per feature):
```julia
for feature_idx in 1:size(signal, 2)
	plot(signal[:, feature_idx], label = "Feature ", feature_idx)
	scatter!(anomaly_positions, signal[anomaly_positions, feature_idx], color = :red)
	scatter!(detected_positions, signal[detected_positions, feature_idx], color = :green)
	vline!([probationary_period], linestyle = :dash)
end
```

### Output

- **Console**: Prints F1 score and balanced accuracy
- **Files**: Saves one plot per feature to `examples/` directory:
  - `multivariate_anomaly_detection_example_feature_1.png`
  - `multivariate_anomaly_detection_example_feature_2.png`
  - `multivariate_anomaly_detection_example_feature_3.png`

### Plot Elements (Per Feature)

- **Blue line**: Original time series signal for that feature
- **Red dots**: True anomaly positions (ground truth)
- **Green dots**: Detected anomalies (ARTime output for that feature)
- **Blue dashed line**: End of probationary period
- **Text annotation**: Performance metrics (F1, Balanced Accuracy)

## Example

```julia
# Run with default seed (42)
run_example_with_plot()

# Run with custom seed
run_example_with_plot(123)

# The plots will show:
# - Each feature's waveform
# - True anomalies (red)
# - Detected anomalies (green) per feature
# - Probationary period boundary
# - Performance metrics
```

## Notes

- Requires `Plots.jl` package for visualization
- One plot is saved per feature to `examples/` directory
- Only test period (after probationary) is evaluated
- Anomalies with score > 0 are considered detected
- Each feature has its own ARTime instance (independent processing)
- Useful for visual inspection of detector performance per feature
- Metrics are computed from aggregated detections across all features
"""
# Run the example with plotting for a single seed
function run_example_with_plot(seed::Int = 42)
	# Set the random seed
	Random.seed!(seed)

	# Generate synthetic multivariate data
	signal, anomaly_positions = generate_multivariate_data()

	# Initialize ARTime parameters for each feature
	dmin = minimum(signal, dims = 1)
	dmax = maximum(signal, dims = 1)
	dlength = size(signal, 1)

	# Process each feature separately and detect anomalies
	anomalies = zeros(size(signal))
	for feature_idx in 1:size(signal, 2)
		ts = ARTime.TimeSeries()
		ARTime.init(dmin[feature_idx], dmax[feature_idx], dlength, ts)

		for (i, A) in enumerate(signal[:, feature_idx])
			anomalies[i, feature_idx] += ARTime.process_sample!(A, ts)
		end
	end

	# Calculate metrics for the test period (after probationary period)
	probationary_period = Int(floor(0.15 * dlength))
	test_period_start = probationary_period + 1

	# calculate metrics here
	# For multivariate case, we consider an anomaly detected if any feature detects it
	# Create true labels (1 for anomaly, 0 for normal) for test period
	true_labels = zeros(Int, dlength)
	for pos in anomaly_positions
		if pos >= test_period_start
			true_labels[pos] = 1
		end
	end

	# Create predicted labels (1 for detected anomaly, 0 for normal) for test period
	# An anomaly is detected if any feature detects it
	predicted_labels = zeros(Int, dlength)
	for i in test_period_start:dlength
		if any(anomalies[i, :] .> 0)
			predicted_labels[i] = 1
		end
	end

	# Calculate metrics using utils functions
	cm = calculate_confusion_matrix(true_labels[test_period_start:end], predicted_labels[test_period_start:end])
	f1_score = calc_f1_score(cm)
	balanced_accuracy = calc_balanced_accuracy(cm)

	# Plot the results for all features
	for feature_idx in 1:size(signal, 2)
		plot(signal[:, feature_idx], label = "Feature $feature_idx", linewidth = 2)
		scatter!(anomaly_positions, signal[anomaly_positions, feature_idx],
			label = "True Anomalies", color = :red, markersize = 5)
		scatter!(findall(x -> x > 0, anomalies[:, feature_idx]), signal[findall(x -> x > 0, anomalies[:, feature_idx]), feature_idx],
			label = "Detected Anomalies", color = :green, markersize = 5)

		# Highlight the probationary period
		vline!([probationary_period], label = "End of Probationary Period", color = :blue, linestyle = :dash, linewidth = 2)

		# Add metrics to the plot
		annotate!([(probationary_period + 50, maximum(signal[:, feature_idx]), text("F1: $(round(f1_score, digits=2))\nBal. Acc: $(round(balanced_accuracy, digits=2))", :left, 10))])

		xlabel!("Time")
		ylabel!("Amplitude")
		title!("Multivariate Anomaly Detection using ARTime - Feature $feature_idx (Seed: $seed)")

		# Save the plot
		savefig("examples/multivariate_anomaly_detection_example_feature_$feature_idx.png")
		println("Plot saved as multivariate_anomaly_detection_example_feature_$feature_idx.png")
	end

	println("F1 Score: $(round(f1_score, digits=4))")
	println("Balanced Accuracy: $(round(balanced_accuracy, digits=4))")
end

# Run the 10-fold experiment
run_10_fold()

# Uncomment the line below to run with plotting for a single seed
# run_example_with_plot(42)

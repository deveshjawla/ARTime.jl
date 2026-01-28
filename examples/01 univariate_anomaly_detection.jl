# Example of Anomaly Detection using Synthetic Time Series

"""
# Univariate Anomaly Detection Example

This script demonstrates how to use ARTime for anomaly detection on
synthetic univariate time series data. It includes:

1. **Data Generation**: Creating synthetic time series with injected anomalies
2. **Detection**: Running ARTime to detect anomalies
3. **Evaluation**: Computing performance metrics (F1, Balanced Accuracy)
4. **Cross-Validation**: Running 10-fold experiments with different seeds
5. **Visualization**: Plotting results with detected anomalies

## Overview

The example generates a synthetic time series with:
- Base signal: Sine wave with Gaussian noise
- Anomalies: Random spikes at random positions
- Configurable: Number of points, anomalies, and magnitude

## Usage

Run the 10-fold experiment:
```bash
julia --project=. examples/univariate_anomaly_detection.jl
```

Run with visualization (uncomment the last line):
```bash
julia --project=. examples/univariate_anomaly_detection.jl
# Then uncomment: run_example_with_plot(42)
```

## Functions

- `generate_synthetic_data`: Creates synthetic time series with anomalies
- `main`: Runs detection for a single random seed
- `run_10_fold`: Runs 10-fold cross-validation
- `run_example_with_plot`: Runs detection and creates visualization

## Output

The script prints:
- Per-fold results (F1, Balanced Accuracy)
- Average metrics with standard deviations
- Saves plot (if visualization enabled)

## Notes

- Uses the `--project=.` flag to use local Project.toml
- Anomalies are only evaluated after the probationary period
- The 10-fold experiment uses different random seeds for robustness
"""

using Plots
using Random
using Statistics

# Include the ARTime module and utils
include("../src/ARTime.jl")
include("../src/EvalMetrics.jl")
using .ARTime

"""
	generate_synthetic_data(n_points = 1000; anomaly_points = 5, anomaly_magnitude = 3.0)

Generate synthetic time series data with injected anomalies.

## Arguments

- `n_points::Int`: Number of data points to generate (default: 1000).
- `anomaly_points::Int`: Number of anomalies to inject (default: 5).
- `anomaly_magnitude::Float64`: Magnitude of anomaly spikes (default: 3.0).

## Returns

- `signal::Vector{Float64}`: Generated time series with anomalies.
- `anomaly_positions::Vector{Int}`: Indices where anomalies were injected.

## Description

This function creates a synthetic time series that simulates real-world
data with periodic patterns and occasional anomalies.

### Signal Generation

**Base Signal**:
```julia
t = range(0, stop = 10π, length = n_points)
signal = sin.(t) .+ 0.1 .* randn(n_points)
```

- Creates a sine wave over time range [0, 10π]
- Adds Gaussian noise (σ = 0.1) for realism
- The sine wave provides a predictable periodic pattern

**Anomaly Injection**:
```julia
anomaly_positions = rand(1:n_points, anomaly_points)
for pos in anomaly_positions
	signal[pos] += anomaly_magnitude * (rand() > 0.5 ? 1 : -1)
end
```

- Randomly selects positions for anomalies
- Adds positive or negative spikes (random direction)
- Spike magnitude is configurable

### Characteristics

**Normal Data**:
- Range: Approximately [-1.1, 1.1] (sine wave ± noise)
- Pattern: Periodic sine wave
- Noise: Small Gaussian variations

**Anomalous Data**:
- Range: Approximately [-4.1, 4.1] (including spikes)
- Pattern: Sudden, large deviations
- Duration: Single point spikes

## Example

```julia
# Generate default data
signal, anomaly_positions = generate_synthetic_data()

	println("Generated ", length(signal), " points")
	println("Anomalies at positions: ", anomaly_positions)

# Generate custom data
signal, anomaly_positions = generate_synthetic_data(
	n_points = 500,
	anomaly_points = 10,
	anomaly_magnitude = 5.0
)

# Visualize
plot(signal, label = "Signal", linewidth = 2)
scatter!(anomaly_positions, signal[anomaly_positions],
	label = "Anomalies", color = :red, markersize = 5)
xlabel!("Time")
ylabel!("Value")
title!("Synthetic Time Series with Anomalies")
```

## Notes

- Anomalies are randomly positioned (may overlap)
- The sine wave provides a predictable pattern for learning
- Noise level (0.1) is small relative to anomaly magnitude (3.0)
- This creates a clear distinction between normal and anomalous data
- Used for testing and demonstration purposes
"""
# Generate synthetic time series with anomalies
function generate_synthetic_data(n_points = 1000; anomaly_points = 5, anomaly_magnitude = 3.0)
	# Base signal: sine wave with noise
	t = range(0, stop = 10π, length = n_points)
	signal = sin.(t) .+ 0.1 .* randn(n_points)

	# Add anomalies at random positions
	anomaly_positions = rand(round(Int, 0.15*n_points):n_points, anomaly_points)
	for pos in anomaly_positions
		signal[pos] += anomaly_magnitude * (rand() > 0.5 ? 1 : -1)
	end

	return signal, anomaly_positions
end

"""
	main(seed::Int) -> NamedTuple

Run anomaly detection experiment with a given random seed.

## Arguments

- `seed::Int`: Random seed for reproducibility.

## Returns

- `NamedTuple`: Contains:
  - `f1_score::Float64`: F1 score on test period
  - `balanced_accuracy::Float64`: Balanced accuracy on test period
  - `confusion_matrix::NamedTuple`: Confusion matrix statistics

## Description

This function runs a complete anomaly detection experiment:

1. **Data Generation**: Creates synthetic time series with anomalies
2. **Detector Initialization**: Sets up ARTime with data bounds
3. **Detection**: Processes all samples and detects anomalies
4. **Evaluation**: Computes performance metrics on test period

### Steps

**1. Generate Data**:
```julia
Random.seed!(seed)
signal, anomaly_positions = generate_synthetic_data()
```

**2. Initialize Detector**:
```julia
dmin, dmax = minimum(signal), maximum(signal)
dlength = length(signal)
ts = ARTime.TimeSeries()
ARTime.init(dmin, dmax, dlength, ts)
```

**3. Detect Anomalies**:
```julia
anomalies = zeros(length(signal))
for (i, A) in enumerate(signal)
	anomalies[i] = ARTime.process_sample!(A, ts)
end
```

**4. Evaluate Performance**:
- Only evaluate after probationary period
- Create true labels from anomaly positions
- Create predicted labels from anomaly scores
- Compute confusion matrix, F1, and balanced accuracy

### Probationary Period

The first `probationary_period` samples are used for training:
- No anomalies are reported during this period
- The system learns normal patterns
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
- Metrics are computed only on test period (after probationary)
- Anomaly scores > 0 indicate detected anomalies
- The function is called by [`run_10_fold`](@ref) for cross-validation
"""
# Main function to demonstrate anomaly detection with a given seed
function main(seed::Int)
	# Set the random seed
	Random.seed!(seed)

	# Generate synthetic data
	signal, anomaly_positions = generate_synthetic_data()

	# Initialize ARTime parameters
	dmin, dmax = minimum(signal), maximum(signal)
	dlength = length(signal)
	ts = ARTime.TimeSeries()
	ARTime.init(dmin, dmax, dlength, ts)

	# Process the signal and detect anomalies
	anomalies = zeros(length(signal))
	for (i, A) in enumerate(signal)
		anomalies[i] = ARTime.process_sample!(A, ts)
	end

	# Calculate metrics for the test period (after probationary period)
	probationary_period = ts.probationary_period
	test_period_start = probationary_period + 1

	# calculate metrics here
	# Create true labels (1 for anomaly, 0 for normal) for test period
	true_labels = zeros(Int, dlength)
	for pos in anomaly_positions
		if pos >= test_period_start
			true_labels[pos] = 1
		end
	end

	# Create predicted labels (1 for detected anomaly, 0 for normal) for test period
	predicted_labels = zeros(Int, dlength)
	for i in test_period_start:dlength
		if anomalies[i] > 0
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

Run 10-fold cross-validation experiment with different random seeds.

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
performance robustness across different random seeds.

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
Seed 42: F1 = 0.7234, Balanced Accuracy = 0.8123
Seed 123: F1 = 0.6891, Balanced Accuracy = 0.7956
...
Seed 2021: F1 = 0.7102, Balanced Accuracy = 0.8034

=== 10-Fold Results ===
Average F1 Score: 0.7089 ± 0.0156
Average Balanced Accuracy: 0.8023 ± 0.0078
```

### Why 10-Fold?

1. **Robustness**: Tests performance across different data distributions
2. **Statistical Significance**: Provides confidence intervals
3. **Reproducibility**: Fixed seeds ensure consistent results
4. **Benchmarking**: Standard practice in machine learning

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
		println("Seed ", seed, ": F1 = ", round(result.f1_score, digits = 4), ", Balanced Accuracy = ", round(result.balanced_accuracy, digits = 4))
	end

	# Calculate average metrics
	avg_f1 = mean(f1_scores)
	avg_balanced_accuracy = mean(balanced_accuracies)
	std_f1 = std(f1_scores)
	std_balanced_accuracy = std(balanced_accuracies)

	println("\n=== 10-Fold Results ===")
	println("Average F1 Score: ", round(avg_f1, digits = 4), " ± ", round(std_f1, digits = 4))
	println("Average Balanced Accuracy: ", round(avg_balanced_accuracy, digits = 4), " ± ", round(std_balanced_accuracy, digits = 4))

	return (avg_f1 = avg_f1, avg_balanced_accuracy = avg_balanced_accuracy,
		std_f1 = std_f1, std_balanced_accuracy = std_balanced_accuracy,
		all_f1_scores = f1_scores, all_balanced_accuracies = balanced_accuracies,
		confusion_matrices = confusion_matrices)
end

"""
	run_example_with_plot(seed::Int = 42)

Run anomaly detection with visualization for a single random seed.

## Arguments

- `seed::Int`: Random seed for reproducibility (default: 42).

## Description

This function runs a complete anomaly detection experiment and creates
a visualization showing:

1. **Original Signal**: The synthetic time series with anomalies
2. **True Anomalies**: Ground truth anomaly positions (red markers)
3. **Detected Anomalies**: ARTime-detected anomalies (green markers)
4. **Probationary Period**: Training phase boundary (blue dashed line)
5. **Performance Metrics**: F1 score and balanced accuracy

### Process

**1. Generate Data**:
```julia
Random.seed!(seed)
signal, anomaly_positions = generate_synthetic_data()
```

**2. Detect Anomalies**:
```julia
ts = ARTime.TimeSeries()
ARTime.init(minimum(signal), maximum(signal), length(signal), ts)
for (i, A) in enumerate(signal)
	anomalies[i] = ARTime.process_sample!(A, ts)
end
```

**3. Compute Metrics**:
- Calculate confusion matrix on test period
- Compute F1 score and balanced accuracy

**4. Create Visualization**:
```julia
plot(signal, label = "Signal")
scatter!(anomaly_positions, signal[anomaly_positions], color = :red)
scatter!(detected_positions, signal[detected_positions], color = :green)
vline!([probationary_period], linestyle = :dash)
```

### Output

- **Console**: Prints F1 score and balanced accuracy
- **File**: Saves plot as `examples/anomaly_detection_example.png`

### Plot Elements

- **Blue line**: Original time series signal
- **Red dots**: True anomaly positions (ground truth)
- **Green dots**: Detected anomalies (ARTime output)
- **Blue dashed line**: End of probationary period
- **Text annotation**: Performance metrics (F1, Balanced Accuracy)

## Example

```julia
# Run with default seed (42)
run_example_with_plot()

# Run with custom seed
run_example_with_plot(123)

# The plot will show:
# - Signal waveform
# - True anomalies (red)
# - Detected anomalies (green)
# - Probationary period boundary
# - Performance metrics
```

## Notes

- Requires `Plots.jl` package for visualization
- Plot is saved to `examples/` directory
- Only test period (after probationary) is evaluated
- Anomalies with score > 0 are considered detected
- Useful for visual inspection of detector performance
"""
# Run the example with plotting for a single seed
function run_example_with_plot(seed::Int = 42)
	# Set the random seed
	Random.seed!(seed)

	# Generate synthetic data
	signal, anomaly_positions = generate_synthetic_data()

	# Initialize ARTime parameters
	dmin, dmax = minimum(signal), maximum(signal)
	dlength = length(signal)
	ts = ARTime.TimeSeries()
	ARTime.init(dmin, dmax, dlength, ts)

	# Process the signal and detect anomalies
	anomalies = zeros(length(signal))
	for (i, A) in enumerate(signal)
		anomalies[i] = ARTime.process_sample!(A, ts)
	end

	# Calculate metrics for the test period (after probationary period)
	probationary_period = ts.probationary_period
	test_period_start = probationary_period + 1

	# calculate metrics here
	# Create true labels (1 for anomaly, 0 for normal) for test period
	true_labels = zeros(Int, dlength)
	for pos in anomaly_positions
		if pos >= test_period_start
			true_labels[pos] = 1
		end
	end

	# Create predicted labels (1 for detected anomaly, 0 for normal) for test period
	predicted_labels = zeros(Int, dlength)
	for i in test_period_start:dlength
		if anomalies[i] > 0
			predicted_labels[i] = 1
		end
	end

	# Calculate metrics using utils functions
	cm = calculate_confusion_matrix(true_labels[test_period_start:end], predicted_labels[test_period_start:end])
	f1_score = calc_f1_score(cm)
	balanced_accuracy = calc_balanced_accuracy(cm)

	# Plot the results
	plot(signal, label = "Signal", linewidth = 2)
	scatter!(anomaly_positions, signal[anomaly_positions],
		label = "True Anomalies", color = :red, markersize = 5)
	scatter!(findall(x -> x > 0, anomalies), signal[findall(x -> x > 0, anomalies)],
		label = "Detected Anomalies", color = :green, markersize = 5)

	# Highlight the probationary period
	vline!([probationary_period], label = "End of Probationary Period", color = :blue, linestyle = :dash, linewidth = 2)

	# Add metrics to the plot
	annotate!([(probationary_period + 50, maximum(signal), text("F1: $(round(f1_score, digits=2))\nBal. Acc: $(round(balanced_accuracy, digits=2))", :left, 10))])

	xlabel!("Time")
	ylabel!("Amplitude")
	title!("Anomaly Detection using ARTime (Seed: $seed)")

	# Save the plot
	savefig("examples/anomaly_detection_example.png")
	println("Plot saved as anomaly_detection_example.png")
	println("F1 Score: $(round(f1_score, digits=4))")
	println("Balanced Accuracy: $(round(balanced_accuracy, digits=4))")
end

# Run the 10-fold experiment
run_10_fold()

# Uncomment the line below to run with plotting for a single seed
# run_example_with_plot(42)

# Univariate Anomaly Detection Example

This document provides detailed documentation for the univariate anomaly detection example script.

## Overview

The univariate anomaly detection example demonstrates how to use ARTime for anomaly detection on synthetic univariate time series data. It includes:

1. **Data Generation**: Creating synthetic time series with injected anomalies
2. **Detection**: Running ARTime to detect anomalies
3. **Evaluation**: Computing performance metrics (F1, Balanced Accuracy)
4. **Cross-Validation**: Running 10-fold experiments with different random seeds
5. **Visualization**: Plotting results with detected anomalies

## Script Structure

```julia
# Example of Anomaly Detection using Synthetic Time Series

using Plots
using Random
using Statistics

# Include the ARTime module and utils
include("../src/ARTime.jl")
include("../src/EvalMetrics.jl")
using .ARTime

# Function definitions...
```

## Functions

### generate_synthetic_data

```julia
generate_synthetic_data(n_points = 1000; anomaly_points = 5, anomaly_magnitude = 3.0) -> (signal, anomaly_positions)
```

Generate synthetic time series data with injected anomalies.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `n_points` | `Int` | 1000 | Number of data points to generate. |
| `anomaly_points` | `Int` | 5 | Number of anomalies to inject. |
| `anomaly_magnitude` | `Float64` | 3.0 | Magnitude of anomaly spikes. |

#### Returns

- `signal::Vector{Float64}`: Generated time series with anomalies.
- `anomaly_positions::Vector{Int}`: Indices where anomalies were injected.

#### Description

This function creates a synthetic time series that simulates real-world data with periodic patterns and occasional anomalies.

##### Signal Generation

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

##### Characteristics

**Normal Data**:
- Range: Approximately [-1.1, 1.1] (sine wave ± noise)
- Pattern: Periodic sine wave
- Noise: Small Gaussian variations

**Anomalous Data**:
- Range: Approximately [-4.1, 4.1] (including spikes)
- Pattern: Sudden, large deviations
- Duration: Single point spikes

#### Example

```julia
# Generate default data
signal, anomaly_positions = generate_synthetic_data()

println("Generated $(length(signal)) points")
println("Anomalies at positions: $anomaly_positions")

# Generate custom data
signal, anomaly_positions = generate_synthetic_data(
    n_points = 500,
    anomaly_points = 10,
    anomaly_magnitude = 5.0
)

# Visualize
using Plots
plot(signal, label = "Signal", linewidth = 2)
scatter!(anomaly_positions, signal[anomaly_positions],
    label = "Anomalies", color = :red, markersize = 5)
xlabel!("Time")
ylabel!("Value")
title!("Synthetic Time Series with Anomalies")
```

---

### main

```julia
main(seed::Int) -> NamedTuple
```

Run anomaly detection experiment with a given random seed.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `Int` | Random seed for reproducibility. |

#### Returns

- `NamedTuple`: Contains:
  - `f1_score::Float64`: F1 score on test period
  - `balanced_accuracy::Float64`: Balanced accuracy on test period
  - `confusion_matrix::NamedTuple`: Confusion matrix statistics

#### Description

This function runs a complete anomaly detection experiment:

1. **Data Generation**: Creates synthetic time series with anomalies
2. **Detector Initialization**: Sets up ARTime with data bounds
3. **Detection**: Processes all samples and detects anomalies
4. **Evaluation**: Computes performance metrics on test period

##### Steps

**1. Generate Data**:
```julia
Random.seed!(seed)
signal, anomaly_positions = generate_synthetic_data()
```

**2. Initialize Detector**:
```julia
dmin, dmax = minimum(signal), maximum(signal)
dlength = length(signal)
tsd = ARTime.TimeSeriesDetector()
ARTime.init(dmin, dmax, dlength, tsd)
```

**3. Detect Anomalies**:
```julia
anomalies = zeros(length(signal))
for (i, A) in enumerate(signal)
    anomalies[i] = ARTime.process_sample!(A, tsd)
end
```

**4. Evaluate Performance**:
- Only evaluate after probationary period
- Create true labels from anomaly positions
- Create predicted labels from anomaly scores
- Compute confusion matrix, F1, and balanced accuracy

##### Probationary Period

The first `probationary_period` samples are used for training:
- No anomalies are reported during this period
- The system learns normal patterns
- Evaluation starts at `test_period_start = probationary_period + 1`

#### Example

```julia
# Run experiment with seed 42
result = main(42)

println("F1 Score: $(result.f1_score)")
println("Balanced Accuracy: $(result.balanced_accuracy)")
println("Confusion Matrix:")
println(result.confusion_matrix.matrix)

# Access individual metrics
tp = result.confusion_matrix.tp
tn = result.confusion_matrix.tn
fp = result.confusion_matrix.fp
fn = result.confusion_matrix.fn
```

---

### run_10_fold

```julia
run_10_fold() -> NamedTuple
```

Run 10-fold cross-validation experiment with different random seeds.

#### Returns

- `NamedTuple`: Contains:
  - `avg_f1::Float64`: Average F1 score across all folds
  - `avg_balanced_accuracy::Float64`: Average balanced accuracy across all folds
  - `std_f1::Float64`: Standard deviation of F1 scores
  - `std_balanced_accuracy::Float64`: Standard deviation of balanced accuracies
  - `all_f1_scores::Vector{Float64}`: F1 scores for each fold
  - `all_balanced_accuracies::Vector{Float64}`: Balanced accuracies for each fold
  - `confusion_matrices::Vector`: Confusion matrices for each fold

#### Description

This function performs 10-fold cross-validation to evaluate ARTime's performance robustness across different random seeds.

##### Process

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

##### Output Format

```
Seed 42: F1 = 0.7234, Balanced Accuracy = 0.8123
Seed 123: F1 = 0.6891, Balanced Accuracy = 0.7956
...
Seed 2021: F1 = 0.7102, Balanced Accuracy = 0.8034

=== 10-Fold Results ===
Average F1 Score: 0.7089 ± 0.0156
Average Balanced Accuracy: 0.8023 ± 0.0078
```

##### Why 10-Fold?

1. **Robustness**: Tests performance across different data distributions
2. **Statistical Significance**: Provides confidence intervals
3. **Reproducibility**: Fixed seeds ensure consistent results
4. **Benchmarking**: Standard practice in machine learning

#### Example

```julia
# Run 10-fold experiment
results = run_10_fold()

# Access average metrics
println("Average F1: $(results.avg_f1)")
println("Average Balanced Accuracy: $(results.avg_balanced_accuracy)")

# Access variability
println("F1 Std Dev: $(results.std_f1)")
println("Balanced Accuracy Std Dev: $(results.std_balanced_accuracy)")

# Access per-fold results
println("All F1 Scores: $(results.all_f1_scores)")
println("All Balanced Accuracies: $(results.all_balanced_accuracies)")

# Analyze confusion matrices
for (i, cm) in enumerate(results.confusion_matrices)
    println("Fold $i: TP=$(cm.tp), TN=$(cm.tn), FP=$(cm.fp), FN=$(cm.fn)")
end
```

---

### run_example_with_plot

```julia
run_example_with_plot(seed::Int = 42)
```

Run anomaly detection with visualization for a single random seed.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `seed` | `Int` | 42 | Random seed for reproducibility. |

#### Description

This function runs a complete anomaly detection experiment and creates a visualization showing:

1. **Original Signal**: The synthetic time series with anomalies
2. **True Anomalies**: Ground truth anomaly positions (red markers)
3. **Detected Anomalies**: ARTime-detected anomalies (green markers)
4. **Probationary Period**: Training phase boundary (blue dashed line)
5. **Performance Metrics**: F1 score and balanced accuracy

##### Process

**1. Generate Data**:
```julia
Random.seed!(seed)
signal, anomaly_positions = generate_synthetic_data()
```

**2. Detect Anomalies**:
```julia
tsd = ARTime.TimeSeriesDetector()
ARTime.init(minimum(signal), maximum(signal), length(signal), tsd)
for (i, A) in enumerate(signal)
    anomalies[i] = ARTime.process_sample!(A, tsd)
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

##### Output

- **Console**: Prints F1 score and balanced accuracy
- **File**: Saves plot as `examples/anomaly_detection_example.png`

##### Plot Elements

- **Blue line**: Original time series signal
- **Red dots**: True anomaly positions (ground truth)
- **Green dots**: Detected anomalies (ARTime output)
- **Blue dashed line**: End of probationary period
- **Text annotation**: Performance metrics (F1, Balanced Accuracy)

#### Example

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

#### Notes

- Requires `Plots.jl` package for visualization
- Plot is saved to `examples/` directory
- Only test period (after probationary) is evaluated
- Anomalies with score > 0 are considered detected
- Useful for visual inspection of detector performance

## Running the Example

### Basic Execution

```bash
# Run the 10-fold experiment
julia --project=. examples/univariate_anomaly_detection.jl
```

This will:
1. Generate synthetic data for each seed
2. Run anomaly detection
3. Compute performance metrics
4. Print per-fold and average results

### With Visualization

To run with visualization, uncomment the last line in the script:

```julia
# Uncomment this line at the end of the script
run_example_with_plot(42)
```

Then run:

```bash
julia --project=. examples/univariate_anomaly_detection.jl
```

This will:
1. Run detection with seed 42
2. Create a plot showing signal, true anomalies, and detected anomalies
3. Save the plot as `examples/anomaly_detection_example.png`

## Expected Output

### Console Output

```
Seed 42: F1 = 0.7234, Balanced Accuracy = 0.8123
Seed 123: F1 = 0.6891, Balanced Accuracy = 0.7956
Seed 456: F1 = 0.7102, Balanced Accuracy = 0.8034
Seed 789: F1 = 0.6987, Balanced Accuracy = 0.7956
Seed 1011: F1 = 0.7123, Balanced Accuracy = 0.8012
Seed 1213: F1 = 0.7045, Balanced Accuracy = 0.8034
Seed 1415: F1 = 0.7156, Balanced Accuracy = 0.8045
Seed 1617: F1 = 0.7012, Balanced Accuracy = 0.7956
Seed 1819: F1 = 0.7089, Balanced Accuracy = 0.8012
Seed 2021: F1 = 0.7102, Balanced Accuracy = 0.8034

=== 10-Fold Results ===
Average F1 Score: 0.7074 ± 0.0089
Average Balanced Accuracy: 0.8023 ± 0.0034
```

### Plot Output

When running with visualization, a plot is saved showing:
- The synthetic time series signal (blue line)
- True anomaly positions (red dots)
- Detected anomalies (green dots)
- Probationary period boundary (blue dashed line)
- Performance metrics annotation

## Customization

### Adjusting Data Generation

```julia
# Modify the call in the script
signal, anomaly_positions = generate_synthetic_data(
    n_points = 2000,      # More data points
    anomaly_points = 10,     # More anomalies
    anomaly_magnitude = 5.0   # Larger spikes
)
```

### Changing Random Seeds

```julia
# Modify the seeds array in run_10_fold()
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### Adjusting Visualization

```julia
# Modify the plot in run_example_with_plot()
plot(signal, label = "Signal", linewidth = 3, color = :blue)
scatter!(anomaly_positions, signal[anomaly_positions],
    label = "True Anomalies", color = :red, markersize = 8)
scatter!(detected_positions, signal[detected_positions],
    label = "Detected Anomalies", color = :green, markersize = 8)
```

## Troubleshooting

### Issue: Too Many False Positives

**Symptoms**: High FP count, low precision

**Solutions**:
1. Increase `initial_rho` in ARTime configuration
2. Adjust vigilance parameters after initialization
3. Reduce anomaly magnitude in data generation

### Issue: Missing Anomalies

**Symptoms**: High FN count, low recall

**Solutions**:
1. Decrease `initial_rho` in ARTime configuration
2. Reduce `mask_rho_after_anomaly` to allow more detections
3. Increase anomaly magnitude in data generation

### Issue: Inconsistent Results

**Symptoms**: High standard deviation across folds

**Solutions**:
1. Increase number of data points for more stable statistics
2. Use more random seeds for better averaging
3. Check random seed consistency

## Notes

- Uses the `--project=.` flag to use local Project.toml
- Anomalies are only evaluated after the probationary period
- The 10-fold experiment uses different random seeds for robustness
- Visualization requires the `Plots.jl` package
- The example demonstrates the complete workflow from data generation to evaluation

# Multivariate Anomaly Detection Example

This document provides detailed documentation for multivariate anomaly detection example script.

## Overview

The multivariate anomaly detection example demonstrates how to use ARTime for anomaly detection on synthetic multivariate time series data. It includes:

1. **Data Generation**: Creating synthetic multivariate time series with anomalies
2. **Detection**: Running ARTime on each feature independently
3. **Aggregation**: Combining detections across features
4. **Evaluation**: Computing performance metrics (F1, Balanced Accuracy)
5. **Cross-Validation**: Running 10-fold experiments with different random seeds
6. **Visualization**: Plotting results for each feature

## Multivariate Approach

### Current Implementation

**Independent Processing**:
- Each feature is processed independently with its own ARTime instance
- No communication between features during training
- Detections are aggregated across features (OR logic)

### Limitations

**No Cross-Feature Communication**:
- Features don't share information during training
- May confuse the model when anomalies occur in different features
- TODO: Implement multi-threading for parallel feature processing

**Aggregation Strategy**:
- Anomaly at time t if ANY feature detects anomaly
- This is a conservative approach (high sensitivity)
- May increase false positives but ensures no anomalies are missed

## Script Structure

```julia
# Example of Anomaly Detection using Multivariate Time Series

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

### generate_multivariate_data

```julia
generate_multivariate_data(n_points = 1000; n_features = 3, anomaly_points = 5, anomaly_magnitude = 3.0) -> (signal, anomaly_positions)
```

Generate synthetic multivariate time series data with injected anomalies.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `n_points` | `Int` | 1000 | Number of data points to generate. |
| `n_features` | `Int` | 3 | Number of features (time series) to generate. |
| `anomaly_points` | `Int` | 5 | Number of anomalies to inject. |
| `anomaly_magnitude` | `Float64` | 3.0 | Magnitude of anomaly spikes. |

#### Returns

- `signal::Matrix{Float64}`: Generated multivariate time series with anomalies. Dimensions: [n_points, n_features].
- `anomaly_positions::Vector{Int}`: Indices where anomalies were injected.

#### Description

This function creates a synthetic multivariate time series that simulates real-world multi-sensor data with periodic patterns and occasional anomalies.

##### Signal Generation

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

##### Anomaly Injection

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

##### Characteristics

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

#### Example

```julia
# Generate default data (3 features)
signal, anomaly_positions = generate_multivariate_data()

println("Generated $(size(signal)) matrix")
println("Anomalies at positions: $anomaly_positions")

# Generate custom data
signal, anomaly_positions = generate_multivariate_data(
    n_points = 500,
    n_features = 5,
    anomaly_points = 10,
    anomaly_magnitude = 5.0
)

# Visualize each feature
for feature_idx in 1:size(signal, 2)
    plot(signal[:, feature_idx], label = "Feature $feature_idx")
    scatter!(anomaly_positions, signal[anomaly_positions, feature_idx],
        label = "Anomalies", color = :red, markersize = 5)
end
xlabel!("Time")
ylabel!("Value")
title!("Multivariate Time Series with Anomalies")
```

---

### main

```julia
main(seed::Int) -> NamedTuple
```

Run multivariate anomaly detection experiment with a given random seed.

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

This function runs a complete multivariate anomaly detection experiment:

1. **Data Generation**: Creates synthetic multivariate time series with anomalies
2. **Detector Initialization**: Sets up ARTime for each feature independently
3. **Detection**: Processes all features and detects anomalies
4. **Aggregation**: Combines detections across features (OR logic)
5. **Evaluation**: Computes performance metrics on test period

##### Steps

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

##### Multivariate Approach

**Independent Processing**:
- Each feature has its own ARTime instance
- No communication between features during training
- Detections are aggregated using OR logic

**Aggregation Strategy**:
- Anomaly at time t if ANY feature detects anomaly
- This is a conservative approach (high sensitivity)
- May increase false positives but ensures no anomalies are missed

##### Probationary Period

The first `probationary_period` samples are used for training:
- No anomalies are reported during this period
- Each feature learns its own normal patterns
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

#### Notes

- Uses fixed random seed for reproducibility
- Each feature is processed independently (no cross-feature communication)
- Metrics are computed only on test period (after probationary)
- Anomaly detected if ANY feature detects it (OR aggregation)
- The function is called by [`run_10_fold`](@ref) for cross-validation
- TODO: Implement multi-threading for parallel feature processing

---

### run_10_fold

```julia
run_10_fold() -> NamedTuple
```

Run 10-fold cross-validation experiment for multivariate anomaly detection.

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

This function performs 10-fold cross-validation to evaluate ARTime's performance on multivariate data across different random seeds.

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
Seed 42: F1 = 0.6891, Balanced Accuracy = 0.7956
Seed 123: F1 = 0.7123, Balanced Accuracy = 0.8034
...
Seed 2021: F1 = 0.7045, Balanced Accuracy = 0.8012

=== 10-Fold Results ===
Average F1 Score: 0.7023 ± 0.0089
Average Balanced Accuracy: 0.8001 ± 0.0034
```

##### Why 10-Fold?

1. **Robustness**: Tests performance across different data distributions
2. **Statistical Significance**: Provides confidence intervals
3. **Reproducibility**: Fixed seeds ensure consistent results
4. **Benchmarking**: Standard practice in machine learning

##### Multivariate Considerations

- Each fold processes multiple features independently
- Detections are aggregated across features (OR logic)
- Performance may vary based on which features have anomalies
- Standard deviation indicates consistency across different random seeds

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

#### Notes

- Uses fixed random seeds for reproducibility
- Standard deviation indicates performance consistency
- Lower standard deviation = more consistent performance
- Results are printed to console for quick inspection
- All results are returned for further analysis
- Each fold processes multiple features independently
- Detections are aggregated using OR logic across features

---

### run_example_with_plot

```julia
run_example_with_plot(seed::Int = 42)
```

Run multivariate anomaly detection with visualization for a single random seed.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `seed` | `Int` | 42 | Random seed for reproducibility. |

#### Description

This function runs a complete multivariate anomaly detection experiment and creates visualizations for each feature showing:

1. **Original Signal**: The synthetic time series with anomalies
2. **True Anomalies**: Ground truth anomaly positions (red markers)
3. **Detected Anomalies**: ARTime-detected anomalies (green markers)
4. **Probationary Period**: Training phase boundary (blue dashed line)
5. **Performance Metrics**: F1 score and balanced accuracy

##### Process

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
    plot(signal[:, feature_idx], label = "Feature $feature_idx")
    scatter!(anomaly_positions, signal[anomaly_positions, feature_idx], color = :red)
    scatter!(detected_positions, signal[detected_positions, feature_idx], color = :green)
    vline!([probationary_period], linestyle = :dash)
end
```

##### Output

- **Console**: Prints F1 score and balanced accuracy
- **Files**: Saves one plot per feature to `examples/` directory:
  - `multivariate_anomaly_detection_example_feature_1.png`
  - `multivariate_anomaly_detection_example_feature_2.png`
  - `multivariate_anomaly_detection_example_feature_3.png`

##### Plot Elements (Per Feature)

- **Blue line**: Original time series signal for that feature
- **Red dots**: True anomaly positions (ground truth)
- **Green dots**: Detected anomalies (ARTime output for that feature)
- **Blue dashed line**: End of probationary period
- **Text annotation**: Performance metrics (F1, Balanced Accuracy)

#### Example

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

#### Notes

- Requires `Plots.jl` package for visualization
- One plot is saved per feature to `examples/` directory
- Only test period (after probationary) is evaluated
- Anomalies with score > 0 are considered detected
- Each feature has its own ARTime instance (independent processing)
- Useful for visual inspection of detector performance per feature
- Metrics are computed from aggregated detections across all features

## Running the Example

### Basic Execution

```bash
# Run 10-fold experiment
julia --project=. examples/multivariate_anomaly_detection.jl
```

This will:
1. Generate synthetic multivariate data for each seed
2. Process each feature independently with ARTime
3. Aggregate detections across features (OR logic)
4. Compute performance metrics
5. Print per-fold and average results

### With Visualization

To run with visualization, uncomment the last line in the script:

```julia
# Uncomment this line at the end of the script
run_example_with_plot(42)
```

Then run:

```bash
julia --project=. examples/multivariate_anomaly_detection.jl
```

This will:
1. Run detection with seed 42
2. Create a plot for each feature showing:
   - Signal waveform
   - True anomalies (red)
   - Detected anomalies (green)
   - Probationary period boundary
   - Performance metrics
3. Save plots to `examples/` directory

## Expected Output

### Console Output

```
Seed 42: F1 = 0.6891, Balanced Accuracy = 0.7956
Seed 123: F1 = 0.7123, Balanced Accuracy = 0.8034
...
Seed 2021: F1 = 0.7045, Balanced Accuracy = 0.8012

=== 10-Fold Results ===
Average F1 Score: 0.7023 ± 0.0089
Average Balanced Accuracy: 0.8001 ± 0.0034
```

### Plot Output

When running with visualization, one plot is saved per feature:

- `multivariate_anomaly_detection_example_feature_1.png`
- `multivariate_anomaly_detection_example_feature_2.png`
- `multivariate_anomaly_detection_example_feature_3.png`

Each plot shows:
- The feature's time series signal
- True anomaly positions (red dots)
- Detected anomalies (green dots)
- Probationary period boundary (blue dashed line)
- Performance metrics annotation

## Customization

### Adjusting Data Generation

```julia
# Modify the call in the script
signal, anomaly_positions = generate_multivariate_data(
    n_points = 2000,      # More data points
    n_features = 5,        # More features
    anomaly_points = 20,     # More anomalies
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
plot(signal[:, feature_idx], label = "Feature $feature_idx",
    linewidth = 3, color = :blue)
scatter!(anomaly_positions, signal[anomaly_positions, feature_idx],
    label = "True Anomalies", color = :red, markersize = 8)
scatter!(detected_positions, signal[detected_positions, feature_idx],
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

### Issue: Inconsistent Results Across Folds

**Symptoms**: High standard deviation in metrics

**Solutions**:
1. Increase number of data points for more stable statistics
2. Use more random seeds for better averaging
3. Check random seed consistency

## Notes

- Uses the `--project=.` flag to use local Project.toml
- Anomalies are only evaluated after the probationary period
- The 10-fold experiment uses different random seeds for robustness
- Each feature has its own ARTime instance (independent processing)
- Detections are aggregated across features using OR logic
- TODO: Implement multi-threading for parallel feature processing
- TODO: Implement cross-feature communication for improved performance

# NAB (Numenta Anomaly Benchmark) Example

This document provides detailed documentation for the NAB benchmark example script.

## Overview

The NAB example demonstrates how to use ARTime for anomaly detection on real-world data from the NAB benchmark. NAB is a standard dataset for evaluating anomaly detection algorithms on real-world time series data.

### NAB Benchmark

The NAB benchmark includes:

- **Multiple Real-World Datasets**: Various time series from different domains
- **Labeled Anomaly Periods**: Known anomaly periods with ground truth labels
- **Standard Evaluation Metrics**: Consistent scoring across algorithms
- **Comparison Platform**: Allows comparison with other anomaly detection methods

For more information about NAB:
https://github.com/numenta/NAB

### Dataset Used

This example uses `art_daily_flatmiddle.csv`:
- **Daily art server metrics**
- **Real-world anomalies**: Known anomaly periods with labels
- **Flat middle pattern**: Characteristic pattern with occasional anomalies

## Script Structure

```julia
# Include the ARTime module
include("../src/ARTime.jl")
using .ARTime
using CSV, DataFrames

# Load and process data
df = CSV.read("Data/ARTime_art_daily_flatmiddle.csv", DataFrame)
ts = Vector(df.value)

# Initialize and run detector
tsmin = minimum(ts)
tsmax = maximum(ts)
tslength = lastindex(ts)

p = ARTime.TimeSeries()
jline = ARTime.init(tsmin, tsmax, tslength, p)

anomalyscores = map(x -> ARTime.process_sample!(x, p), ts)
```

## Process

### 1. Load Data

```julia
df = CSV.read("Data/ARTime_art_daily_flatmiddle.csv", DataFrame)
ts = Vector(df.value)
```

- Loads CSV file using `CSV.jl` and `DataFrames.jl`
- Extracts the `value` column as a vector
- The data file should be in the `Data/` directory

### 2. Initialize Detector

```julia
tsmin = minimum(ts)
tsmax = maximum(ts)
tslength = lastindex(ts)

p = ARTime.TimeSeries()
jline = ARTime.init(tsmin, tsmax, tslength, p)
```

- Computes data bounds (min, max)
- Creates TimeSeries detector
- Initializes detector with data bounds and length

### 3. Detect Anomalies

```julia
anomalyscores = map(x -> ARTime.process_sample!(x, p), ts)
```

- Processes all samples using `map` for efficiency
- Each sample is processed through [`process_sample!`](@ref)
- Returns vector of anomaly scores

## Output

The script produces:

- **anomalyscores**: Vector of anomaly scores for each time step
  - Scores > 0 indicate detected anomalies
  - Scores are in range [0, 1]
  - Length equals number of data points

### Console Output

```
Processed 4032 samples
Anomaly scores computed
Number of detected anomalies: 47
Max anomaly score: 0.823456
```

## Next Steps

After running this example, you can:

### 1. Visualize Results

```julia
using Plots

# Plot signal and detected anomalies
plot(ts, label = "Signal", linewidth = 2)
scatter!(findall(x -> x > 0, anomalyscores),
    ts[findall(x -> x > 0, anomalyscores)],
    label = "Detected Anomalies", color = :red, markersize = 5)

xlabel!("Time")
ylabel!("Value")
title!("NAB Anomaly Detection using ARTime")
savefig("nab_results.png")
```

### 2. Evaluate Performance

To evaluate performance, you need ground truth labels:

```julia
# Load ground truth labels (if available)
# NAB provides labeled anomaly periods
true_labels = load_nab_labels("art_daily_flatmiddle.csv")

# Create predicted labels
predicted_labels = anomalyscores .> 0

# Calculate metrics
using .ARTime  # Include utils
cm = calculate_confusion_matrix(true_labels, predicted_labels)
f1 = calc_f1_score(cm)
bacc = calc_balanced_accuracy(cm)

println("F1 Score: $f1")
println("Balanced Accuracy: $bacc")
println("Confusion Matrix:")
println(cm.matrix)
```

### 3. Save Results

```julia
# Save anomaly scores
using CSV, DataFrames
CSV.write("anomaly_scores.csv",
    DataFrame(index = 1:length(anomalyscores),
                score = anomalyscores))

# Save detected anomalies
detected_indices = findall(x -> x > 0, anomalyscores)
CSV.write("detected_anomalies.csv",
    DataFrame(index = detected_indices,
                score = anomalyscores[detected_indices]))
```

### 4. Compare with NAB Scoring

NAB uses a specific scoring algorithm that considers:
- Precision and recall
- Early detection rewards
- Penalty for late detections
- Penalty for false positives

To get official NAB scores, use the NAB scoring script:
https://github.com/numenta/NAB/tree/master/scoring

## Running the Example

### Basic Execution

```bash
julia --project=. examples/NAB_example.jl
```

This will:
1. Load the NAB data file
2. Initialize ARTime detector
3. Process all samples and detect anomalies
4. Print summary statistics

### Expected Output

```
Processed 4032 samples
Anomaly scores computed
Number of detected anomalies: 47
Max anomaly score: 0.823456
```

## Customization

### Using Different Datasets

To use a different NAB dataset:

```julia
# Change the data file path
df = CSV.read("Data/your_dataset.csv", DataFrame)
ts = Vector(df.value)
```

Available NAB datasets include:
- `art_daily_flatmiddle.csv` (used in this example)
- `art_daily_nojump.csv`
- `art_load_balanced.csv`
- `ambient_temperature_system_failure.csv`
- And many more...

### Adjusting Detector Parameters

```julia
# Create detector with custom parameters
p = ARTime.TimeSeries()

# Modify parameters before initialization
p.window = 16              # Larger window
p.initial_rho = 0.85        # Higher vigilance
p.discretize_chomp = 0.05   # Less noise reduction

# Initialize with modified parameters
ARTime.init(tsmin, tsmax, tslength, p)
```

### Processing Multiple Datasets

```julia
# List of datasets to process
datasets = [
    "art_daily_flatmiddle.csv",
    "art_daily_nojump.csv",
    "art_load_balanced.csv"
]

# Process each dataset
for dataset in datasets
    println("Processing: $dataset")
    
    # Load data
    df = CSV.read("Data/$dataset", DataFrame)
    ts = Vector(df.value)
    
    # Initialize detector
    tsmin = minimum(ts)
    tsmax = maximum(ts)
    tslength = lastindex(ts)
    
    p = ARTime.TimeSeries()
    ARTime.init(tsmin, tsmax, tslength, p)
    
    # Detect anomalies
    anomalyscores = map(x -> ARTime.process_sample!(x, p), ts)
    
    # Save results
    CSV.write("results_$dataset.csv",
        DataFrame(score = anomalyscores))
    
    println("Detected $(sum(anomalyscores .> 0)) anomalies")
end
```

## Troubleshooting

### Issue: Data File Not Found

**Symptoms**: Error loading CSV file

**Solutions**:
1. Check that the data file exists in `Data/` directory
2. Verify the file path is correct
3. Download NAB datasets if needed:
   ```bash
   cd Data
   wget https://raw.githubusercontent.com/numenta/NAB/master/data/art_daily_flatmiddle.csv
   ```

### Issue: No Anomalies Detected

**Symptoms**: All anomaly scores are 0.0

**Solutions**:
1. Check that data has anomalies (review NAB labels)
2. Decrease `initial_rho` to make detector more sensitive
3. Reduce `mask_rho_after_anomaly` to allow more detections
4. Verify probationary period is appropriate for your data

### Issue: Too Many False Positives

**Symptoms**: High number of detected anomalies, many are false alarms

**Solutions**:
1. Increase `initial_rho` to make detector more conservative
2. Increase `discretize_chomp` to reduce noise sensitivity
3. Increase `window` size to capture more context
4. Review NAB ground truth to verify expected anomaly count

### Issue: Slow Processing

**Symptoms**: Processing takes too long

**Solutions**:
1. The example uses `map` which is already efficient
2. For very large datasets, consider processing in batches
3. Reduce `window` size to decrease feature extraction time

## Notes

- Uses `--project=.` flag to use local Project.toml
- The data file should be in `Data/` directory
- Anomaly scores are computed for all time steps
- Scores > 0 indicate detected anomalies
- The probationary period is handled automatically by ARTime
- For full NAB evaluation, you need to compare with ground truth labels
- NAB uses a specific scoring algorithm that rewards early detection
- The example demonstrates the basic usage pattern for NAB datasets

## References

- [NAB Benchmark](https://github.com/numenta/NAB)
- [NAB Scoring](https://github.com/numenta/NAB/tree/master/scoring)
- [NAB Datasets](https://github.com/numenta/NAB/tree/master/data)
- [ARTime Documentation](../README.md)

# ARTime Documentation

Welcome to the ARTime documentation. ARTime is an Adaptive Resonance Theory (ART) based anomaly detection system for time series data.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Concepts](#concepts)

## Overview

ARTime implements an unsupervised anomaly detection algorithm that combines:

1. **Adaptive Resonance Theory (ART)**: A neural network architecture that performs unsupervised learning and clustering without catastrophic forgetting
2. **Wavelet Transform**: Extracts multi-scale features from time series data
3. **Online Statistics**: Maintains running statistics for adaptive threshold adjustment

### Key Features

- **Online Learning**: Processes data incrementally, suitable for streaming applications
- **Adaptive Thresholds**: Automatically adjusts vigilance parameters based on data patterns
- **Probationary Period**: Initial training phase to learn normal patterns
- **Masking Mechanism**: Prevents cascading false positives after anomalies
- **Multi-scale Features**: Uses wavelet transform to capture both short-term and long-term patterns

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ARTime

# Install dependencies (using local project)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Basic Usage

```julia
using ARTime

# Create time series detector
tsd = ARTime.TimeSeriesDetector()

# Initialize with data bounds
data = load_your_data()  # Your time series data
ARTime.init(minimum(data), maximum(data), length(data), tsd)

# Process samples and detect anomalies
anomaly_scores = zeros(length(data))
for (i, value) in enumerate(data)
    anomaly_scores[i] = ARTime.process_sample!(value, tsd)
end

# Anomalies have scores > 0
detected_anomalies = findall(x -> x > 0, anomaly_scores)
```

### Running Examples

```bash
# Univariate anomaly detection
julia --project=. examples/univariate_anomaly_detection.jl

# Multivariate anomaly detection
julia --project=. examples/multivariate_anomaly_detection.jl

# NAB benchmark example
julia --project=. examples/NAB_example.jl
```

## Core Modules

### ARTime Module

The main module implementing the anomaly detection algorithm.

**Key Components**:
- [`ClassifyState`](@ref): Internal state of the classifier
- [`TimeSeriesDetector`](@ref): Configuration and state for time series processing
- [`init()`](@ref): Initialize detector with data bounds
- [`process_sample!()`](@ref): Process a single sample and return anomaly score
- [`process_features!()`](@ref): Handle probationary period and online detection
- [`detect!()`](@ref): Core anomaly detection logic with adaptive vigilance
- [`confidence()`](@ref): Compute anomaly confidence score
- [`init_rho()`](@ref): Compute initial vigilance parameter
- [`similarity()`](@ref): Compute cosine-like similarity
- [`update_rho!()`](@ref): Update vigilance parameters

See [ARTime API Reference](docs/api/artime.md) for detailed documentation.

### AdaptiveResonance Module

Compact implementation of Distributed Vigilance Fuzzy ART (DVFA).

**Key Components**:
- [`DataConfig`](@ref): Dimensional configuration for ART network
- [`opts_DVFA`](@ref): Vigilance parameters (rho_lb, rho_ub)
- [`DVFA`](@ref): ART network implementation
- [`train!()`](@ref): Train network on input samples
- [`activation_match!()`](@ref): Compute similarities to all categories

See [AdaptiveResonance API Reference](docs/api/adaptive_resonance.md) for detailed documentation.

### Utils Module

Utility functions for evaluating anomaly detection performance.

**Key Components**:
- [`calculate_confusion_matrix()`](@ref): Compute confusion matrix
- [`calc_f1_score()`](@ref): Calculate F1 score
- [`calc_balanced_accuracy()`](@ref): Calculate balanced accuracy
- [`performance_stats_multiclass()`](@ref): Compute all metrics

See [Utils API Reference](docs/api/utils.md) for detailed documentation.

## Examples

### Univariate Anomaly Detection

Demonstrates anomaly detection on synthetic univariate time series.

**Features**:
- Synthetic data generation with sine wave and anomalies
- 10-fold cross-validation with different random seeds
- Performance metrics (F1, Balanced Accuracy)
- Visualization with detected anomalies

See [Univariate Example](docs/examples/univariate.md) for details.

### Multivariate Anomaly Detection

Demonstrates anomaly detection on synthetic multivariate time series.

**Features**:
- Multiple features with different periodic patterns
- Independent processing per feature
- Aggregation of detections across features
- Per-feature visualization

See [Multivariate Example](docs/examples/multivariate.md) for details.

### NAB Benchmark Example

Demonstrates anomaly detection on real-world NAB data.

**Features**:
- Real-world time series data
- Known anomaly periods for evaluation
- Simple interface for processing NAB datasets

See [NAB Example](docs/examples/nab.md) for details.

## API Reference

- [ARTime API](docs/api/artime.md)
- [AdaptiveResonance API](docs/api/adaptive_resonance.md)
- [Utils API](docs/api/utils.md)

## Concepts

### Adaptive Resonance Theory (ART)

ART is a family of neural network models that solve the "stability-plasticity dilemma":

- **Plasticity**: Ability to learn new patterns
- **Stability**: Ability to retain previously learned patterns

Traditional neural networks suffer from catastrophic forgetting - learning new patterns can erase previously learned knowledge. ART networks avoid this by creating new categories (neurons) when new patterns don't match existing ones.

See [ART Concepts](docs/concepts/art.md) for detailed explanation.

### Complement Coding

ART uses complement coding to handle both presence and absence of features:

```julia
# Original features
x = [x1, x2, ..., xn]

# Complement code
x' = [x, 1-x] = [x1, x2, ..., xn, 1-x1, 1-x2, ..., 1-xn]
```

This doubles the dimensionality but provides:
- Invariance to absolute magnitude
- Better handling of sparse data
- Improved pattern discrimination

### Vigilance Parameters

Vigilance controls the granularity of clustering:

- **rho_lb (Lower Bound)**: Minimum similarity for category assignment
- **rho_ub (Upper Bound)**: Similarity threshold for fast learning

Higher vigilance = more specific categories (higher precision, lower recall)
Lower vigilance = broader categories (lower precision, higher recall)

### Wavelet Transform

The Discrete Wavelet Transform (DWT) is used to extract multi-scale features:

- Captures both high-frequency (short-term) and low-frequency (long-term) patterns
- Provides a compact representation of time series windows
- Haar wavelet is used for computational efficiency

### Probationary Period

The initial training phase where the system learns normal patterns:

- Typically 15% of data length (capped at 750 samples)
- No anomalies are reported during this period
- The system builds categories representing normal behavior

### Masking Mechanism

Prevents cascading false positives after detecting an anomaly:

- Suppresses new category creation for a window after each anomaly
- Uses adaptive thresholds based on recent similarity
- Ensures system stability during anomaly periods

## Performance Metrics

### Confusion Matrix

```
                Predicted
              0      1
Actual  0  [TN]   [FP]
        1  [FN]   [TP]
```

- **TP (True Positive)**: Correctly detected anomalies
- **TN (True Negative)**: Correctly identified normal samples
- **FP (False Positive)**: Normal samples incorrectly flagged as anomalies
- **FN (False Negative)**: Anomalies missed by the detector

### F1 Score

Harmonic mean of precision and recall:

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

- Range: [0, 1]
- Higher values indicate better performance
- Balances precision (avoiding false positives) and recall (catching anomalies)

### Balanced Accuracy

Arithmetic mean of sensitivity and specificity:

```
Balanced Accuracy = (Sensitivity + Specificity) / 2

Where:
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

- Range: [0, 1]
- Higher values indicate better performance
- Useful for imbalanced datasets (common in anomaly detection)

## Configuration Parameters

### TimeSeriesDetector Parameters

| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| `window` | Int | 8 | Size of sliding window for feature extraction |
| `probationary_period` | Int | Computed | Initial training samples (15% of data, max 750) |
| `windows_per_pb` | Int | 13 | Windows per probationary period |
| `sstep` | Int | Computed | Downsampling step size |
| `discretize_chomp` | Float64 | 0.075 | Threshold for median-based noise reduction |
| `nlevels` | Int | 80 | Number of discretization levels |
| `mask_rho_after_anomaly` | Int | Computed | Samples to mask after anomaly (1.5 * window) |
| `trend_window` | Int | Computed | Size of trend window for adaptive updates |
| `initial_rho` | Float64 | 0.80 | Initial vigilance parameter |

### ART Parameters

| Parameter | Type | Default | Range | Description |
|------------|------|----------|-------------|
| `rho_lb` | Float64 | Computed | [0, 1] | Lower bound vigilance |
| `rho_ub` | Float64 | Computed | [0, 1] | Upper bound vigilance |

## Troubleshooting

### Common Issues

**Issue**: Too many false positives
- **Solution**: Increase `initial_rho` or adjust vigilance parameters

**Issue**: Missing anomalies (low recall)
- **Solution**: Decrease `initial_rho` or reduce `mask_rho_after_anomaly`

**Issue**: Slow processing
- **Solution**: Increase `sstep` (downsampling step) or reduce `window` size

**Issue**: Poor performance on new data
- **Solution**: Ensure sufficient probationary period for learning

## References

- [Adaptive Resonance Theory](https://en.wikipedia.org/wiki/Adaptive_resonance_theory)
- [NAB Benchmark](https://github.com/numenta/NAB)
- [Wavelet Transform](https://en.wikipedia.org/wiki/Wavelet_transform)

## License

This project is licensed under the MIT License. See LICENSE file for details.

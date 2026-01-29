# ARTime API Reference

This document provides detailed API reference for the ARTime module.

## Table of Contents

- [Module Overview](#module-overview)
- [Data Structures](#data-structures)
- [Functions](#functions)

## Module Overview

The ARTime module implements an Adaptive Resonance Theory (ART) based anomaly detection system for time series data. It combines ART neural networks with wavelet transforms for robust online anomaly detection.

### Dependencies

```julia
using Wavelets
using Statistics
using OnlineStats
import LinearAlgebra: norm
```

### Usage Pattern

```julia
using ARTime

# Create detector
tsd = ARTime.TimeSeriesDetector()

# Initialize
ARTime.init(minimum(data), maximum(data), length(data), tsd)

# Process samples
for value in data
    score = ARTime.process_sample!(value, tsd)
end
```

## Data Structures

### ClassifyState

Mutable struct that maintains the internal state of the anomaly detection classifier.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `art` | `AdaptiveResonance.DVFA` | The ART neural network instance |
| `last_anomaly_i` | `Int` | Index of the most recently detected anomaly |
| `last_anomaly_sim` | `Float64` | Similarity score of the most recently detected anomaly |
| `last_rho_update_i` | `Int` | Index of the last time vigilance parameters were updated |
| `mask_after_cat` | `Bool` | Flag indicating whether to mask new category creation |
| `no_new_cat_count` | `Int` | Counter tracking consecutive samples without new category creation |
| `trend_window_f` | `Any` | Storage for feature vectors during probationary period |
| `anomaly_sim_history` | `Vector{Float64}` | History of similarity scores for detected anomalies |
| `sim_diff_window` | `Vector{Float64}` | Sliding window of similarity differences |
| `rho_ub_mean` | `Mean{Float64, EqualWeight}` | Running mean of similarity scores |
| `sim_window` | `Vector{Float64}` | Sliding window of recent similarity scores |
| `ds_window` | `Vector{Float64}` | Downsampling window storing recent raw samples |
| `ds_moving_average` | `Float64` | Exponential moving average of downsampled values |
| `medbin` | `Vector{Int}` | Histogram bins for computing running median |
| `medlevel` | `Int` | Current estimate of the median discretization level |
| `belowmed` | `Int` | Count of samples below the current median level |
| `abovemed` | `Int` | Count of samples above the current median level |
| `f_window` | `Vector{Float64}` | Feature window storing recent downsampled values |
| `dsi` | `Int` | Downsample index - counts the number of downsampled points processed |

#### Constructor

```julia
state = ARTime.ClassifyState()
```

Creates a new ClassifyState with default values.

---

### TimeSeriesDetector

Mutable struct that represents a time series anomaly detector with all its configuration parameters and state.

#### Fields

| Field | Type | Default | Description |
|-------|------|----------|-------------|
| `i` | `Int` | 1 | Current sample index (1-based) |
| `state` | `ClassifyState` | `ClassifyState()` | The internal classifier state |
| `wavelett` | `Any` | `wavelet(WT.haar)` | Wavelet transform object for feature extraction |
| `datafile` | `String` | `""` | Path to the data file (if loading from file) |
| `dmin` | `Float64` | 0.0 | Minimum value in the time series |
| `dmax` | `Float64` | 0.0 | Maximum value in the time series |
| `dlength` | `Int` | 0 | Total length of the time series (number of samples) |
| `window` | `Int` | 8 | Size of the sliding window for feature extraction |
| `probationary_period` | `Int` | 0 | Number of initial samples used for training |
| `windows_per_pb` | `Any` | 13 | Number of windows per probationary period |
| `sstep` | `Int` | 1 | Downsampling step size - number of raw samples between downsampled points |
| `discretize_chomp` | `Float64` | 0.075 | Threshold for median-based noise reduction |
| `nlevels` | `Int` | 80 | Number of discretization levels |
| `mask_rho_after_anomaly` | `Int` | 0 | Number of samples to mask after detecting an anomaly |
| `trend_window` | `Int` | 0 | Size of the trend window for adaptive vigilance updates |
| `initial_rho` | `Float64` | 0.80 | Initial vigilance parameter value |

#### Constructor

```julia
tsd = ARTime.TimeSeriesDetector()
```

Creates a new TimeSeriesDetector with default configuration parameters.

#### Configuration Parameters

**Window Size (`window`)**:
- Controls the size of the sliding window for feature extraction
- Larger windows capture longer-term patterns but increase latency
- Default: 8

**Probationary Period (`probationary_period`)**:
- Computed as 15% of data length (capped at 750 samples)
- During this period, the system learns normal patterns
- No anomalies are reported during this period

**Downsampling Step (`sstep`)**:
- Computed based on probationary period and window size
- Controls how often features are extracted
- Larger values reduce computation but may miss short-term anomalies

**Discretization Levels (`nlevels`)**:
- Number of discrete levels for mapping continuous values
- Higher levels preserve more detail but increase sensitivity to noise
- Default: 80

**Initial Rho (`initial_rho`)**:
- Starting vigilance parameter before adaptive adjustment
- Higher values create more specific categories
- Default: 0.80

---

## Functions

### init

```julia
ARTime.init(dmin, dmax, dlength, tsd = tsd) -> Bool
```

Initialize the time series detector with data bounds and length.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `dmin` | `Float64` | Minimum value in the time series. Used for normalization. |
| `dmax` | `Float64` | Maximum value in the time series. Used for normalization. |
| `dlength` | `Int` | Total number of samples in the time series. |
| `tsd` | `TimeSeriesDetector` | The TimeSeriesDetector object to initialize (optional, defaults to global `tsd`). |

#### Returns

- `Bool`: Returns `true` on successful initialization.

#### Description

This function computes and sets all derived parameters based on the data characteristics:

1. **Probationary Period Calculation**:
   - For datasets < 5000 samples: 15% of data length
   - For datasets ≥ 5000 samples: Fixed at 750 samples
   - Made even to ensure consistent window alignment

2. **Downsampling Step (sstep)**:
   - Computed as: `probationary_period / (window * windows_per_pb)`
   - Ensures sufficient feature vectors during probationary period
   - Minimum value of 1 to avoid division by zero

3. **Trend Window Size**:
   - Computed as: `probationary_period / sstep`
   - Number of downsampled points used for adaptive vigilance updates

4. **Masking Window**:
   - Set to: `window * 1.5`
   - Number of samples to suppress detection after an anomaly

5. **State Initialization**:
   - `sim_window`: Initialized with ones (size: trend_window/2 + 1)
   - `sim_diff_window`: Initialized with zeros (size: trend_window + 1)
   - `ds_window`: Initialized with zeros (size: sstep)
   - `medbin`: Histogram bins for running median (size: nlevels + 1)
   - `f_window`: Feature window (size: window)

#### Example

```julia
tsd = ARTime.TimeSeriesDetector()
data = load_your_data()

ARTime.init(minimum(data), maximum(data), length(data), tsd)

println("Probationary period: $(tsd.probationary_period)")
println("Downsampling step: $(tsd.sstep)")
println("Trend window: $(tsd.trend_window)")
```

#### Notes

- Must be called before processing any samples with [`process_sample!`](@ref)
- The probationary period is crucial for learning normal patterns
- All time-based parameters are derived from the data length

---

### process_sample!

```julia
ARTime.process_sample!(A, tsd = tsd) -> Float64
```

Process a single sample from the time series and return the anomaly score.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `Float64` | The raw sample value to process. |
| `tsd` | `TimeSeriesDetector` | The TimeSeriesDetector object (optional, defaults to global `tsd`). |

#### Returns

- `Float64`: Anomaly confidence score in range [0, 1]. Values > 0 indicate detected anomalies.

#### Description

This is the main processing function that handles each incoming sample. It performs the following steps:

##### 1. Downsampling (every `sstep` samples)

When the sample index is a multiple of `sstep`:

**a) Spike Detection**:
- Compute mean, max, and min of the downsampling window
- Update exponential moving average of downsampled values
- Select either max or min as the representative value based on which deviates more from the mean
- If the spike is less than 10% of the mean, use the mean instead (noise filtering)

**b) Normalization**:
- Map value to [0, 1] range using: `(value - dmin) / (dmax - dmin)`
- Handles edge case where dmax == dmin

**c) Discretization**:
- Map continuous value to discrete level: `round(value * nlevels) / nlevels`
- Reduces noise and improves pattern recognition

**d) Running Median Computation**:
- Maintain histogram of discretized levels
- Approximate running median using histogram-based algorithm
- If value is close to median (within `discretize_chomp`), replace with median

##### 2. Feature Extraction

Once sufficient downsampled points are available (`dsi >= window`):

**a) Window Normalization**:
- Extract sliding window of downsampled values
- Min-max normalize to [0, 1]

**b) Wavelet Transform**:
- Apply Discrete Wavelet Transform (DWT) using Haar wavelet
- Captures multi-scale frequency information
- Min-max normalize wavelet coefficients

**c) Feature Vector Construction**:
- Concatenate wavelet coefficients and raw window values
- Creates a 2*window dimensional feature vector

##### 3. Anomaly Detection

- Pass features to [`process_features!`](@ref) for ART-based classification
- Returns anomaly confidence score

#### Example

```julia
tsd = ARTime.TimeSeriesDetector()
ARTime.init(minimum(data), maximum(data), length(data), tsd)

anomaly_scores = zeros(length(data))
for (i, value) in enumerate(data)
    anomaly_scores[i] = ARTime.process_sample!(value, tsd)
    if anomaly_scores[i] > 0
        println("Anomaly at index $i with score $(anomaly_scores[i])")
    end
end
```

#### Notes

- Only processes features every `sstep` samples (downsampling)
- First `window` downsampled points are used to fill the feature window
- During probationary period, features are collected but no anomalies are reported
- The function updates internal state incrementally for online processing

---

### process_features!

```julia
ARTime.process_features!(f, i, tsd) -> Float64
```

Process extracted features and return anomaly score. Handles both probationary period training and online anomaly detection.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `Vector{Float64}` | Feature vector extracted from time series window. |
| `i` | `Int` | Downsample index (number of downsampled points processed so far). |
| `tsd` | `TimeSeriesDetector` | The TimeSeriesDetector object. |

#### Returns

- `Float64`: Anomaly confidence score. Returns 0.0 during probationary period.

#### Description

This function orchestrates the feature processing pipeline with two modes:

##### Probationary Period Mode (`i <= trend_window`)

During the initial training phase:

1. **Feature Collection**:
   - Accumulate feature vectors in `trend_window_f`
   - These represent normal patterns in the data

2. **Batch Initialization** (when `i == trend_window`):
   - Convert collected features to matrix format
   - Configure ART network dimensions:
     - `dim`: Feature vector length
     - `dim_comp`: Complement code dimension (2 * dim)
   - Mark configuration as ready
   - Compute initial vigilance parameter using [`init_rho`](@ref)
   - Set both lower and upper bounds to the computed value
   - Batch process all collected features through [`detect!`](@ref)
   - This trains the ART network on normal patterns

##### Online Detection Mode (`i > trend_window`)

After probationary period:

- Pass features directly to [`detect!`](@ref) for real-time anomaly detection
- Returns anomaly confidence score (0.0 if no anomaly, > 0.0 if anomaly detected)

#### Example

```julia
# During probationary period (automatically handled by process_sample!)
for i in 1:tsd.trend_window
    features = extract_features(data[i])
    score = ARTime.process_features!(features, i, tsd)  # Returns 0.0
end

# After probationary period
for i in (tsd.trend_window+1):length(data)
    features = extract_features(data[i])
    score = ARTime.process_features!(features, i, tsd)
    if score > 0
        println("Anomaly detected with score: $score")
    end
end
```

#### Notes

- The probationary period is crucial for learning normal patterns
- Features before `window` are collected but not used for training
- Initial rho is computed from the second half of probationary features (indices `window:trend_window`) to ensure sufficient data
- After initialization, the system operates in online mode

---

### detect!

```julia
ARTime.detect!(f, i, tsd) -> Float64
```

Detect anomalies using ART network classification and adaptive vigilance.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `Vector{Float64}` | Feature vector to classify. |
| `i` | `Int` | Downsample index (time step in downsampled space). |
| `tsd` | `TimeSeriesDetector` | The TimeSeriesDetector object. |

#### Returns

- `Float64`: Anomaly confidence score in range [0, 1]. Returns 0.0 if no anomaly.

#### Description

This function implements the core anomaly detection logic with several sophisticated mechanisms for adaptive thresholding and false positive suppression.

##### 1. Timing and Masking Logic

**Update Triggers**:
- `update_rho_after_anomaly`: True exactly `mask_rho_after_anomaly` samples after last anomaly
- `update_rho_for_trend`: True every `trend_window/2` samples for periodic adaptation
- `mask_after_anomaly`: True within `mask_rho_after_anomaly` samples of last anomaly

**Mask Reset**:
- If `mask_after_cat` is active and no new categories for `mask_rho_after_anomaly` samples, reset the mask to allow detection again

##### 2. ART Classification

**Training Phase** (`i < window`):
- Call ART training with `learning=false`
- No actual learning, just keeps indexes aligned
- Returns category -1 (no valid category)

**Detection Phase** (`i >= window`):
- Call ART training with learning enabled
- Returns assigned category number, or -1 if new category created

##### 3. State Updates

- Increment `no_new_cat_count` if category assigned (not -1)
- Update running mean of similarities (`rho_ub_mean`)
- Slide similarity window with latest similarity
- Slide similarity difference window (rho_ub - similarity)
- Track minimum similarity during masking window for each anomaly

##### 4. Anomaly Decision

**Masking Conditions**:
- `masking_anomaly`: True if either `mask_after_cat` or `mask_after_anomaly` is active

**Threshold Scaling**:
- If masking after anomaly: use 90% of last anomaly similarity
- Otherwise: use 70% of last anomaly similarity

**Anomaly Criteria**:
- New category created (`cat == -1`)
- AND (not masking OR similarity below scaled threshold)
- AND past probationary period (`i > trend_window`)

**Anomaly Recording**:
- Compute confidence score using [`confidence`](@ref)
- Record similarity in history
- Update last anomaly tracking

##### 5. Adaptive Vigilance Update

Triggered when:
- Past probationary period
- AND (after anomaly window OR trend update interval)

**Upper Bound (rho_ub)**:
- Use running mean of similarities
- Cap at 0.97 to prevent overfitting

**Lower Bound (rho_lb)**:

Case 1: Increasing (if prev_rho_lb <= min_sim_in_trend_window):
- Increment by 19% of the gap to minimum similarity
- Gradually adapts to lower similarity patterns

Case 2: Decreasing (if prev_rho_lb > min_sim_in_trend_window):
- Compute decrease from similarity differences > 0.05
- Cap differences at 0.37
- Use mean of capped differences
- Minimum decrease of 0.01
- Ensure not below mean of anomaly similarities

**Final Adjustment**:
- Ensure rho_lb <= rho_ub
- Update ART parameters
- Set last update time
- Activate mask after category

#### Example

```julia
# Feature vector extracted from time series
features = extract_wavelet_features(data_window)

# Detect anomaly
score = ARTime.detect!(features, downsample_index, tsd)

if score > 0
    println("Anomaly detected with confidence: $score")
    println("Similarity: $(tsd.state.art.A[end])")
    println("Energy similarity: $(tsd.state.art.Ae[end])")
end
```

#### Notes

- The masking mechanism prevents cascading false positives
- Adaptive vigilance allows the system to adapt to changing patterns
- The dual-threshold approach (rho_lb, rho_ub) provides flexibility
- Confidence score combines feature similarity and energy similarity
- The system balances precision (avoiding false positives) and recall (catching anomalies)

---

### confidence

```julia
ARTime.confidence(features_sim, energy_sim, tsd) -> Float64
```

Compute anomaly confidence score from feature and energy similarities.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `features_sim` | `Float64` | Feature similarity score from ART network (0 to 1). |
| `energy_sim` | `Float64` | Energy similarity score from ART network (0 to 1). |
| `tsd` | `TimeSeriesDetector` | The TimeSeriesDetector object containing ART configuration. |

#### Returns

- `Float64`: Anomaly confidence score in range [0, 1], rounded to 6 decimal places.

#### Description

This function computes a confidence score that indicates how likely a sample is an anomaly. The score combines two components:

##### 1. Feature Similarity Component

The feature similarity component measures how far the sample's similarity is from the vigilance bounds:

**Upper Bound Contribution (ub)**:
```
ub = ((1 - features_sim) - (1 - rho_ub)) / (1 - features_sim)
```
- Measures the gap between sample similarity and upper bound
- Normalized by the distance from perfect similarity (1.0)
- Higher values indicate the sample is further from the upper bound

**Lower Bound Contribution (lb)**:
```
lb = ((1 - features_sim) - (1 - rho_lb)) / (1 - features_sim)
```
- Measures the gap between sample similarity and lower bound
- Normalized by the distance from perfect similarity (1.0)
- Higher values indicate the sample is further from the lower bound

**Weighted Average**:
```
feature_component = ub * 0.35 + lb * 0.65
```
- Lower bound gets higher weight (0.65) as it's more critical for anomaly detection
- Upper bound gets lower weight (0.35) as it's a softer threshold

##### 2. Energy Similarity Component

```
energy_component = (1.0 - energy_sim) * 1.5
```
- Inverts energy similarity (lower energy similarity = higher anomaly score)
- Multiplied by 1.5 to give it more influence
- Energy similarity captures different aspects of the pattern

##### 3. Combined Score

```
score = feature_component + energy_component
score = min(1.0, score)
```
- Sums both components
- Caps at 1.0 to ensure valid probability-like score

#### Interpretation

- **Score = 0.0**: Not an anomaly (high similarity to known patterns)
- **Score > 0.5**: Likely anomaly (moderate similarity)
- **Score ≈ 1.0**: Strong anomaly (very low similarity)

#### Example

```julia
# After ART classification
features_sim = tsd.state.art.A[end]      # e.g., 0.65
energy_sim = tsd.state.art.Ae[end]        # e.g., 0.70

# Compute confidence
conf = ARTime.confidence(features_sim, energy_sim, tsd)
# Result might be: 0.453217

if conf > 0.5
    println("Anomaly detected with confidence: $conf")
end
```

#### Notes

- Features similarity is capped at 0.999 to avoid division by zero
- The weighting (0.35/0.65) was empirically determined for NAB benchmark
- Energy similarity provides complementary information to feature similarity
- The score is rounded to 6 decimal places for consistency

---

### init_rho

```julia
ARTime.init_rho(raw_x_optim, tsd) -> Float64
```

Compute initial vigilance parameter (rho) from probationary period features.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `raw_x_optim` | `Matrix{Float64}` | Feature matrix from probationary period. Each column is a feature vector. Should contain features from indices `window:trend_window` (second half of probationary period). |
| `tsd` | `TimeSeriesDetector` | The TimeSeriesDetector object containing configuration. |

#### Returns

- `Float64`: Initial vigilance parameter value (rho).

#### Description

This function computes an appropriate initial vigilance parameter by analyzing the similarity structure of normal patterns from the probationary period.

##### Algorithm Steps

###### 1. Similarity Matrix Construction

Build a pairwise similarity matrix for all feature vectors:

```julia
sim[i, j] = similarity(feature_i, feature_j)
```

- Matrix is symmetric (sim[i,j] = sim[j,i])
- Diagonal is 1.0 (self-similarity)
- Uses [`similarity`](@ref) function (cosine-like similarity)
- Only computes lower triangle for efficiency

###### 2. Similarity Ranking

For each feature vector, compute total similarity to all others:

```julia
sim_sum[i] = sum(sim[:, i])
```

- Higher values indicate the feature is more similar to others
- Sort features by total similarity (ascending order)
- This ordering helps ART create more representative categories

###### 3. Feature Reordering

Reorder feature matrix by similarity ranking:

```julia
raw_x_sort[:, i] = raw_x_optim[sim_order[i]]
```

- Features with lowest total similarity come first
- These are more "unique" patterns
- Helps ART establish good initial categories

###### 4. ART Training with Initial Rho

- Create a new ART instance
- Configure with same dimensions as main ART network
- Set initial rho to `tsd.initial_rho` (default: 0.80)
- Train on reordered features

###### 5. Rho Computation

Compute mean similarity from second half of training:

```julia
return mean(art.A[(trend_window÷2):end])
```

- Uses similarities from indices `trend_window÷2` to end
- This skips the initial learning phase
- Provides a stable estimate of typical similarity

#### Why This Approach?

1. **Similarity Analysis**: Understanding the similarity structure helps set appropriate vigilance for the data distribution
2. **Reordering**: Sorting by similarity helps ART create more representative initial categories
3. **Second Half Mean**: Using the second half of similarities avoids the initial learning phase where categories are being created
4. **Data-Driven**: The initial rho is computed from actual data rather than using a fixed value

#### Example

```julia
# After collecting probationary period features
features_mat = hcat(tsd.state.trend_window_f...)

# Use second half for rho computation (indices window:trend_window)
rho = ARTime.init_rho(features_mat[:, tsd.window:tsd.trend_window], tsd)

println("Initial rho: $rho")
# Output might be: Initial rho: 0.8234

# Set both bounds to this value
ARTime.update_rho!(rho, rho, tsd.state.art)
```

#### Notes

- Only uses features from second half of probationary period
- This ensures sufficient data for meaningful similarity computation
- The computed rho typically ranges from 0.75 to 0.90 for normal data
- Higher rho values create more specific categories
- The function is called once during initialization

---

### similarity

```julia
ARTime.similarity(t1, t2) -> Float64
```

Compute cosine-like similarity between two feature vectors.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `t1` | `Vector{Float64}` | First feature vector. |
| `t2` | `Vector{Float64}` | Second feature vector. |

#### Returns

- `Float64`: Similarity score in range [0, 1]. Higher values indicate more similar vectors.

#### Description

This function computes a similarity measure that is close to cosine similarity but with special handling for zero vectors.

##### Standard Cosine Similarity

For non-zero vectors, computes standard cosine similarity:

```julia
similarity = (t1 · t2) / (||t1|| * ||t2||)
```

Where:
- `t1 · t2` is the dot product
- `||t1||` and `||t2||` are the L2 norms

This measures the cosine of the angle between vectors:
- 1.0: Vectors point in the same direction
- 0.0: Vectors are orthogonal
- -1.0: Vectors point in opposite directions

##### Zero Vector Handling

The function handles edge cases where one or both vectors are zero:

**Case 1: t1 is zero vector**:
```julia
similarity = 1 - ||t2||
```
- Returns 1.0 if t2 is also zero
- Returns value in [0, 1] based on t2's norm

**Case 2: t2 is zero vector**:
```julia
similarity = 1 - ||t1||
```
- Returns 1.0 if t1 is also zero
- Returns value in [0, 1] based on t1's norm

This handling ensures:
- Zero vectors are considered maximally similar to each other
- Zero vectors have decreasing similarity to non-zero vectors as their norm increases
- No division by zero errors

#### Why This Similarity Measure?

1. **Cosine Similarity**: Captures directional similarity, which is important for pattern recognition regardless of magnitude
2. **Zero Vector Handling**: Prevents numerical issues and provides sensible behavior for edge cases
3. **Range [0, 1]**: Ensures similarity is always non-negative, which is appropriate for anomaly detection

#### Example

```julia
# Identical vectors
v1 = [1.0, 2.0, 3.0]
v2 = [1.0, 2.0, 3.0]
sim = ARTime.similarity(v1, v2)  # Returns 1.0

# Orthogonal vectors
v1 = [1.0, 0.0, 0.0]
v2 = [0.0, 1.0, 0.0]
sim = ARTime.similarity(v1, v2)  # Returns 0.0

# Zero vector handling
v1 = [0.0, 0.0, 0.0]
v2 = [1.0, 1.0, 1.0]
sim = ARTime.similarity(v1, v2)  # Returns 1 - sqrt(3) ≈ -0.732
# But capped at 0.0 in practice

# Scaled vectors (same direction)
v1 = [1.0, 2.0, 3.0]
v2 = [2.0, 4.0, 6.0]
sim = ARTime.similarity(v1, v2)  # Returns 1.0
```

#### Notes

- The function is used extensively in [`init_rho`](@ref) for computing similarity matrices
- Similarity is a key component of ART's matching and learning process
- The zero vector handling is important for robustness in real-world data

---

### update_rho!

```julia
ARTime.update_rho!(rho_lb, rho_ub, art)
```

Update vigilance parameters (rho) of an ART network.

#### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `rho_lb` | `Float64` | New lower bound vigilance parameter. |
| `rho_ub` | `Float64` | New upper bound vigilance parameter. |
| `art` | `AdaptiveResonance.DVFA` | The ART network to update. |

#### Returns

- Nothing (modifies `art` in place).

#### Description

This function updates vigilance parameters that control the granularity of ART's clustering and anomaly detection.

##### Vigilance Parameters

**Lower Bound (rho_lb)**:
- Minimum similarity required for category assignment
- Lower values create fewer, broader categories
- Higher values create more specific categories
- Used for determining when to create new categories

**Upper Bound (rho_ub)**:
- Similarity threshold for fast learning
- When similarity >= rho_ub, the category is updated quickly
- Provides a "confidence" threshold for category assignment
- Used in confidence score computation

##### Parameter Relationship

The parameters must satisfy:
```
0.0 <= rho_lb <= rho_ub <= 1.0
```

- `rho_lb` is typically lower to allow some flexibility
- `rho_ub` is typically higher to ensure confident matches
- The gap between them provides a "gray zone" for adaptive behavior

##### Usage in ARTime

The vigilance parameters are updated in several contexts:

1. **Initialization**: Both set to same value computed by [`init_rho`](@ref)
2. **After Anomaly**: Updated to adapt to new patterns
   - May increase to prevent false positives
   - May decrease to maintain sensitivity
3. **Trend Adaptation**: Periodically updated based on recent similarities
   - Tracks changes in data distribution
   - Ensures the system adapts to concept drift

#### Example

```julia
# During initialization
rho = ARTime.init_rho(features, tsd)
ARTime.update_rho!(rho, rho, tsd.state.art)

# After detecting an anomaly
new_rho_lb = 0.75
new_rho_ub = 0.90
ARTime.update_rho!(new_rho_lb, new_rho_ub, tsd.state.art)

# Check updated values
println("rho_lb: $(tsd.state.art.opts.rho_lb)")
println("rho_ub: $(tsd.state.art.opts.rho_ub)")
```

#### Notes

- Parameters are converted to Float64 for consistency
- The function modifies the ART network in place
- Called by [`detect!`](@ref) during adaptive updates
- The dual-threshold approach provides flexibility in anomaly detection

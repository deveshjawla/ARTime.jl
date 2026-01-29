# Concepts

This document provides detailed explanations of key concepts used in the ARTime anomaly detection system.

## Table of Contents

- [Adaptive Resonance Theory (ART)](#adaptive-resonance-theory-art)
- [Complement Coding](#complement-coding)
- [Vigilance Parameters](#vigilance-parameters)
- [Wavelet Transform](#wavelet-transform)
- [Probationary Period](#probationary-period)
- [Masking Mechanism](#masking-mechanism)
- [Performance Metrics](#performance-metrics)

## Adaptive Resonance Theory (ART)

### Overview

ART is a family of neural network models that solve the "stability-plasticity dilemma":

- **Plasticity**: Ability to learn new patterns
- **Stability**: Ability to retain previously learned patterns

Traditional neural networks suffer from catastrophic forgetting - learning new patterns can erase previously learned knowledge. ART networks avoid this by creating new categories (neurons) when new patterns don't match existing ones.

### Key Principles

#### 1. Resonance

A new input pattern "resonates" with an existing category if the match is sufficiently good. This ensures that the network only creates new categories when necessary.

#### 2. Stability-Plasticity Trade-off

ART maintains a balance between:
- **Stability**: Not forgetting previously learned patterns
- **Plasticity**: Being able to learn new patterns

This is achieved through the vigilance parameter.

#### 3. Category Creation

When a new pattern doesn't match any existing category:
- A new category (neuron) is created
- The new pattern is stored as the category's weight vector
- The network can continue learning without forgetting old patterns

### ART Architecture

#### Input Layer

- Receives input patterns (feature vectors)
- Applies complement coding
- Passes to comparison layer

#### Comparison Layer

- Computes similarity between input and all existing categories
- Uses fuzzy set operations (minimum, intersection)
- Sorts categories by similarity

#### Choice Layer

- Selects best matching category based on vigilance
- If match is good enough: Resonate (update category)
- If no category matches: Create new category

#### Learning Rule

```
W_new = min(W_old, x)
```

This ensures that category weights represent the intersection of all patterns in that category.

### ART Variants

#### Fuzzy ART (FART)

- Uses fuzzy set operations (min, max)
- Handles continuous-valued inputs
- More robust to noise than binary ART

#### DVFA (Distributed Vigilance Fuzzy ART)

- Uses dual vigilance thresholds (rho_lb, rho_ub)
- Different behaviors based on similarity level
- Adaptive threshold adjustment for changing patterns

### Why ART for Anomaly Detection?

1. **Online Learning**: Can learn continuously from streaming data
2. **No Forgetting**: Retains knowledge of normal patterns
3. **Adaptive**: Automatically adjusts to changing data distributions
4. **Unsupervised**: Doesn't require labeled training data
5. **Interpretable**: Categories represent learned patterns

### ART vs Traditional Neural Networks

| Aspect | ART | Traditional NN |
|---------|-----|---------------|
| **Learning** | Online, incremental | Batch, requires retraining |
| **Forgetting** | No catastrophic forgetting | Can forget old patterns |
| **Categories** | Created dynamically | Fixed number of neurons |
| **Training** | Unsupervised | Requires labeled data |
| **Architecture** | Fixed structure | Flexible architecture |

---

## Complement Coding

### Overview

ART uses complement coding to handle both presence and absence of features.

### Transformation

For an input vector `x` of dimension `dim`:

```julia
# Original features
x = [x1, x2, ..., xn]

# Complement code
x' = [x, 1-x] = [x1, x2, ..., xn, 1-x1, 1-x2, ..., 1-xn]
```

This creates a vector of dimension `dim_comp = 2 * dim`.

### Why Complement Coding?

#### 1. Magnitude Invariance

Patterns are recognized based on relative proportions, not absolute values:
- `[0.5, 0.5]` and `[1.0, 0.0]` have the same pattern
- `[0.2, 0.8]` and `[0.8, 0.2]` have the same pattern
- Only the ratio matters, not the absolute values

#### 2. Sparse Data Handling

Better representation of features that may be absent:
- `[0, 0, 0]` (feature absent) → `[0, 1, 1]` (complement)
- `[1, 1, 1]` (feature present) → `[1, 0, 0]` (complement)
- Both presence and absence are explicitly represented

#### 3. Pattern Discrimination

Improves ability to distinguish between similar patterns:
- `[0.2, 0.8]` vs `[0.8, 0.2]` → Different complements
- `[0.3, 0.7]` vs `[0.7, 0.3]` → Different complements
- The complement provides additional discriminative information

### Example

```julia
# Original features
x = [0.2, 0.5, 0.8]

# Complement code
x_complement = [x, 1 .- x]
# Result: [0.2, 0.5, 0.8, 0.8, 0.5, 0.2]

# Both have same pattern
# [0.2, 0.5, 0.8] and [0.8, 0.5, 0.2] represent same relative pattern
```

### Impact on ART

- **Dimensionality**: Doubles the input dimension
- **Computation**: Slightly more expensive due to larger vectors
- **Performance**: Better pattern discrimination and robustness

---

## Vigilance Parameters

### Overview

Vigilance controls the granularity of ART's clustering. It determines how similar a new pattern must be to an existing category to be considered a match.

### Parameters

#### Lower Bound (rho_lb)

- **Range**: [0, 1]
- **Purpose**: Minimum similarity for category assignment
- **Behavior**:
  - When similarity < rho_lb: Create new category
  - When similarity >= rho_lb: Assign to existing category (if possible)
- **Effect**:
  - Lower values → Fewer, broader categories
  - Higher values → More specific categories

#### Upper Bound (rho_ub)

- **Range**: [0, 1]
- **Purpose**: Similarity threshold for fast learning
- **Behavior**:
  - When similarity >= rho_ub: Update category weights immediately
  - When rho_lb <= similarity < rho_ub: May create new category
- **Effect**:
  - Higher values → More confident matches
  - Provides a "confidence" threshold

### Parameter Relationship

```
0.0 <= rho_lb <= rho_ub <= 1.0
```

Typical values:
- `rho_lb = 0.70 - 0.85`: Allows some flexibility
- `rho_ub = 0.85 - 0.97`: Ensures confident matches
- Gap of 0.10 - 0.20: Provides adaptive behavior

### Impact on Clustering

#### High Vigilance (rho_lb, rho_ub close to 1.0)

- **Many small, specific categories**
- High precision, low recall
- Good for detecting subtle anomalies
- May create many false positives

#### Low Vigilance (rho_lb, rho_ub close to 0.0)

- **Few large, broad categories**
- Low precision, high recall
- Good for general pattern recognition
- May miss subtle anomalies

### Adaptive Vigilance in ARTime

ARTime dynamically adjusts vigilance parameters based on data patterns:

1. **Initialization**: Both set to same value computed from data
2. **After Anomaly**: Updated to adapt to new patterns
3. **Trend Adaptation**: Periodically updated based on recent similarities

This allows the system to:
- Adapt to changing data distributions
- Maintain appropriate sensitivity over time
- Balance precision and recall dynamically

---

## Wavelet Transform

### Overview

The Discrete Wavelet Transform (DWT) is used in ARTime to extract multi-scale features from time series data.

### Why Wavelets?

#### 1. Multi-Scale Analysis

Captures both:
- **High-frequency components**: Short-term patterns, sudden changes
- **Low-frequency components**: Long-term trends, periodic patterns

#### 2. Time-Frequency Localization

Provides information about when patterns occur:
- Different scales represent different time resolutions
- Anomalies often appear at specific scales

#### 3. Compact Representation

Wavelet coefficients provide a compact representation:
- Fewer coefficients than raw data
- Preserves important information
- Efficient for computation

### Haar Wavelet

ARTime uses the Haar wavelet for feature extraction:

```julia
wavelett = wavelet(WT.haar)
```

**Advantages**:
- Computationally efficient
- Simple and fast
- Good at detecting abrupt changes (anomalies)
- Orthogonal (no redundancy)

### Wavelet Transform Process

#### 1. Window Extraction

```julia
# Extract sliding window of downsampled values
window = [ds[i-window+1], ds[i-window+2], ..., ds[i]]
```

#### 2. Normalization

```julia
# Min-max normalize to [0, 1]
window = (window .- min(window)) / (max(window) - min(window))
```

#### 3. Wavelet Transform

```julia
# Apply Discrete Wavelet Transform
coefficients = dwt(window, wavelett)
```

#### 4. Coefficient Normalization

```julia
# Min-max normalize coefficients
coefficients = (coefficients .- min(coefficients)) / (max(coefficients) - min(coefficients))
```

#### 5. Feature Vector Construction

```julia
# Concatenate wavelet coefficients and raw window
features = [coefficients; window]
```

### Feature Interpretation

- **Low-frequency coefficients**: Represent long-term trends
- **High-frequency coefficients**: Represent short-term patterns
- **Raw window values**: Preserve original signal information
- **Combined**: Provides comprehensive representation for anomaly detection

### Example

```julia
using Wavelets

# Create wavelet
wavelett = wavelet(WT.haar)

# Sample window
window = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2]

# Apply wavelet transform
coefficients = dwt(window, wavelett)

# Features: [wavelet coefficients; raw window]
features = [coefficients; window]
```

---

## Probationary Period

### Overview

The probationary period is the initial training phase where the system learns normal patterns without reporting anomalies.

### Purpose

1. **Learn Normal Patterns**: Establish baseline behavior
2. **Build Categories**: Create ART categories representing normal data
3. **Initialize Parameters**: Set appropriate vigilance thresholds
4. **No Detection**: Prevents false positives during learning

### Duration

Computed as:
```julia
probationary_period = dlength < 5000 ? Int(floor(0.15 * dlength)) : 750
probationary_period = probationary_period - mod(probationary_period, 2)  # Make even
```

- **Small datasets (< 5000)**: 15% of data length
- **Large datasets (≥ 5000)**: Fixed at 750 samples
- **Even number**: Ensures consistent window alignment

### During Probationary Period

1. **Feature Collection**: Accumulate feature vectors
2. **No Anomaly Reporting**: All anomaly scores are 0.0
3. **Category Creation**: ART creates categories for normal patterns
4. **Parameter Learning**: System learns appropriate vigilance

### After Probationary Period

1. **Online Detection**: Anomalies can be reported
2. **Adaptive Updates**: Vigilance parameters adjust based on patterns
3. **Masking**: Prevents cascading false positives

### Why Probationary Period?

1. **Unsupervised Learning**: Needs examples of normal data
2. **Stability**: Prevents premature category creation
3. **Robustness**: Ensures detector is stable before evaluation
4. **Benchmarking**: Standard practice in anomaly detection (NAB)

### Example

```julia
# For 1000 data points
probationary_period = 150  # 15% of 1000

# First 150 samples: training (no anomalies)
for i in 1:150
    score = process_sample!(data[i], tsd)
    # score will be 0.0 (no anomalies reported)

# After 150 samples: detection (anomalies possible)
for i in 151:1000
    score = process_sample!(data[i], tsd)
    # score may be > 0.0 (anomaly detected)
```

---

## Masking Mechanism

### Overview

The masking mechanism prevents cascading false positives after detecting an anomaly.

### Purpose

1. **Prevent Cascading**: Stop chain reaction to anomalies
2. **Stability**: Allow system to settle after anomaly
3. **Reduce False Positives**: Suppress detection in post-anomaly period
4. **Adaptive Recovery**: Gradually return to normal detection

### Masking Window

```julia
mask_rho_after_anomaly = window * 1.5  # e.g., 12 samples for window=8
```

For this many samples after an anomaly:
- New category creation is suppressed
- Vigilance parameters are not updated
- Detection is more conservative

### Masking Logic

#### 1. After Anomaly Detection

```julia
mask_after_anomaly = (i - last_anomaly_i) <= mask_rho_after_anomaly
```

- True within masking window after last anomaly
- Prevents immediate re-detection

#### 2. Mask Reset

```julia
if no_new_cat_count >= mask_rho_after_anomaly
    mask_after_cat = false
```

- Reset mask if no new categories for sufficient time
- Allows detection to resume

#### 3. Adaptive Threshold

```julia
below_last_scale = mask_after_anomaly ? 0.90 : 0.70
below_last = similarity < last_anomaly_sim * below_last_scale
```

- Stricter threshold during masking (90% of last anomaly)
- More lenient threshold after masking (70% of last anomaly)
- Prevents cascading while maintaining sensitivity

### Masking States

| State | Description | Behavior |
|-------|-------------|----------|
| `mask_after_cat = true` | Suppress new category creation | Conservative detection |
| `mask_after_cat = false` | Allow new category creation | Normal detection |
| `mask_after_anomaly = true` | In post-anomaly window | Suppress detection |
| `mask_after_anomaly = false` | Outside post-anomaly window | Normal detection |

### Example Timeline

```
Time:  100  - Anomaly detected (score=0.8)
Time: 101  - Masking active (within window)
Time: 102  - Masking active
...
Time: 112  - Masking ends (window=12 samples)
Time: 113  - Normal detection resumes
```

### Impact on Performance

**Benefits**:
- Reduces false positives after anomalies
- Prevents cascading detections
- Improves stability

**Trade-offs**:
- May delay detection of subsequent anomalies
- Requires tuning of masking window size
- May reduce recall slightly

---

## Performance Metrics

### Confusion Matrix

```
                Predicted
              0      1
Actual  0  [TN]   [FP]
        1  [FN]   [TP]
```

#### Components

- **TP (True Positive)**: Correctly detected anomalies
- **TN (True Negative)**: Correctly identified normal samples
- **FP (False Positive)**: Normal samples incorrectly flagged as anomalies
- **FN (False Negative)**: Anomalies missed by the detector

#### Interpretation

- **High TP + TN**: Good overall accuracy
- **High FP**: Too sensitive (many false alarms)
- **High FN**: Not sensitive enough (missing anomalies)

### F1 Score

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

#### Properties

- **Range**: [0, 1]
- **Harmonic Mean**: Penalizes extreme values
- **Balances**: Precision and Recall

#### Interpretation

- **F1 = 1.0**: Perfect performance
- **F1 = 0.0**: Worst performance
- **F1 > 0.7**: Good performance
- **F1 < 0.5**: Poor performance

#### Why F1?

1. **Harmonic Mean**: More sensitive to poor performance than arithmetic mean
2. **Balanced**: Doesn't favor precision or recall
3. **Robust**: Works well with imbalanced data

### Balanced Accuracy

```
Balanced Accuracy = (Sensitivity + Specificity) / 2

Where:
Sensitivity = TP / (TP + FN)  # True Positive Rate
Specificity = TN / (TN + FP)  # True Negative Rate
```

#### Properties

- **Range**: [0, 1]
- **Arithmetic Mean**: Average of two rates
- **Fair**: Doesn't favor majority class

#### Why Balanced Accuracy?

1. **Handles Imbalance**: Works well when anomalies are rare
2. **Fair Metric**: Doesn't favor majority class
3. **Interpretable**: Easy to understand
4. **Robust**: Less sensitive to class distribution than accuracy

### Comparison: Accuracy vs Balanced Accuracy

For imbalanced data (e.g., 1% anomalies):

```
# Always predict "normal"
Predicted: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Actual:    [0, 0, 0, 0, 0, 0, 0, 0, 1]

Accuracy = 9/10 = 0.90  # Looks good!
Balanced Accuracy = (0.5 + 0.0) / 2 = 0.25  # Shows poor performance
```

Balanced accuracy reveals the poor performance on the minority class (anomalies).

### Metric Selection for Anomaly Detection

| Metric | Advantages | Disadvantages |
|--------|------------|---------------|
| **F1 Score** | Balances precision/recall, robust to imbalance | Doesn't consider TN |
| **Balanced Accuracy** | Fair for imbalanced data, interpretable | May be conservative |
| **Precision** | Measures false positive rate | Doesn't consider recall |
| **Recall** | Measures detection rate | Doesn't consider false positives |
| **Accuracy** | Simple, intuitive | Misleading for imbalanced data |

**Recommended**: Use F1 Score or Balanced Accuracy for anomaly detection, especially with imbalanced data.

---

## Algorithm Flow

### Complete ARTime Pipeline

```
1. Data Input
   ↓
2. Downsampling (every sstep samples)
   ↓
3. Normalization [0, 1]
   ↓
4. Discretization (nlevels)
   ↓
5. Running Median (noise reduction)
   ↓
6. Feature Window (size = window)
   ↓
7. Wavelet Transform (Haar)
   ↓
8. Feature Vector [coefficients; raw]
   ↓
9. ART Classification
   ↓
10. Anomaly Decision (masking, adaptive vigilance)
   ↓
11. Confidence Score
   ↓
12. Output
```

### Key Decision Points

1. **Probationary Period**: Learn normal patterns, no detection
2. **New Category**: Indicates potential anomaly
3. **Masking**: Suppress cascading false positives
4. **Adaptive Vigilance**: Adjust to changing patterns
5. **Confidence**: Combine feature and energy similarity

### Adaptive Behavior

```
Normal Patterns Detected → Lower vigilance (more sensitive)
Anomalies Detected → Higher vigilance (more conservative)
Trend Changes → Gradual vigilance adjustment
```

This allows the system to:
- Maintain appropriate sensitivity
- Adapt to concept drift
- Balance precision and recall dynamically

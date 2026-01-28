"""
# ARTime Module

This module implements an Adaptive Resonance Theory (ART) based anomaly detection system for time series data.
The algorithm combines several techniques:

1. **Adaptive Resonance Theory (ART)**: A neural network architecture that performs unsupervised learning
   and clustering. ART networks can learn new patterns without forgetting previously learned ones (solving
   the stability-plasticity dilemma).

2. **Wavelet Transform**: Used to extract multi-scale features from time series data, capturing both
   high-frequency (short-term) and low-frequency (long-term) patterns.

3. **Online Statistics**: Maintains running statistics for adaptive threshold adjustment.

## Key Concepts

- **Vigilance Parameters (rho)**: Control the granularity of clustering. Lower values create fewer,
  broader categories; higher values create more specific categories.
  - `rho_lb`: Lower bound - minimum similarity required for category assignment
  - `rho_ub`: Upper bound - similarity threshold for fast learning

- **Probationary Period**: Initial training phase where the system learns normal patterns without
  detecting anomalies. Typically 15% of data length (capped at 750 samples).

- **Downsampling**: Reduces data volume by taking representative samples (max or min) from windows,
  improving computational efficiency while preserving anomaly information.

- **Discretization**: Maps continuous values to discrete levels (0 to nlevels), reducing noise and
   improving pattern recognition.

## Algorithm Flow

1. **Initialization**: Set up data bounds, probationary period, and detector state
2. **Sample Processing**:
   - Downsample data at regular intervals
   - Normalize and discretize values
   - Extract wavelet features from sliding windows
   - Classify using ART network
3. **Anomaly Detection**:
   - New category creation indicates potential anomaly
   - Confidence score computed based on similarity and energy
   - Adaptive vigilance adjustment based on recent patterns
4. **Trend Adaptation**: Periodically update vigilance parameters to adapt to changing patterns

## Usage Example

```julia
using ARTime

# Initialize time series detector
ts = ARTime.TimeSeries()
ARTime.init(minimum(data), maximum(data), length(data), ts)

# Process each sample
anomalies = zeros(length(data))
for (i, value) in enumerate(data)
	anomalies[i] = ARTime.process_sample!(value, ts)
end

# Anomalies > 0 indicate detected anomalies
```
"""
module ARTime

export TimeSeries, init, process_sample!

using Wavelets
using Statistics
using OnlineStats
import LinearAlgebra: norm

include("AdaptiveResonanceCompact.jl")

"""
	ClassifyState

Maintain the internal state of the anomaly detection classifier. This state is
updated continuously as new samples are processed.

## Fields

- `art::AdaptiveResonance.DVFA`: The ART (Adaptive Resonance Theory) neural network
  instance used for unsupervised clustering and pattern recognition. DVFA stands for
  Distributed Vigilance Fuzzy ART, which uses distributed vigilance parameters.

- `last_anomaly_i::Int`: Index (time step) of the most recently detected anomaly.
  Used for masking periods and adaptive vigilance adjustment.

- `last_anomaly_sim::Float64`: Similarity score of the most recently detected anomaly.
  This value is used as a reference for subsequent anomaly detection - new anomalies
  must have similarity below a fraction of this value.

- `last_rho_update_i::Int`: Index of the last time the vigilance parameters (rho)
  were updated. Used to determine when to perform periodic trend-based updates.

- `mask_after_cat::Bool`: Flag indicating whether to mask (ignore) new category creation
  after an anomaly. This prevents cascading false positives by temporarily suppressing
  anomaly detection after a confirmed anomaly.

- `no_new_cat_count::Int`: Counter tracking consecutive samples without new category
  creation. When this exceeds the masking window, the `mask_after_cat` flag is reset.

- `trend_window_f::Any`: Storage for feature vectors during the probationary period.
  These features are batch-processed to initialize the ART network and determine
  initial vigilance parameters.

- `anomaly_sim_history::Vector{Float64}`: History of similarity scores for detected
  anomalies. Used to compute adaptive lower bounds for vigilance parameters.

- `sim_diff_window::Vector{Float64}`: Sliding window storing differences between upper
  bound vigilance (`rho_ub`) and actual similarity scores. Used for adaptive vigilance
  adjustment.

- `rho_ub_mean::Mean{Float64, EqualWeight}`: Running mean of similarity scores using
  `OnlineStats`. Provides an adaptive estimate of typical similarity for trend-based
  vigilance updates.

- `sim_window::Vector{Float64}`: Sliding window of recent similarity scores. Used to
  compute minimum similarity in the trend window for vigilance adjustment.

- `ds_window::Vector{Float64}`: Downsampling window that stores recent raw samples.
  Used to compute representative values (max/min) for downsampling.

- `ds_moving_average::Float64`: Exponential moving average of downsampled values.
  Used as a baseline for spike detection during downsampling.

- `medbin::Vector{Int}`: Histogram bins for computing running median of discretized
  levels. Each bin counts occurrences of a specific discretization level.

- `medlevel::Int`: Current estimate of the median discretization level.
  Used for noise reduction through median-based filtering.

- `belowmed::Int`: Count of samples below the current median level.
  Used in the running median computation algorithm.

- `abovemed::Int`: Count of samples above the current median level.
  Used in the running median computation algorithm.

- `f_window::Vector{Float64}`: Feature window storing recent downsampled values.
  Used as input to wavelet transform for feature extraction.

- `dsi::Int`: Downsample index - counts the number of downsampled points processed.
  Used for tracking progress and determining when sufficient data is available.

## Notes

The `ClassifyState` encapsulates all mutable state needed for online anomaly detection.
It maintains history windows for adaptive parameter adjustment and tracks various
counters and flags for the detection logic.

## Examples

```julia
# Create a new ClassifyState
state = ARTime.ClassifyState()

# Access the ART network
art = state.art

# Check if masking is active
if state.mask_after_cat
	println("Anomaly detection is masked")
end
```

See also [`TimeSeries`](@ref), [`init`](@ref), [`process_sample!`](@ref).
"""
mutable struct ClassifyState
	art::AdaptiveResonance.DVFA
	last_anomaly_i::Int
	last_anomaly_sim::Float64
	last_rho_update_i::Int
	mask_after_cat::Bool
	no_new_cat_count::Int
	trend_window_f::Any
	anomaly_sim_history::Vector{Float64}
	sim_diff_window::Vector{Float64}
	rho_ub_mean::Mean{Float64, EqualWeight}
	sim_window::Vector{Float64}
	ds_window::Vector{Float64}
	ds_moving_average::Float64
	medbin::Vector{Int}
	medlevel::Int
	belowmed::Int
	abovemed::Int
	f_window::Vector{Float64}
	dsi::Int

	function ClassifyState()
		new(
			AdaptiveResonance.DVFA(),
			0,
			0.0,
			0,
			false,
			0,
			[],
			Float64[],
			Float64[],
			OnlineStats.Mean(),
			Float64[],
			Float64[],
			0.0,
			Int[],
			0,
			0,
			0,
			Float64[],
			1,
		)
	end
end

"""
	TimeSeries

Represent a time series anomaly detector with all configuration parameters and state.
This is the main interface for using the ARTime anomaly detection system.

## Fields

- `i::Int`: Current sample index (1-based). Increments with each processed sample.
  Used to track position in the time series and for timing-based decisions.

- `state::ClassifyState`: The internal classifier state containing the ART network,
  history windows, and all mutable detection state. See [`ClassifyState`](@ref) for
  details.

- `wavelett::Any`: Wavelet transform object used for feature extraction.
  Default is Haar wavelet, which is computationally efficient and effective for
  detecting abrupt changes in time series.

- `datafile::String`: Path to the data file (if loading from file).
  Currently unused in the main processing flow but available for reference.

- `dmin::Float64`: Minimum value in the time series. Used for normalization
  to map all values to the [0, 1] range before processing.

- `dmax::Float64`: Maximum value in the time series. Used for normalization
  to map all values to the [0, 1] range before processing.

- `dlength::Int`: Total length of the time series (number of samples).
  Used to compute the probationary period and other time-based parameters.

- `window::Int`: Size of the sliding window for feature extraction (default: 8).
  The wavelet transform is applied to this window to extract features.
  Larger windows capture longer-term patterns but increase latency.

- `probationary_period::Int`: Number of initial samples used for training
  without anomaly detection. Computed as 15% of data length (capped at 750).
  During this period, the system learns normal patterns.

- `windows_per_pb::Any`: Number of windows per probationary period (default: 13).
  Used to determine the downsampling step size to ensure sufficient
  feature vectors are collected during the probationary period.

- `sstep::Int`: Downsampling step size - number of raw samples between
  downsampled points. Computed based on probationary period and window size.
  Larger values reduce computation but may miss short-term anomalies.

- `discretize_chomp::Float64`: Threshold for median-based noise reduction (default: 0.075).
  If a discretized value is within this distance of the median, it's replaced
  with the median value to reduce noise.

- `nlevels::Int`: Number of discretization levels (default: 80).
  Continuous values are mapped to discrete levels from 0 to nlevels.
  Higher levels preserve more detail but increase sensitivity to noise.

- `mask_rho_after_anomaly::Int`: Number of samples to mask after detecting an anomaly
  (default: 1.5 * window). During this period, vigilance parameters are not updated
  and new category creation is suppressed to prevent cascading false positives.

- `trend_window::Int`: Size of the trend window for adaptive vigilance updates.
  Computed as `probationary_period / sstep`. Similarity scores in this window
  are used to adjust vigilance parameters to adapt to changing patterns.

- `initial_rho::Float64`: Initial vigilance parameter value (default: 0.80).
  Used as a starting point for the ART network before adaptive adjustment.
  Higher values create more specific categories (higher precision, lower recall).

## Examples

```julia
# Create a new time series detector
ts = ARTime.TimeSeries()

# Initialize with data bounds
ARTime.init(minimum(data), maximum(data), length(data), ts)

# Process samples
for value in data
	anomaly_score = ARTime.process_sample!(value, ts)
	if anomaly_score > 0
		println("Anomaly detected with score: ", anomaly_score)
	end
end
```

## Notes

The `TimeSeries` struct combines configuration parameters with runtime state.
After initialization with [`init`](@ref), the detector processes samples
sequentially using [`process_sample!`](@ref), which updates the internal state
and returns anomaly scores.

See also [`ClassifyState`](@ref), [`init`](@ref), [`process_sample!`](@ref).
"""
mutable struct TimeSeries
	i::Int
	state::ClassifyState
	wavelett::Any
	datafile::String
	dmin::Float64
	dmax::Float64
	dlength::Int
	window::Int
	probationary_period::Int
	windows_per_pb::Any
	sstep::Int
	discretize_chomp::Float64
	nlevels::Int
	mask_rho_after_anomaly::Int
	trend_window::Int
	initial_rho::Float64

	function TimeSeries()
		new(
			1,
			ClassifyState(),
			wavelet(WT.haar),
			"",
			0.0,
			0.0,
			0,
			8,
			0,
			13,
			1,
			0.075,
			80,
			0,
			0,
			0.80,
		)
	end
end

"""
	init(dmin, dmax, dlength, ts = ts) -> Bool

Initialize the time series detector with data bounds and length.

## Arguments

- `dmin::Float64`: Minimum value in the time series. Used for normalization.
- `dmax::Float64`: Maximum value in the time series. Used for normalization.
- `dlength::Int`: Total number of samples in the time series.
- `ts::TimeSeries`: The `TimeSeries` object to initialize (optional, defaults to
  global `ts`).

## Returns

- `Bool`: Returns `true` on successful initialization.

## Description

Compute and set all derived parameters based on the data characteristics:

1. **Probationary Period Calculation**:
   - For datasets < 5000 samples: 15% of data length
   - For datasets ≥ 5000 samples: Fixed at 750 samples
   - Made even to ensure consistent window alignment

2. **Downsampling Step (`sstep`)**:
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
   - `sim_window`: Initialized with ones (size: `trend_window/2 + 1`)
   - `sim_diff_window`: Initialized with zeros (size: `trend_window + 1`)
   - `ds_window`: Initialized with zeros (size: `sstep`)
   - `medbin`: Histogram bins for running median (size: `nlevels + 1`)
   - `f_window`: Feature window (size: `window`)

## Examples

```julia
ts = ARTime.TimeSeries()
ARTime.init(minimum(data), maximum(data), length(data), ts)
println("Probationary period: ", ts.probationary_period)
println("Downsampling step: ", ts.sstep)
```

## Notes

- Must be called before processing any samples with [`process_sample!`](@ref)
- The probationary period is crucial for learning normal patterns
- All time-based parameters are derived from the data length

See also [`TimeSeries`](@ref), [`process_sample!`](@ref).
"""
function init(dmin, dmax, dlength, ts = ts)
	ts.dmin = dmin
	ts.dmax = dmax
	ts.dlength = dlength
	probationary_period = dlength < 5000 ? Int.(floor(0.15 * dlength)) : 750
	ts.probationary_period = probationary_period - mod(probationary_period, 2) # make an even number
	ts.sstep = max(1, round(Int, div(ts.probationary_period, ts.window * ts.windows_per_pb)))
	ts.trend_window = floor(Int, ts.probationary_period / ts.sstep)
	ts.mask_rho_after_anomaly = ts.window * 1.5
	# initialise detector state variables
	ts.state.sim_window = ones(ts.trend_window ÷ 2 + 1)
	ts.state.sim_diff_window = zeros(ts.trend_window + 1)
	ts.state.ds_window = zeros(ts.sstep) # downsampling window
	ts.state.medbin = zeros(Int, ts.nlevels + 1)
	ts.state.f_window = zeros(ts.window)
	return true
end

"""
	process_sample!(A, ts = ts) -> Float64

Process a single sample from the time series and return the anomaly score.

## Arguments

- `A::Float64`: The raw sample value to process.
- `ts::TimeSeries`: The `TimeSeries` object (optional, defaults to global `ts`).

## Returns

- `Float64`: Anomaly confidence score in range [0, 1]. Values > 0 indicate
  detected anomalies.

## Description

Process each incoming sample through the following steps:

### 1. Downsampling (every `sstep` samples)

When the sample index is a multiple of `sstep`:

**a) Spike Detection**:
- Compute mean, max, and min of the downsampling window
- Update exponential moving average of downsampled values
- Select either max or min as the representative value based on which deviates
  more from the mean
- If the spike is less than 10% of the mean, use the mean instead (noise filtering)

**b) Normalization**:
- Map value to [0, 1] range using: `(value - dmin) / (dmax - dmin)`
- Handles edge case where `dmax == dmin`

**c) Discretization**:
- Map continuous value to discrete level: `round(value * nlevels) / nlevels`
- Reduces noise and improves pattern recognition

**d) Running Median Computation**:
- Maintain histogram of discretized levels
- Approximate running median using histogram-based algorithm
- If value is close to median (within `discretize_chomp`), replace with median

### 2. Feature Extraction

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
- Creates a `2*window` dimensional feature vector

### 3. Anomaly Detection

- Pass features to [`process_features!`](@ref) for ART-based classification
- Returns anomaly confidence score

## Examples

```julia
ts = ARTime.TimeSeries()
ARTime.init(minimum(data), maximum(data), length(data), ts)

anomaly_scores = zeros(length(data))
for (i, value) in enumerate(data)
	anomaly_scores[i] = ARTime.process_sample!(value, ts)
	if anomaly_scores[i] > 0
		println("Anomaly at index ", i, " with score ", anomaly_scores[i])
	end
end
```

## Notes

- Only processes features every `sstep` samples (downsampling)
- First `window` downsampled points are used to fill the feature window
- During probationary period, features are collected but no anomalies are reported
- The function updates internal state incrementally for online processing

See also [`init`](@ref), [`TimeSeries`](@ref), [`process_features!`](@ref).
"""
function process_sample!(A, ts = ts)
	i = ts.i
	ts.state.ds_window = [ts.state.ds_window[2:end]; A]
	anomaly = 0.0
	if mod(i, ts.sstep) == 0
		# Downsample
		mean = Statistics.mean(ts.state.ds_window)
		max = maximum(ts.state.ds_window)
		min = minimum(ts.state.ds_window)
		if ts.state.dsi == 1
			ts.state.ds_moving_average = mean
		end
		ts.state.ds_moving_average = (ts.state.ds_moving_average + mean) / 2
		ds = max
		# Spike below the mean
		if abs(max - ts.state.ds_moving_average) < abs(min - ts.state.ds_moving_average)
			ds = min
		end
		if abs(ds - mean) < (0.1 * mean) # spike must be at least 10%
			ds = mean
		end
		# Normalize
		ds = ds - ts.dmin
		if (ts.dmax - ts.dmin) != 0
			ds = ds / (ts.dmax - ts.dmin)
		end
		# Discretize
		level = round(Int, ds * ts.nlevels)
		ds = level / ts.nlevels
		# Levelize
		ts.state.medbin[level+1] += 1
		medpos = ts.state.dsi ÷ 2
		if ts.state.dsi == 1
			ts.state.medlevel = level
		end
		if ts.state.medlevel > level
			ts.state.belowmed += 1
		elseif ts.state.medlevel < level
			ts.state.abovemed += 1
		end
		# Not strictly a running median but close enough
		if medpos < ts.state.abovemed
			ts.state.belowmed += ts.state.medbin[ts.state.medlevel+1]
			ts.state.medlevel += 1
			while ts.state.medbin[ts.state.medlevel+1] == 0
				ts.state.medlevel += 1
			end
			ts.state.abovemed -= ts.state.medbin[ts.state.medlevel+1]
		elseif medpos < ts.state.belowmed
			ts.state.abovemed += ts.state.medbin[ts.state.medlevel+1]
			ts.state.medlevel -= 1
			while ts.state.medbin[ts.state.medlevel+1] == 0
				ts.state.medlevel -= 1
			end
			ts.state.belowmed -= ts.state.medbin[ts.state.medlevel+1]
		end
		med = ts.state.medlevel / ts.nlevels
		if Base.abs(ds - med) < ts.discretize_chomp
			ds = med
		end
		# Extract features
		features = zeros(ts.window * 2)
		ts.state.f_window = [ts.state.f_window[2:end]; ds]
		if ts.state.dsi >= ts.window
			dw = copy(ts.state.f_window)
			dw_min = minimum(dw)
			dw = dw .- dw_min
			dw_max = maximum(dw)
			if dw_max != 0
				dw = dw ./ dw_max
			end
			fw = dwt(dw, ts.wavelett)
			fw_min = minimum(fw)
			fw = (fw .- fw_min)
			fw_max = maximum(fw)
			if fw_max != 0
				fw = fw ./ fw_max
			end
			features = [fw; ts.state.f_window]
		end
		anomaly = process_features!(features, ts.state.dsi, ts)
		ts.state.dsi += 1
	end
	ts.i += 1
	return anomaly
end

"""
	process_features!(f, i, ts) -> Float64

Process extracted features and return anomaly score. Handle both probationary
period training and online anomaly detection.

## Arguments

- `f::Vector{Float64}`: Feature vector extracted from time series window.
- `i::Int`: Downsample index (number of downsampled points processed so far).
- `ts::TimeSeries`: The `TimeSeries` object.

## Returns

- `Float64`: Anomaly confidence score. Returns 0.0 during probationary period.

## Description

Orchestrate the feature processing pipeline with two modes:

### Probationary Period Mode (`i <= trend_window`)

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

### Online Detection Mode (`i > trend_window`)

After probationary period:

- Pass features directly to [`detect!`](@ref) for real-time anomaly detection
- Returns anomaly confidence score (0.0 if no anomaly, > 0.0 if anomaly detected)

## Examples

```julia
# During probationary period (automatically handled by process_sample!)
for i in 1:ts.trend_window
	features = extract_features(data[i])
	score = process_features!(features, i, ts)  # Returns 0.0
end

# After probationary period
for i in (ts.trend_window+1):length(data)
	features = extract_features(data[i])
	score = process_features!(features, i, ts)
	if score > 0
		println("Anomaly detected with score: ", score)
	end
end
```

## Notes

- The probationary period is crucial for learning normal patterns
- Features before `window` are collected but not used for training
- Initial rho is computed from the second half of probationary features
  (indices `window:trend_window`) to ensure sufficient data
- After initialization, the system operates in online mode

See also [`detect!`](@ref), [`init_rho`](@ref), [`process_sample!`](@ref).
"""
function process_features!(f, i, ts)
	anomaly = 0.0
	if i <= ts.trend_window
		# Here we could build a matrix instead of an array
		push!(ts.state.trend_window_f, f)
		# Batch process the probationary period
		if i == ts.trend_window
			features_mat = hcat(ts.state.trend_window_f...)
			ts.state.art.config.dim = length(f)
			ts.state.art.config.dim_comp = 2 * ts.state.art.config.dim
			ts.state.art.config.setup = true
			rho = init_rho(features_mat[:, ts.window:ts.trend_window], ts)
			update_rho!(rho, rho, ts.state.art)
			for (fi, ff) in enumerate(eachcol(features_mat))
				detect!(ff, fi, ts)
			end
		end
	else
		anomaly = detect!(f, i, ts)
	end
	return anomaly
end

"""
	detect!(f, i, ts) -> Float64

Detect anomalies using ART network classification and adaptive vigilance.

## Arguments

- `f::Vector{Float64}`: Feature vector to classify.
- `i::Int`: Downsample index (time step in downsampled space).
- `ts::TimeSeries`: The TimeSeries object.

## Returns

- `Float64`: Anomaly confidence score in range [0, 1]. Returns 0.0 if no anomaly.

## Description

This function implements the core anomaly detection logic with several sophisticated
mechanisms for adaptive thresholding and false positive suppression.

### 1. Timing and Masking Logic

**Update Triggers**:
- `update_rho_after_anomaly`: True exactly `mask_rho_after_anomaly` samples after last anomaly
- `update_rho_for_trend`: True every `trend_window/2` samples for periodic adaptation
- `mask_after_anomaly`: True within `mask_rho_after_anomaly` samples of last anomaly

**Mask Reset**:
- If `mask_after_cat` is active and no new categories for `mask_rho_after_anomaly` samples,
  reset the mask to allow detection again

### 2. ART Classification

**Training Phase** (`i < window`):
- Call ART training with `learning=false`
- No actual learning, just keeps indexes aligned
- Returns category -1 (no valid category)

**Detection Phase** (`i >= window`):
- Call ART training with learning enabled
- Returns assigned category number, or -1 if new category created

### 3. State Updates

- Increment `no_new_cat_count` if category assigned (not -1)
- Update running mean of similarities (`rho_ub_mean`)
- Slide similarity window with latest similarity
- Slide similarity difference window (rho_ub - similarity)
- Track minimum similarity during masking window for each anomaly

### 4. Anomaly Decision

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

### 5. Adaptive Vigilance Update

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

## Example

```julia
# Feature vector extracted from time series
features = extract_wavelet_features(data_window)

# Detect anomaly
score = detect!(features, downsample_index, ts)

if score > 0
	println("Anomaly detected with confidence: ", score)
	println("Similarity: ", ts.state.art.A[end])
	println("Energy similarity: ", ts.state.art.Ae[end])
end
```

## Notes

- The masking mechanism prevents cascading false positives
- Adaptive vigilance allows the system to adapt to changing patterns
- The dual-threshold approach (rho_lb, rho_ub) provides flexibility
- Confidence score combines feature similarity and energy similarity
- The system balances precision (avoiding false positives) and recall (catching anomalies)
"""
function detect!(f, i, ts)
	update_rho_after_anomaly = (i - ts.state.last_anomaly_i) == ts.mask_rho_after_anomaly
	update_rho_for_trend = (i - ts.state.last_rho_update_i) >= ts.trend_window ÷ 2
	mask_after_anomaly = (i - ts.state.last_anomaly_i) <= ts.mask_rho_after_anomaly
	if i > ts.trend_window + ts.mask_rho_after_anomaly && ts.state.mask_after_cat
		if ts.state.no_new_cat_count >= ts.mask_rho_after_anomaly
			ts.state.mask_after_cat = false
		end
	end
	# The samples prior to a complete feature window are not used for training
	# Call train! anyway but don't learn - this keeps ART indexes and arrays aligned with input data
	if i < ts.window
		AdaptiveResonance.train!(ts.state.art, f, learning = false)
		cat = -1
	else
		cat = AdaptiveResonance.train!(ts.state.art, f)
	end
	ts.state.no_new_cat_count = cat == -1 ? 0 : ts.state.no_new_cat_count + 1
	OnlineStats.fit!(ts.state.rho_ub_mean, ts.state.art.A[i]) # running mean
	ts.state.sim_window = [ts.state.sim_window[2:end]; ts.state.art.A[i]]
	ts.state.sim_diff_window = [ts.state.sim_diff_window[2:end]; ts.state.art.opts.rho_ub - ts.state.art.A[i]]
	# Store the smallest similarity during the masking window for each anomaly
	if (i - ts.state.last_anomaly_i) < ts.mask_rho_after_anomaly && length(ts.state.anomaly_sim_history) > 0
		if ts.state.art.A[i] < ts.state.anomaly_sim_history[end]
			ts.state.anomaly_sim_history[end] = ts.state.art.A[i]
		end
	end
	masking_anomaly = ts.state.mask_after_cat || mask_after_anomaly
	below_last_scale = mask_after_anomaly ? 0.90 : 0.70
	below_last = ts.state.art.A[i] < ts.state.last_anomaly_sim * below_last_scale
	anomaly_with_cat = cat == -1 && (!masking_anomaly || below_last)
	if i > ts.trend_window && anomaly_with_cat
		anomaly = confidence(ts.state.art.A[i], ts.state.art.Ae[i], ts)
		push!(ts.state.anomaly_sim_history, ts.state.art.A[i])
		ts.state.last_anomaly_sim = ts.state.art.A[i]
		ts.state.last_anomaly_i = i
	else
		anomaly = 0.0
	end
	ts.state.mask_after_cat = cat == -1 || ts.state.mask_after_cat
	# ART could use supervised learning to improve here, but NAB does not allow this
	if i > ts.trend_window && (update_rho_after_anomaly || update_rho_for_trend)
		min_sim_in_trend_window = minimum(ts.state.sim_window)
		new_rho_ub = OnlineStats.value(ts.state.rho_ub_mean)
		new_rho_ub = min(0.97, new_rho_ub) # capping ub
		prev_rho_lb = ts.state.art.opts.rho_lb
		if prev_rho_lb <= min_sim_in_trend_window
			incr = (min_sim_in_trend_window - prev_rho_lb) * 0.19
			new_rho_lb = prev_rho_lb + incr
		else
			decr = 0.0
			if i > ts.trend_window * 2
				below_rho_idxs = findall(x -> x > 0.05, ts.state.sim_diff_window)
				below_rho = ts.state.sim_diff_window[below_rho_idxs]
				below_rho = map(x -> min(0.37, x), below_rho)
				if length(below_rho) > 0
					decr = mean(below_rho)
				end
			end
			decr = max(0.01, decr)
			new_rho_lb = prev_rho_lb - (decr / 2)
			if length(ts.state.anomaly_sim_history) > 0
				new_rho_lb = max(mean(ts.state.anomaly_sim_history), new_rho_lb)
			end
		end
		new_rho_lb = min(new_rho_ub, new_rho_lb)
		update_rho!(new_rho_lb, new_rho_ub, ts.state.art)
		ts.state.last_rho_update_i = i
		ts.state.mask_after_cat = true
	end
	return anomaly
end

"""
	confidence(features_sim, energy_sim, ts) -> Float64

Compute anomaly confidence score from feature and energy similarities.

## Arguments

- `features_sim::Float64`: Feature similarity score from ART network (0 to 1).
- `energy_sim::Float64`: Energy similarity score from ART network (0 to 1).
- `ts::TimeSeries`: The TimeSeries object containing ART configuration.

## Returns

- `Float64`: Anomaly confidence score in range [0, 1], rounded to 6 decimal places.

## Description

This function computes a confidence score that indicates how likely a sample is an anomaly.
The score combines two components:

### 1. Feature Similarity Component

The feature similarity component measures how far the sample's similarity is from the
vigilance bounds:

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

### 2. Energy Similarity Component

```
energy_component = (1.0 - energy_sim) * 1.5
```
- Inverts energy similarity (lower energy similarity = higher anomaly score)
- Multiplied by 1.5 to give it more influence
- Energy similarity captures different aspects of the pattern

### 3. Combined Score

```
score = feature_component + energy_component
score = min(1.0, score)
```
- Sums both components
- Caps at 1.0 to ensure valid probability-like score

## Interpretation

- **Score = 0.0**: Not an anomaly (high similarity to known patterns)
- **Score > 0.5**: Likely anomaly (moderate similarity)
- **Score ≈ 1.0**: Strong anomaly (very low similarity)

## Examples

```julia
# After ART classification
features_sim = ts.state.art.A[end]      # e.g., 0.65
energy_sim = ts.state.art.Ae[end]        # e.g., 0.70

# Compute confidence
conf = confidence(features_sim, energy_sim, ts)
# Result might be: 0.453217

if conf > 0.5
	println("Anomaly detected with confidence: ", conf)
end
```

## Notes

- Features similarity is capped at 0.999 to avoid division by zero
- The weighting (0.35/0.65) was empirically determined for NAB benchmark
- Energy similarity provides complementary information to feature similarity
- The score is rounded to 6 decimal places for consistency

See also [`detect!`](@ref), [`similarity`](@ref).
"""
function confidence(features_sim, energy_sim, ts)
	features_sim = min(0.999, features_sim)
	ub = ((1 - features_sim) - (1-ts.state.art.opts.rho_ub))/(1-features_sim)
	lb = ((1 - features_sim) - (1-ts.state.art.opts.rho_lb))/(1-features_sim)
	s = (ub*0.35 + lb*0.65) + (1.0 - energy_sim)*1.5
	s = min(1.0, s)
	return round(s, digits = 6)
end

"""
	init_rho(raw_x_optim, ts) -> Float64

Compute initial vigilance parameter (rho) from probationary period features.

## Arguments

- `raw_x_optim::Matrix{Float64}`: Feature matrix from probationary period.
  Each column is a feature vector. Should contain features from indices
  `window:trend_window` (second half of probationary period).
- `ts::TimeSeries`: The TimeSeries object containing configuration.

## Returns

- `Float64`: Initial vigilance parameter value (rho).

## Description

This function computes an appropriate initial vigilance parameter by analyzing
the similarity structure of normal patterns from the probationary period.

### Algorithm Steps

#### 1. Similarity Matrix Construction

Build a pairwise similarity matrix for all feature vectors:

```julia
sim[i, j] = similarity(feature_i, feature_j)
```

- Matrix is symmetric (sim[i,j] = sim[j,i])
- Diagonal is 1.0 (self-similarity)
- Uses [`similarity`](@ref) function (cosine-like similarity)
- Only computes lower triangle for efficiency

#### 2. Similarity Ranking

For each feature vector, compute total similarity to all others:

```julia
sim_sum[i] = sum(sim[:, i])
```

- Higher values indicate the feature is more similar to others
- Sort features by total similarity (ascending order)
- This ordering helps ART create more representative categories

#### 3. Feature Reordering

Reorder feature matrix by similarity ranking:

```julia
raw_x_sort[:, i] = raw_x_optim[sim_order[i]]
```

- Features with lowest total similarity come first
- These are more "unique" patterns
- Helps ART establish good initial categories

#### 4. ART Training with Initial Rho

- Create a new ART instance
- Configure with same dimensions as main ART network
- Set initial rho to `ts.initial_rho` (default: 0.80)
- Train on reordered features

#### 5. Rho Computation

Compute mean similarity from second half of training:

```julia
return mean(art.A[(trend_window÷2):end])
```

- Uses similarities from indices `trend_window÷2` to end
- This skips the initial learning phase
- Provides a stable estimate of typical similarity

## Why This Approach?

1. **Similarity Analysis**: Understanding the similarity structure helps set
   appropriate vigilance for the data distribution

2. **Reordering**: Sorting by similarity helps ART create more representative
   initial categories

3. **Second Half Mean**: Using the second half of similarities avoids the
   initial learning phase where categories are being created

4. **Data-Driven**: The initial rho is computed from actual data rather than
   using a fixed value

## Example

```julia
# After collecting probationary period features
features_mat = hcat(ts.state.trend_window_f...)

# Use second half for rho computation (indices window:trend_window)
rho = init_rho(features_mat[:, ts.window:ts.trend_window], ts)

println("Initial rho: ", rho)
# Output might be: Initial rho: 0.8234

# Set both bounds to this value
update_rho!(rho, rho, ts.state.art)
```

## Notes

- Only uses features from second half of probationary period
- This ensures sufficient data for meaningful similarity computation
- The computed rho typically ranges from 0.75 to 0.90 for normal data
- Higher rho values create more specific categories
- The function is called once during initialization
"""
function init_rho(raw_x_optim, ts)
	lengthx = length(raw_x_optim[1, :])
	raw_x_sort = raw_x_optim
	# Build a similarity matrix
	sim = ones(lengthx, lengthx)
	for i in 1:lengthx
		for j in 1:lengthx
			if j > i
				continue
			end # symmetrical so save some computation
			if i == j
				continue
			end
			sim_score = similarity(raw_x_optim[:, i], raw_x_optim[:, j])
			sim[i, j] = sim_score
			sim[j, i] = sim_score
		end
	end
	sim_sum = zeros(lengthx)
	for i in 1:lengthx
		sim_sum[i] = sum(sim[:, i])
	end
	sim_order = sortperm(sim_sum)
	raw_x_sort = copy(raw_x_optim)
	for (i1, i2) in enumerate(sim_order)
		raw_x_sort[:, i1] = raw_x_optim[:, i2]
	end
	# Find initial rho
	art = AdaptiveResonance.DVFA()
	art.config = ts.state.art.config
	opt_rho = ts.initial_rho
	update_rho!(opt_rho, opt_rho, art)
	AdaptiveResonance.train!(art, raw_x_sort)
	return mean(art.A[(ts.trend_window÷2):end])
end

"""
	similarity(t1, t2) -> Float64

Compute cosine-like similarity between two feature vectors.

## Arguments

- `t1::Vector{Float64}`: First feature vector.
- `t2::Vector{Float64}`: Second feature vector.

## Returns

- `Float64`: Similarity score in range [0, 1]. Higher values indicate more similar vectors.

## Description

This function computes a similarity measure that is close to cosine similarity but
with special handling for zero vectors.

### Standard Cosine Similarity

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

### Zero Vector Handling

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

## Why This Similarity Measure?

1. **Cosine Similarity**: Captures directional similarity, which is important
   for pattern recognition regardless of magnitude

2. **Zero Vector Handling**: Prevents numerical issues and provides sensible
   behavior for edge cases

3. **Range [0, 1]**: Ensures similarity is always non-negative, which is
   appropriate for anomaly detection where we care about dissimilarity

## Examples

```julia
# Identical vectors
v1 = [1.0, 2.0, 3.0]
v2 = [1.0, 2.0, 3.0]
sim = similarity(v1, v2)  # Returns 1.0

# Orthogonal vectors
v1 = [1.0, 0.0, 0.0]
v2 = [0.0, 1.0, 0.0]
sim = similarity(v1, v2)  # Returns 0.0

# Zero vector handling
v1 = [0.0, 0.0, 0.0]
v2 = [1.0, 1.0, 1.0]
sim = similarity(v1, v2)  # Returns 1 - sqrt(3) ≈ -0.732
# But capped at 0.0 in practice

# Scaled vectors (same direction)
v1 = [1.0, 2.0, 3.0]
v2 = [2.0, 4.0, 6.0]
sim = similarity(v1, v2)  # Returns 1.0
```

## Notes

- The function is used extensively in [`init_rho`](@ref) for computing similarity matrices
- Similarity is a key component of ART's matching and learning process
- The zero vector handling is important for robustness in real-world data

See also [`init_rho`](@ref), [`confidence`](@ref).
"""
# Close to cosine similarity
function similarity(t1, t2)
	nt1 = norm(t1)
	nt2 = norm(t2)
	if nt1 == 0.0
		s = 1 - nt2
	else
		if nt2 == 0.0
			s = 1 - nt1
		else
			s = sum(t1 .* t2) / (nt1 * nt2)
		end
	end
	return s
end

"""
	update_rho!(rho_lb, rho_ub, art)

Update the vigilance parameters (rho) of an ART network.

## Arguments

- `rho_lb::Float64`: New lower bound vigilance parameter.
- `rho_ub::Float64`: New upper bound vigilance parameter.
- `art::AdaptiveResonance.DVFA`: The ART network to update.

## Returns

- Nothing (modifies `art` in place).

## Description

This function updates the vigilance parameters that control the granularity
of ART's clustering and anomaly detection.

### Vigilance Parameters

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

### Parameter Relationship

The parameters must satisfy:
```
0.0 <= rho_lb <= rho_ub <= 1.0
```

- `rho_lb` is typically lower to allow some flexibility
- `rho_ub` is typically higher to ensure confident matches
- The gap between them provides a "gray zone" for adaptive behavior

### Usage in ARTime

The vigilance parameters are updated in several contexts:

1. **Initialization**: Both set to the same value computed by [`init_rho`](@ref)

2. **After Anomaly**: Updated to adapt to new patterns
   - May increase to prevent false positives
   - May decrease to maintain sensitivity

3. **Trend Adaptation**: Periodically updated based on recent similarities
   - Tracks changes in data distribution
   - Ensures the system adapts to concept drift

## Examples

```julia
# During initialization
rho = init_rho(features, ts)
update_rho!(rho, rho, ts.state.art)

# After detecting an anomaly
new_rho_lb = 0.75
new_rho_ub = 0.90
update_rho!(new_rho_lb, new_rho_ub, ts.state.art)

# Check updated values
println("rho_lb: ", ts.state.art.opts.rho_lb)
println("rho_ub: ", ts.state.art.opts.rho_ub)
```

## Notes

- Parameters are converted to Float64 for consistency
- The function modifies the ART network in place
- Called by [`detect!`](@ref) during adaptive updates
- The dual-threshold approach provides flexibility in anomaly detection

See also [`init_rho`](@ref), [`detect!`](@ref), [`confidence`](@ref).
"""
function update_rho!(rho_lb, rho_ub, art)
	art.opts.rho_lb = Float64.(rho_lb)
	art.opts.rho_ub = Float64.(rho_ub)
end

end

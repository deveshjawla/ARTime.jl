# This code is derived from https://github.com/AP6YC/AdaptiveResonance.jl
# Some of the changes for ARTime made it difficult to request changes to AdaptiveResonance
# Rather than forking, the essential feature were "compacted" into this file
# Please support the excellent AdaptiveResonance project (licensed under MIT)

"""
# AdaptiveResonance Module

This module implements a compact version of Adaptive Resonance Theory (ART) neural networks,
specifically the Distributed Vigilance Fuzzy ART (DVFA) variant.

## What is Adaptive Resonance Theory?

ART is a family of neural network models that solve the "stability-plasticity dilemma":
- **Plasticity**: Ability to learn new patterns
- **Stability**: Ability to retain previously learned patterns

Traditional neural networks suffer from catastrophic forgetting - learning new patterns
can erase previously learned knowledge. ART networks avoid this by creating new
categories (neurons) when new patterns don't match existing ones.

## Key Concepts

### Complement Coding

ART uses complement coding to handle both presence and absence of features:
- Original features: `x = [x1, x2, ..., xn]`
- Complement code: `x' = [x, 1-x] = [x1, x2, ..., xn, 1-x1, 1-x2, ..., 1-xn]`

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

### Learning Rules

**Fast Commit Learning**:
- When similarity >= rho_ub: Update category weights immediately
- When similarity >= rho_lb: Create new category with current sample

**Weight Update**:
```
W_new = min(W_old, x)
```
This ensures weights represent the intersection of all samples in the category.

### Activation and Match

The network computes two similarity measures:

1. **Feature Similarity (M)**:
```
M = (||min(W, x)||_1 / ||W||_1)^3 * (energy_similarity)^2
```

2. **Energy Similarity (Me)**:
```
Me = ||min(W_e, x_e)||_1 / ||W_e||_1
```
Where `W_e` and `x_e` are the first and last elements (energy components).

## DVFA (Distributed Vigilance Fuzzy ART)

This implementation uses distributed vigilance:
- Dual thresholds (rho_lb, rho_ub) provide flexibility
- Different behaviors based on which threshold is met
- Adaptive threshold adjustment for changing patterns

## Usage Example

julia
using AdaptiveResonance

# Create DVFA network
art = DVFA()

# Configure dimensions
art.config.dim = 10           # Original feature dimension
art.config.dim_comp = 20      # Complement code dimension (2 * dim)
art.config.setup = true

# Set vigilance parameters
update_rho!(0.7, 0.9, art)

# Train on samples
samples = [rand(10) for _ in 1:100]
for sample in samples
	category = train!(art, sample)
	println("Assigned to category: category")
end
```

## Notes

- This is a compact version derived from the full AdaptiveResonance.jl package
- Modifications were made for ARTime's specific requirements
- The original package is licensed under MIT and should be supported
"""
module AdaptiveResonance

using Parameters    # ARTopts are parameters (@with_kw)
using Logging       # Logging utils used as main method of terminal reporting
using LinearAlgebra: norm   # Trace and norms
using Statistics: median, mean  # Medians and mean for linkage methods

# Abstract types
abstract type ARTOpts end               # ART module options
abstract type ARTModule end             # ART modules
abstract type ART <: ARTModule end      # ART (unsupervised)

"""
	DataConfig

Configuration struct for ART network data dimensions.

## Fields

- `setup::Bool`: Flag indicating whether the network has been configured.
  Set to `true` after dimensions are set and the network is ready for training.

- `dim::Integer`: Original feature dimension (before complement coding).
  This is the length of input feature vectors.

- `dim_comp::Integer`: Complement code dimension (after complement coding).
  This is `2 * dim` because complement coding doubles the dimensionality.

## Description

The DataConfig struct stores the dimensional configuration of the ART network.
This is necessary because ART uses complement coding, which transforms input
vectors by concatenating them with their complements.

### Complement Coding

For an input vector `x` of dimension `dim`:
```
x_complement = [x, 1-x]
```

This creates a vector of dimension `dim_comp = 2 * dim`.

### Why Complement Coding?

1. **Magnitude Invariance**: Patterns are recognized based on relative
   proportions, not absolute values

2. **Sparse Data Handling**: Better representation of features that may
   be absent

3. **Pattern Discrimination**: Improves ability to distinguish between
   similar patterns

## Example

```julia
# Create configuration
config = DataConfig()

# Set dimensions
config.dim = 10           # 10-dimensional input
config.dim_comp = 20      # 20-dimensional after complement coding
config.setup = true        # Mark as configured

# Use in ART network
art = DVFA()
art.config = config
```

## Notes

- Must be set before training the network
- `dim_comp` should always be `2 * dim`
- The `setup` flag is checked before training operations
"""
mutable struct DataConfig
	setup::Bool
	dim::Integer
	dim_comp::Integer
end

function DataConfig()
	DataConfig(
		false,                      # setup
		0,                          # dim
		0,                           # dim_comp
	)
end

"""
	opts_DVFA <: ARTOpts

Options struct for DVFA (Distributed Vigilance Fuzzy ART) network parameters.

## Fields

- `rho_lb::Float64`: Lower-bound vigilance parameter in range [0, 1].
  This is the minimum similarity required for category assignment.
  Lower values create fewer, broader categories. Higher values create
  more specific categories.

- `rho_ub::Float64`: Upper bound vigilance parameter in range [0, 1].
  This is the similarity threshold for fast learning. When a sample's
  similarity to a category meets or exceeds this threshold, the category
  is updated immediately (fast commit).

## Description

The opts_DVFA struct contains the vigilance parameters that control the
granularity of ART's clustering behavior. These parameters are critical
for balancing precision and recall in anomaly detection.

### Vigilance Parameter Behavior

**Lower Bound (rho_lb)**:
- Minimum similarity for category assignment
- When similarity < rho_lb: Create new category
- When similarity >= rho_lb: Assign to existing category (if possible)
- Controls the minimum precision of category matching

**Upper Bound (rho_ub)**:
- Threshold for fast learning
- When similarity >= rho_ub: Update category weights immediately
- When rho_lb <= similarity < rho_ub: May create new category
- Provides a "confidence" threshold for category updates

### Parameter Relationship

The parameters must satisfy:
```
0.0 <= rho_lb <= rho_ub <= 1.0
```

Typical values:
- `rho_lb = 0.70 - 0.85`: Allows some flexibility
- `rho_ub = 0.85 - 0.97`: Ensures confident matches
- Gap of 0.10 - 0.20 provides adaptive behavior

### Impact on Clustering

**High Vigilance (rho_lb, rho_ub close to 1.0)**:
- Many small, specific categories
- High precision, low recall
- Good for detecting subtle anomalies
- May create many false positives

**Low Vigilance (rho_lb, rho_ub close to 0.0)**:
- Few large, broad categories
- Low precision, high recall
- Good for general pattern recognition
- May miss subtle anomalies

## Example

```julia
# Create options with default values
opts = opts_DVFA()

# Set vigilance parameters
opts.rho_lb = 0.75  # Minimum similarity for assignment
opts.rho_ub = 0.90  # Threshold for fast learning

# Create DVFA network with options
art = DVFA(opts)

# Or update existing network
art.opts.rho_lb = 0.80
art.opts.rho_ub = 0.95
```

## Notes

- Parameters are validated to be in [0, 1] range
- The @with_kw macro allows keyword argument construction
- Used by ARTime for adaptive threshold adjustment
- Parameters are updated dynamically during operation
"""
@with_kw mutable struct opts_DVFA <: ARTOpts
	@deftype Float64
	# Lower-bound vigilance parameter: [0, 1]
	rho_lb = 0.0;
	@assert rho_lb >= 0.0 && rho_lb <= 1.0
	# Upper bound vigilance parameter: [0, 1]
	rho_ub = 0.0;
	@assert rho_ub >= 0.0 && rho_ub <= 1.0
end # opts_DVFA

"""
	DVFA <: ART

Distributed Vigilance Fuzzy ART network implementation.

## Fields

### Configuration

- `opts::opts_DVFA`: Network options including vigilance parameters (rho_lb, rho_ub).
  See [`opts_DVFA`](@ref) for details.

- `config::DataConfig`: Data configuration including dimensions (dim, dim_comp).
  See [`DataConfig`](@ref) for details.

### Working Variables

- `labels::Vector{Integer}`: Category labels for each training sample.
  Labels are assigned sequentially (1, 2, 3, ...) as new categories are created.

- `W::AbstractArray{Float64, 2}`: Weight matrix for all categories.
  Each column represents a category's weight vector.
  Dimensions: [dim_comp, n_categories].

- `Wx::AbstractArray{Float64, 2}`: Stored minimum values for fast commit learning.
  Pre-computed min(W, x) values stored for reuse.
  Dimensions: [dim_comp, n_categories].

- `M::Vector{Float64}`: Feature similarity scores for each training sample.
  Computed as: `(||min(W, x)||_1 / ||W||_1)^3 * (energy_similarity)^2`
  Length: number of training samples processed.

- `Me::Vector{Float64}`: Energy similarity scores for each training sample.
  Computed from first and last elements of weight and input vectors.
  Length: number of training samples processed.

- `A::Vector{Float64}`: Activation values (feature similarity) for each sample.
  Same as M, but stored separately for convenience.
  Used by ARTime for anomaly detection.

- `Ae::Vector{Float64}`: Energy activation values for each sample.
  Same as Me, but stored separately for convenience.
  Used by ARTime for confidence computation.

- `map::Vector{Integer}`: Category mapping for each training sample.
  Indicates which category each sample was assigned to.
  Length: number of training samples processed.

- `bmu::Vector{Integer}`: Best matching unit (category) for each sample.
  The category with highest similarity to each sample.
  Length: number of training samples processed.

- `n_categories::Integer`: Total number of categories created.
  Increments when new categories are added.

- `n_clusters::Integer`: Total number of clusters (distinct categories).
  May differ from n_categories in some ART variants.

## Description

The DVFA struct implements a Distributed Vigilance Fuzzy ART network,
which is a variant of ART that uses dual vigilance thresholds for
more flexible pattern recognition.

### Key Features

1. **Distributed Vigilance**: Uses both rho_lb and rho_ub for
   different behaviors based on similarity level

2. **Fast Commit Learning**: When similarity >= rho_ub, category weights
   are updated immediately using pre-computed values

3. **Complement Coding**: Automatically applies complement coding to inputs
   for magnitude invariance

4. **Energy Similarity**: Computes additional similarity measure from
   first and last elements for improved discrimination

### Learning Process

1. **Input**: Receive complement-coded input vector
2. **Activation**: Compute similarity to all categories
3. **Match**: Check vigilance criteria
4. **Learning**:
   - If similarity >= rho_ub: Fast commit (update existing category)
   - If similarity >= rho_lb: Create new category
   - Otherwise: Try next category
5. **Output**: Return assigned category label

## Example

```julia
# Create DVFA network
art = DVFA()

# Configure dimensions
art.config.dim = 10
art.config.dim_comp = 20
art.config.setup = true

# Set vigilance parameters
art.opts.rho_lb = 0.75
art.opts.rho_ub = 0.90

# Train on samples
samples = [rand(10) for _ in 1:100]
for sample in samples
	category = train!(art, sample)
end

# Access results
println("Total categories: ", art.n_categories)
println("Similarities: ", art.A)
```

## Notes

- W and Wx matrices grow dynamically as new categories are created
- A and Ae vectors store similarity history for all processed samples
- The network is unsupervised (no labels required)
- Used by ARTime for online anomaly detection
"""
mutable struct DVFA <: ART
	# Get parameters
	opts::opts_DVFA
	config::DataConfig
	# Working variables
	labels::Vector{Integer}
	W::AbstractArray{Float64, 2}
	Wx::AbstractArray{Float64, 2}
	M::Vector{Float64}
	Me::Vector{Float64}
	A::Vector{Float64}
	Ae::Vector{Float64}
	map::Vector{Integer}
	bmu::Vector{Integer}
	n_categories::Integer
	n_clusters::Integer
end

function DVFA()
	opts = opts_DVFA()
	DVFA(opts)
end # DVFA()

function DVFA(opts::opts_DVFA)
	DVFA(
		opts,                           # opts
		DataConfig(),                   # config
		Array{Integer}(undef, 0),       # labels
		Array{Float64}(undef, 0, 0),    # W
		Array{Float64}(undef, 0, 0),    # Wx
		Array{Float64}(undef, 0),       # M
		Array{Float64}(undef, 0),       # Me
		Array{Float64}(undef, 0),       # A
		Array{Float64}(undef, 0),       # Ae
		Array{Integer}(undef, 0),       # map
		Array{Integer}(undef, 0),       # bmu
		0,                              # n_categories
		0,                              # n_clusters
	)
end

"""
	train!(art::DVFA, x; learning::Bool = true) -> Union{Integer, Vector{Integer}}

Train the DVFA network on input sample(s) and return category label(s).

## Arguments

- `art::DVFA`: The DVFA network to train.
- `x::Union{Vector{Float64}, Matrix{Float64}}`: Input sample(s) to train on.
  Can be a single vector or a matrix where each column is a sample.
- `learning::Bool`: Whether to update network weights (default: `true`).
  Set to `false` to only compute similarities without learning.

## Returns

- `Union{Integer, Vector{Integer}}`: Category label(s) for input sample(s).
  Returns `-1` if a new category was created (indicating potential anomaly).

## Description

This function implements the core ART training algorithm with distributed vigilance.
It processes input samples, computes similarities to existing categories, and
updates the network based on vigilance criteria.

### Algorithm Steps

#### 1. Input Preparation

```julia
# Handle single sample or batch
if ndims(x) > 1
	n_samples = size(x)[2]
else
	n_samples = 1
end

# Apply complement coding
x = vcat(x, 1 .- x)
```

Complement coding doubles the dimensionality and provides magnitude invariance.

#### 2. Initialization (First Sample)

If the network has no categories yet:
- Create first category with weight vector of all ones
- Initialize Wx (stored minima) with zeros
- Set category count to 1
- Skip actual training on first sample (just setup)

#### 3. Category Matching Loop

For each sample:

**a) Compute Similarities**:
- Call [`activation_match!`](@ref) to compute similarity to all categories
- Sort categories by similarity (descending order)

**b) Vigilance Testing**:

For each category in sorted order:

1. **Upper Bound Test** (`similarity >= rho_ub`):
   - Fast commit: Update category weights with pre-computed minima
   - Assign sample to this category
   - Stop searching

2. **Lower Bound Test** (`similarity >= rho_lb`):
   - Create new category with current sample
   - Assign sample to new category
   - Stop searching

3. **Mismatch** (`similarity < rho_lb`):
   - Try next category
   - If no categories pass test, create new category at end

#### 4. Weight Update Rules

**Fast Commit Learning** (when `similarity >= rho_ub`):
```julia
W[:, category] = Wx[:, category]
```
Uses pre-computed minima for efficiency.

**New Category Creation**:
```julia
W = hcat(W, sample)
Wx = hcat(Wx, zeros(dim_comp, 1))
```
Adds new column with sample as initial weight.

#### 5. Output

- Store similarity scores in `A` and `Ae` vectors
- Store best matching unit in `bmu` vector
- Return category label (-1 for new category)

### Learning Modes

**Learning Mode** (`learning = true`):
- Updates category weights
- Creates new categories as needed
- Normal training behavior

**No-Learning Mode** (`learning = false`):
- Only computes similarities
- No weight updates
- No new categories created
- Used by ARTime during probationary period

## Example

```julia
# Create and configure network
art = DVFA()
art.config.dim = 10
art.config.dim_comp = 20
art.config.setup = true
art.opts.rho_lb = 0.75
art.opts.rho_ub = 0.90

# Train on single sample
sample = rand(10)
category = train!(art, sample)
println("Assigned to category: ", category)

# Train on batch of samples
samples = hcat([rand(10) for _ in 1:10]...)
categories = train!(art, samples)
println("Categories: ", categories)

# No-learning mode (just compute similarities)
category = train!(art, sample, learning = false)
```

## Notes

- Returns -1 when new category is created (used for anomaly detection)
- Complement coding is applied automatically
- Categories are searched in order of decreasing similarity
- The network grows dynamically as new categories are created
- Used by ARTime for online anomaly detection
"""
function train!(art::DVFA, x; learning::Bool = true)
	# Data information and setup
	if ndims(x) > 1
		n_samples = size(x)[2]
	else
		n_samples = 1
	end

	x = vcat(x, 1 .- x) # complement code
	if n_samples == 1
		y_hat = zero(Integer)
	else
		y_hat = zeros(Integer, n_samples)
	end
	# Initialization
	if isempty(art.W)
		# Set the first label as either 1 or the first provided label
		local_label = 1
		# Add the local label to the output vector
		if n_samples == 1
			y_hat = local_label
		else
			y_hat[1] = local_label
		end
		# Create a new category and cluster
		art.W = ones(art.config.dim_comp, 1)
		art.Wx = zeros(art.config.dim_comp, 1)
		art.n_categories = 1
		art.n_clusters = 1
		push!(art.labels, local_label)
		# Skip the first training entry
		push!(art.A, 0.0)
		push!(art.Ae, 0.0)
		push!(art.bmu, 1)
		skip_first = true
	else
		skip_first = false
	end
	for i in 1:n_samples
		# Skip the first sample if we just initialized
		(i == 1 && skip_first) && continue
		# Grab the sample slice
		sample = x[:, i]
		max_bmu = 0.0
		max_bmue = 0.0
		bmu_with_max = -1
		# Compute the activation and match for all categories
		activation_match!(art, sample)
		# Sort activation function values in descending order
		index = sortperm(art.M, rev = true)
		# Default to mismatch
		mismatch_flag = true
		label = -1
		# Loop over all categories
		for j ∈ 1:art.n_categories
			# Best matching unit, order does not matter
			bmu = index[j]
			# Vigilance test upper bound
			if art.M[bmu] > max_bmu
				bmu_with_max = bmu
				max_bmue = art.Me[bmu]
				max_bmu = art.M[bmu]
			end
			if !learning
				# no learning
			elseif art.M[bmu] >= art.opts.rho_ub
				# learn with fast commit
				art.W[:, bmu] = art.Wx[:, bmu]
				# Update sample label for output`
				if art.M[bmu] >= max_bmu
					label = art.labels[bmu]
				end
				mismatch_flag = false
				# Vigilance test lower bound
			elseif art.M[bmu] >= art.opts.rho_lb && mismatch_flag
				if art.M[bmu] >= max_bmu
					label = art.labels[bmu]
				end
				push!(art.labels, label)
				# Fast commit the sample, same as per mismatch
				art.W = hcat(art.W, sample)
				art.Wx = hcat(art.W, zeros(art.config.dim_comp, 1))
				art.n_categories += 1
				# No mismatch
				mismatch_flag = false
				break
			else
				break
			end
		end
		push!(art.A, max_bmu)
		push!(art.Ae, max_bmue)
		push!(art.bmu, bmu_with_max)
		# If there was no resonant category, make a new one
		if mismatch_flag && learning
			label = -1
			push!(art.labels, art.n_clusters + 1)
			# Fast commit the sample
			art.W = hcat(art.W, sample)
			art.Wx = hcat(art.W, zeros(art.config.dim_comp, 1))
			# Increment the number of categories and clusters
			art.n_categories += 1
			art.n_clusters += 1
		end

		if n_samples == 1
			y_hat = label
		else
			y_hat[i] = label
		end
	end
	return y_hat
end

"""
	activation_match!(art::DVFA, x::Vector)

Compute activation (similarity) and match values for input sample against all categories.

## Arguments

- `art::DVFA`: The DVFA network.
- `x::Vector{Float64}`: Complement-coded input sample.

## Returns

- Nothing (modifies `art.M` and `art.Me` in place).

## Description

This function computes similarity measures between the input sample and all
existing categories in the ART network. It computes two types of similarity:
feature similarity and energy similarity.

### Algorithm Steps

#### 1. Initialize Similarity Vectors

```julia
art.M = zeros(art.n_categories)   # Feature similarity
art.Me = zeros(art.n_categories)  # Energy similarity
```

#### 2. Compute Energy Index

```julia
ei = length(x) ÷ 2
```

This is the midpoint index, separating original features from their complements.

#### 3. For Each Category

**a) Compute Element-wise Minimum**:
```julia
em = minimum([x W], dims = 2)
```
- Computes element-wise minimum between input and category weights
- This is the fuzzy AND operation
- Stored in `Wx` for reuse in fast commit learning

**b) Compute Feature Similarity**:
```julia
numerator = norm(em, 1)      # L1 norm of minima
nW = norm(W, 1)              # L1 norm of weights
if nW == 0
	nW = 0.001  # Prevent division by zero
end
feature_similarity = numerator / nW
```

This measures how much of the category's weight is covered by the input.

**c) Compute Energy Similarity**:
```julia
eme = [em[ei]; em[end]]    # Energy of minima
We = [W[ei]; W[end]]        # Energy of weights
neme = norm(eme, 1)
nWe = norm(We, 1)
energy_similarity = neme / nWe
```

This measures similarity based on first and last elements (energy components).

**d) Compute Combined Activation**:
```julia
art.M[jx] = feature_similarity^3 * energy_similarity^2
art.Me[jx] = energy_similarity
```

The feature similarity is cubed and energy similarity is squared, giving
more weight to feature similarity in the final activation.

### Similarity Interpretation

**Feature Similarity**:
- Range: [0, 1]
- Higher values indicate input matches category well
- Based on fuzzy set intersection

**Energy Similarity**:
- Range: [0, 1]
- Higher values indicate energy components match
- Provides complementary information

**Combined Activation (M)**:
- Range: [0, 1]
- Used for category selection and vigilance testing
- Weighted combination favoring feature similarity

### Why Two Similarity Measures?

1. **Feature Similarity**: Captures overall pattern match
2. **Energy Similarity**: Focuses on boundary/energy components
3. **Combined**: Provides robust discrimination between categories

## Example

```julia
# Create and configure network
art = DVFA()
art.config.dim = 10
art.config.dim_comp = 20
art.config.setup = true

# Train some categories
for _ in 1:5
	sample = rand(10)
	train!(art, sample)
end

# Compute similarities for new sample
new_sample = rand(10)
new_sample_complement = vcat(new_sample, 1 .- new_sample)

activation_match!(art, new_sample_complement)

# Access results
println("Feature similarities: ", art.M)
println("Energy similarities: ", art.Me)

# Find best matching category
best_category = argmax(art.M)
println("Best match: Category ", best_category, " with similarity ", art.M[best_category])
```

## Notes

- Modifies `art.M` and `art.Me` in place
- Stores minima in `art.Wx` for reuse in fast commit learning
- Uses L1 norms (sum of absolute values)
- Energy components are first and last elements of complement-coded vectors
- Called by [`train!`](@ref) during category matching
"""
function activation_match!(art::DVFA, x::Vector)
	art.M = zeros(art.n_categories)
	art.Me = zeros(art.n_categories)
	ei = length(x)÷2
	for jx ∈ 1:art.n_categories
		W = art.W[:, jx]
		em = minimum([x W], dims = 2)
		art.Wx[:, jx] = em #stored because this can be reused in the learning (fast commit)
		numerator = norm(em, 1)
		nW = norm(W, 1)
		if nW == 0
			nW = 0.001
		end
		feature_similarity = numerator/nW
		eme = [em[ei]; em[end]]
		We = [W[ei]; W[end]]
		neme = norm(eme, 1)
		nWe = norm(We, 1)
		energy_similarity = neme / nWe
		art.M[jx] = feature_similarity^3 * energy_similarity^2
		art.Me[jx] = energy_similarity
	end
end

end

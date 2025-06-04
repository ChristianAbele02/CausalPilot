# Theory Behind Causal Inference Estimators

## Introduction

This document provides a comprehensive theoretical foundation for the causal inference estimators implemented in CausalPilot. Each method addresses specific challenges in causal inference while making different assumptions about the data-generating process and treatment effect heterogeneity [16][17].

## 1. Double/Debiased Machine Learning (DoubleML)

### Theoretical Foundation

Double Machine Learning, introduced by Chernozhukov et al. (2018), addresses the challenge of estimating causal effects in high-dimensional settings where traditional parametric methods may be biased or inefficient [9][10][13].

### Mathematical Framework

Consider the partially linear regression model:
```
Y = θT + g(X) + U
T = m(X) + V
```

Where:
- `Y` is the outcome variable
- `T` is the treatment variable  
- `X` are confounding covariates
- `θ` is the causal parameter of interest
- `g(X) = E[Y|X]` is the conditional expectation function
- `m(X) = E[T|X]` is the propensity score
- `U, V` are error terms with `E[U|X] = E[V|X] = 0`

### Key Theoretical Components

#### 1. Neyman Orthogonality
The estimating equation satisfies:
```
E[ψ(W; θ, η)] = 0
```
where `ψ` is approximately insensitive to perturbations in nuisance parameters `η = (g, m)`. This ensures that small errors in nuisance function estimation don't dramatically affect the causal parameter estimate [9].

#### 2. Cross-Fitting Procedure
1. **Sample Splitting**: Randomly partition data into K folds
2. **Nuisance Estimation**: For each fold k:
   - Train models `ĝ₋ₖ` and `m̂₋ₖ` on data excluding fold k
   - Predict on fold k to get `ĝ₋ₖ(Xᵢ)` and `m̂₋ₖ(Xᵢ)` for i ∈ k
3. **Parameter Estimation**: Solve the moment condition using all folds

#### 3. Moment Condition
The orthogonal moment condition for the partially linear model is:
```
ψ(Wᵢ; θ, η) = (Tᵢ - m(Xᵢ))(Yᵢ - θTᵢ - g(Xᵢ))
```

### Theoretical Guarantees

Under regularity conditions, the DoubleML estimator is:
- **√n-consistent**: Converges at the parametric rate
- **Asymptotically normal**: Normal limiting distribution enables confidence intervals
- **Semiparametrically efficient**: Achieves the efficiency bound

### Advantages and Limitations

**Advantages:**
- Robust to misspecification of either `g(X)` or `m(X)` 
- Handles high-dimensional confounders
- Provides valid inference under weak conditions

**Limitations:**
- Requires unconfoundedness assumption
- May struggle with very small treatment groups
- Computational complexity scales with cross-fitting

## 2. Causal Forest

### Theoretical Foundation

Causal Forests, developed by Wager and Athey (2018), extend random forests to estimate heterogeneous treatment effects by building trees that explicitly optimize for treatment effect heterogeneity rather than prediction accuracy [11][14].

### Mathematical Framework

The goal is to estimate the Conditional Average Treatment Effect (CATE):
```
τ(x) = E[Y(1) - Y(0)|X = x]
```

Where `Y(1)` and `Y(0)` are potential outcomes under treatment and control.

### Algorithm Structure

#### 1. Honest Tree Construction
Each tree is built using sample splitting:
- **I**: Subsample for determining splits
- **J**: Disjoint subsample for estimating treatment effects

This "honesty" prevents overfitting and ensures valid statistical inference [11].

#### 2. Splitting Criterion
Instead of optimizing prediction accuracy, causal trees maximize treatment effect heterogeneity. For a potential split of parent node P into children C₁ and C₂:

```
Split criterion = (n₁n₂)/(n₁ + n₂) × (τ̂(C₁) - τ̂(C₂))²
```

Where `τ̂(Cₖ)` is the estimated treatment effect in child k.

#### 3. Treatment Effect Estimation
In each leaf, the treatment effect is estimated using:
```
τ̂ₗₑₐf = (∑ᵢ∈ₗₑₐf TᵢYᵢ)/(∑ᵢ∈ₗₑₐf Tᵢ) - (∑ᵢ∈ₗₑₐf (1-Tᵢ)Yᵢ)/(∑ᵢ∈ₗₑₐf (1-Tᵢ))
```

#### 4. Forest Aggregation
The final estimate is a weighted average:
```
τ̂(x) = ∑ᵢ₌₁ⁿ αᵢ(x)Yᵢ
```

Where `αᵢ(x)` represents the weight given to observation i when predicting at point x, determined by how often i and x fall in the same leaf across trees.

### Theoretical Properties

- **Consistency**: `τ̂(x) →ᵖ τ(x)` under regularity conditions
- **Asymptotic Normality**: Enables confidence intervals and hypothesis testing
- **Adaptive**: Automatically discovers effect heterogeneity without pre-specification

### Advantages and Limitations

**Advantages:**
- No parametric assumptions about treatment effect function
- Automatically discovers effect heterogeneity patterns
- Provides valid confidence intervals
- Handles complex interactions

**Limitations:**
- Requires larger sample sizes for accurate effect estimation
- Black-box nature limits interpretability
- Assumes unconfoundedness

## 3. Meta-Learners (T-learner, S-learner, X-learner)

### Theoretical Framework

Meta-learners are flexible approaches that use any supervised learning algorithm as a "base learner" to estimate treatment effects [12][15].

### T-learner (Two-Model Approach)

#### Algorithm
1. **Separate Models**: Train two models:
   - `μ₀(x) = E[Y|X = x, T = 0]` (control model)
   - `μ₁(x) = E[Y|X = x, T = 1]` (treatment model)

2. **Effect Estimation**: 
   ```
   τ̂(x) = μ̂₁(x) - μ̂₀(x)
   ```

#### Theoretical Properties
- **Consistency**: Consistent if both base models are consistent
- **Bias**: Bias depends on bias of both models:
  ```
  Bias[τ̂(x)] = Bias[μ̂₁(x)] - Bias[μ̂₀(x)]
  ```

#### Advantages and Limitations
**Advantages:**
- Simple and intuitive
- Flexible choice of base learner
- Good when treatment and control responses differ substantially

**Limitations:**
- Requires sufficient data in both treatment groups
- No regularization across treatment groups
- Ignores connection between μ₀ and μ₁

### S-learner (Single-Model Approach)

#### Algorithm
1. **Combined Model**: Train single model:
   ```
   μ(x,t) = E[Y|X = x, T = t]
   ```

2. **Effect Estimation**:
   ```
   τ̂(x) = μ̂(x,1) - μ̂(x,0)
   ```

#### Theoretical Properties
- **Bias**: Depends on how well the model captures treatment-covariate interactions
- **Efficiency**: Can be more efficient when treatment effects are small relative to main effects

#### Advantages and Limitations
**Advantages:**
- Uses all data for model training
- Good when treatment effects are small
- Natural when treatment is just another feature

**Limitations:**
- May miss treatment effect heterogeneity if base learner doesn't capture interactions well
- Biased if treatment effect is large relative to model capacity

### X-learner (Cross-learner)

#### Algorithm
1. **Stage 1**: Train separate outcome models (like T-learner)
2. **Stage 2**: Estimate individual treatment effects:
   - For treated units: `τ̂₁ᵢ = Yᵢ - μ̂₀(Xᵢ)`
   - For control units: `τ̂₀ᵢ = μ̂₁(Xᵢ) - Yᵢ`
3. **Stage 3**: Train models to predict these effects:
   - `τ̂₁(x) = E[τ̂₁|X = x, T = 1]`  
   - `τ̂₀(x) = E[τ̂₀|X = x, T = 0]`
4. **Final Estimate**: Weighted combination using propensity scores

#### Theoretical Advantages
- **Doubly Robust**: Good performance if either outcome models or propensity model is correct
- **Efficiency**: Often more efficient than T-learner or S-learner

## 4. Comparison of Methods

| Method | Best Use Case | Key Assumption | Computational Cost |
|--------|---------------|----------------|-------------------|
| DoubleML | High-dimensional confounders | Unconfoundedness | Medium |
| Causal Forest | Effect heterogeneity discovery | Unconfoundedness | High |
| T-learner | Clear treatment groups | Sufficient sample in each group | Low |
| S-learner | Small treatment effects | Treatment-outcome relationship captured | Low |
| X-learner | Imbalanced treatment groups | Either outcomes or propensity correct | Medium |

## 5. Identification Assumptions

All methods rely on fundamental causal identification assumptions:

### Unconfoundedness (No Unmeasured Confounding)
```
{Y(0), Y(1)} ⊥ T | X
```
Treatment assignment is as good as random conditional on observed covariates.

### Positivity (Overlap)
```
0 < P(T = 1|X = x) < 1 for all x in support of X
```
Every unit has positive probability of receiving either treatment.

### Consistency
```
Y = T·Y(1) + (1-T)·Y(0)
```
Observed outcome equals the potential outcome under the received treatment.

## Conclusion

Each estimator offers different strengths for various causal inference scenarios. DoubleML excels with high-dimensional data, Causal Forests discover complex heterogeneity patterns, and meta-learners provide flexible, interpretable approaches. The choice depends on the specific research question, data characteristics, and computational constraints [9][11][12].
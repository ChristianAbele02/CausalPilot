# Theory Behind Causal Inference Estimators

## Introduction

This document provides a comprehensive theoretical foundation for the causal inference estimators I have implemented in CausalPilot. Each method addresses specific challenges in causal inference while making different assumptions about the data-generating process and treatment effect heterogeneity.

## 1. Double/Debiased Machine Learning (DoubleML)

### Theoretical Foundation

Double Machine Learning addresses the challenge of estimating causal effects in high-dimensional settings where traditional parametric methods may be biased or inefficient. I chose to implement this because it allows for the use of powerful machine learning models (like Random Forests or Gradient Boosting) to control for confounders without introducing regularization bias into the causal estimate.

### Mathematical Framework

Consider the partially linear regression model:
$$ Y = \theta T + g(X) + U $$
$$ T = m(X) + V $$

Where:
- $Y$ is the outcome variable
- $T$ is the treatment variable  
- $X$ are confounding covariates
- $\theta$ is the causal parameter of interest
- $g(X) = E[Y|X]$ is the conditional expectation function
- $m(X) = E[T|X]$ is the propensity score
- $U, V$ are error terms with $E[U|X] = E[V|X] = 0$

### Key Theoretical Components

#### 1. Neyman Orthogonality
The estimating equation satisfies:
$$ E[\psi(W; \theta, \eta)] = 0 $$
where $\psi$ is approximately insensitive to perturbations in nuisance parameters $\eta = (g, m)$. This ensures that small errors in nuisance function estimation don't dramatically affect the causal parameter estimate.

#### 2. Cross-Fitting Procedure
I implemented cross-fitting to avoid overfitting:
1. **Sample Splitting**: Randomly partition data into $K$ folds
2. **Nuisance Estimation**: For each fold $k$:
   - Train models $\hat{g}_{-k}$ and $\hat{m}_{-k}$ on data excluding fold $k$
   - Predict on fold $k$ to get $\hat{g}_{-k}(X_i)$ and $\hat{m}_{-k}(X_i)$ for $i \in k$
3. **Parameter Estimation**: Solve the moment condition using all folds

#### 3. Moment Condition
The orthogonal moment condition for the partially linear model is:
$$ \psi(W_i; \theta, \eta) = (T_i - m(X_i))(Y_i - \theta T_i - g(X_i)) $$

### Theoretical Guarantees

Under regularity conditions, the DoubleML estimator is:
- **$\sqrt{n}$-consistent**: Converges at the parametric rate
- **Asymptotically normal**: Normal limiting distribution enables confidence intervals
- **Semiparametrically efficient**: Achieves the efficiency bound

## 2. Causal Forest

### Theoretical Foundation

Causal Forests extend random forests to estimate heterogeneous treatment effects by building trees that explicitly optimize for treatment effect heterogeneity rather than prediction accuracy. I included this estimator to allow users to understand *who* benefits most from a treatment.

### Mathematical Framework

The goal is to estimate the Conditional Average Treatment Effect (CATE):
$$ \tau(x) = E[Y(1) - Y(0)|X = x] $$

Where $Y(1)$ and $Y(0)$ are potential outcomes under treatment and control.

### Algorithm Structure

#### 1. Honest Tree Construction
Each tree is built using sample splitting:
- **I**: Subsample for determining splits
- **J**: Disjoint subsample for estimating treatment effects

This "honesty" prevents overfitting and ensures valid statistical inference.

#### 2. Splitting Criterion
Instead of optimizing prediction accuracy, causal trees maximize treatment effect heterogeneity. For a potential split of parent node $P$ into children $C_1$ and $C_2$:

$$ \text{Split criterion} = \frac{n_1 n_2}{n_1 + n_2} \times (\hat{\tau}(C_1) - \hat{\tau}(C_2))^2 $$

Where $\hat{\tau}(C_k)$ is the estimated treatment effect in child $k$.

#### 3. Treatment Effect Estimation
In each leaf, the treatment effect is estimated using:
$$ \hat{\tau}_{leaf} = \frac{\sum_{i \in leaf} T_i Y_i}{\sum_{i \in leaf} T_i} - \frac{\sum_{i \in leaf} (1-T_i) Y_i}{\sum_{i \in leaf} (1-T_i)} $$

#### 4. Forest Aggregation
The final estimate is a weighted average:
$$ \hat{\tau}(x) = \sum_{i=1}^n \alpha_i(x) Y_i $$

Where $\alpha_i(x)$ represents the weight given to observation $i$ when predicting at point $x$, determined by how often $i$ and $x$ fall in the same leaf across trees.

## 3. Meta-Learners (T-learner, S-learner, X-learner)

### Theoretical Framework

Meta-learners are flexible approaches that use any supervised learning algorithm as a "base learner" to estimate treatment effects.

### T-learner (Two-Model Approach)

#### Algorithm
1. **Separate Models**: Train two models:
   - $\mu_0(x) = E[Y|X = x, T = 0]$ (control model)
   - $\mu_1(x) = E[Y|X = x, T = 1]$ (treatment model)

2. **Effect Estimation**: 
   $$ \hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x) $$

#### Advantages and Limitations
**Advantages:**
- Simple and intuitive
- Flexible choice of base learner

**Limitations:**
- Requires sufficient data in both treatment groups
- No regularization across treatment groups

### S-learner (Single-Model Approach)

#### Algorithm
1. **Combined Model**: Train single model:
   $$ \mu(x,t) = E[Y|X = x, T = t] $$

2. **Effect Estimation**:
   $$ \hat{\tau}(x) = \hat{\mu}(x,1) - \hat{\mu}(x,0) $$

#### Advantages and Limitations
**Advantages:**
- Uses all data for model training
- Good when treatment effects are small

**Limitations:**
- May miss treatment effect heterogeneity if base learner doesn't capture interactions well

### X-learner (Cross-learner)

I implemented the X-Learner specifically for cases where one treatment group is much larger than the other (e.g., a small treatment group and a large control group).

#### Algorithm
1. **Stage 1**: Train separate outcome models (like T-learner)
2. **Stage 2**: Estimate individual treatment effects:
   - For treated units: $\hat{\tau}_{1i} = Y_i - \hat{\mu}_0(X_i)$
   - For control units: $\hat{\tau}_{0i} = \hat{\mu}_1(X_i) - Y_i$
3. **Stage 3**: Train models to predict these effects:
   - $\hat{\tau}_1(x) = E[\hat{\tau}_1|X = x, T = 1]$  
   - $\hat{\tau}_0(x) = E[\hat{\tau}_0|X = x, T = 0]$
4. **Final Estimate**: Weighted combination using propensity scores

#### Theoretical Advantages
- **Doubly Robust**: Good performance if either outcome models or propensity model is correct
- **Efficiency**: Often more efficient than T-learner or S-learner

## 4. Instrumental Variables (IV2SLS)

### Theoretical Foundation

Sometimes, we have **unobserved confounders** ($U$) that affect both $T$ and $Y$. In this case, adjusting for $X$ is not enough. I implemented **Two-Stage Least Squares (IV2SLS)** to handle this scenario.

### Mathematical Framework

We assume the following structural equations:
$$ Y = \beta T + \gamma X + U $$
$$ T = \delta Z + \phi X + V $$

Where $Z$ is an **Instrument** that satisfies:
1. **Relevance**: $Z$ affects $T$ ($\delta \neq 0$).
2. **Exclusion**: $Z$ affects $Y$ *only* through $T$ (no direct path $Z \to Y$).
3. **Independence**: $Z$ is independent of $U$.

### Algorithm

1. **First Stage**: Regress $T$ on $Z$ and $X$ to get $\hat{T}$.
   $$ \hat{T} = \hat{\delta} Z + \hat{\phi} X $$
2. **Second Stage**: Regress $Y$ on $\hat{T}$ and $X$.
   $$ Y = \beta_{IV} \hat{T} + \gamma_{IV} X + \epsilon $$

This isolates the variation in $T$ that is caused by $Z$ (which is clean) and ignores the variation caused by $U$ (which is confounded).

## 5. Comparison of Methods

| Method | Best Use Case | Key Assumption | Computational Cost |
|--------|---------------|----------------|-------------------|
| DoubleML | High-dimensional confounders | Unconfoundedness | Medium |
| Causal Forest | Effect heterogeneity discovery | Unconfoundedness | High |
| T-learner | Clear treatment groups | Sufficient sample in each group | Low |
| S-learner | Small treatment effects | Treatment-outcome relationship captured | Low |
| X-learner | Imbalanced treatment groups | Either outcomes or propensity correct | Medium |
| IV2SLS | Unobserved confounding | Valid Instrument | Low |

## Conclusion

Each estimator offers different strengths for various causal inference scenarios. DoubleML excels with high-dimensional data, Causal Forests discover complex heterogeneity patterns, and meta-learners provide flexible, interpretable approaches. The choice depends on the specific research question, data characteristics, and computational constraints.

## References

- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment and structural parameters."
- Wager, S., & Athey, S. (2018). "Estimation and inference of heterogeneous treatment effects using random forests."
- Künzel, S. R., et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning."
- Angrist, J. D., & Pischke, J. S. (2009). "Mostly harmless econometrics: An empiricist's companion."
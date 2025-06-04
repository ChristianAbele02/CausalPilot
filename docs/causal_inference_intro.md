# Statistical Introduction to Causal Inference

## Introduction

Causal inference represents one of the most fundamental challenges in statistics and data science: moving beyond correlation to establish true cause-and-effect relationships. This document provides a comprehensive statistical introduction to causal inference, covering the theoretical foundations, key frameworks, and fundamental assumptions that underlie modern causal analysis [17][18].

## 1. The Fundamental Problem of Causal Inference

### Beyond Correlation: The Need for Causal Thinking

Traditional statistical methods excel at identifying associations and correlations between variables. However, answering questions about the effects of interventions requires a fundamentally different approach [16][17]. Consider these examples:

- **Predictive Question**: What is the probability a patient will recover given they take medication X?
- **Causal Question**: What would happen to the patient's recovery if we intervene and give them medication X?

The key distinction lies in the counterfactual nature of causal questions: we seek to understand what would happen under different interventions, not just what we observe [17].

### The Missing Data Problem

The fundamental challenge in causal inference stems from the fact that we can never observe what would have happened to the same unit under a different treatment condition. This is known as the "fundamental problem of causal inference" [17][19].

For each individual i:
- We observe either Y_i(1) (outcome under treatment) OR Y_i(0) (outcome under control)
- We never observe both potential outcomes for the same individual
- The unobserved outcome represents missing data that cannot be recovered through repeated experiments

## 2. Frameworks for Causal Inference

### 2.1 Potential Outcomes Framework (Rubin Causal Model)

The potential outcomes framework, developed by Rubin (1974-1978), provides a mathematical foundation for causal inference by explicitly modeling counterfactual outcomes [19][21].

#### Core Concepts

**Units**: The entities (people, places, things) on which treatments operate at specific times.

**Treatments**: Interventions whose effects we wish to assess. For simplicity, consider binary treatment T ∈ {0,1}.

**Potential Outcomes**: For each unit i, define:
- Y_i(1): Outcome if unit i receives treatment (T=1)
- Y_i(0): Outcome if unit i receives control (T=0)

**Observed Outcome**: What we actually observe:
```
Y_i = T_i · Y_i(1) + (1 - T_i) · Y_i(0)
```

**Individual Treatment Effect**: The causal effect for unit i:
```
τ_i = Y_i(1) - Y_i(0)
```

#### Key Estimands

**Average Treatment Effect (ATE)**:
```
ATE = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]
```

**Average Treatment Effect on the Treated (ATT)**:
```
ATT = E[Y_i(1) - Y_i(0) | T_i = 1]
```

**Conditional Average Treatment Effect (CATE)**:
```
CATE(x) = E[Y_i(1) - Y_i(0) | X_i = x]
```

### 2.2 Structural Causal Models and Causal Graphs

The graphical approach to causation, developed by Pearl (1995, 2000), uses directed acyclic graphs (DAGs) to represent causal relationships and identify conditions under which causal effects can be estimated [17][18].

#### Causal Graphs

A causal graph G = (V, E) consists of:
- **Vertices (V)**: Variables in the system
- **Directed Edges (E)**: Direct causal relationships
- **DAG Property**: No directed cycles (A cannot cause B which causes A)

**Causal Interpretation**: An edge X → Y means X is a direct cause of Y, holding all other variables constant.

#### Path Analysis

**Causal Paths**: Sequences of edges following the direction of arrows (X → Z → Y)
**Backdoor Paths**: Paths from treatment to outcome that begin with an arrow into the treatment
**Confounding Paths**: Backdoor paths that create spurious associations

## 3. Fundamental Identification Assumptions

To move from observational data to causal conclusions, we must make untestable assumptions about the data-generating process [17][18][19].

### 3.1 Unconfoundedness (Ignorability)

**Assumption**: Treatment assignment is unconfounded given observed covariates X.

**Mathematical Statement**:
```
{Y_i(0), Y_i(1)} ⊥ T_i | X_i
```

**Interpretation**: Conditional on X, treatment assignment is as good as random—there are no unmeasured confounders affecting both treatment and outcome.

**Plausibility**: This is the most critical and often most questionable assumption. It requires:
- All relevant confounders are observed and measured
- No hidden bias from unmeasured variables
- Complete understanding of the selection mechanism

### 3.2 Positivity (Overlap)

**Assumption**: Every unit has positive probability of receiving any treatment level.

**Mathematical Statement**:
```
0 < P(T_i = t | X_i = x) < 1
```
for all t ∈ {0,1} and all x in the support of X.

**Interpretation**: There is sufficient overlap between treatment groups across all covariate levels.

**Violations**: 
- **Structural violations**: Some subgroups never receive treatment (e.g., men don't get pregnant)
- **Random violations**: Small sample sizes lead to missing combinations

### 3.3 Consistency (SUTVA)

The Stable Unit Treatment Value Assumption (SUTVA) has two components:

#### No Interference
**Assumption**: Treatment of one unit doesn't affect outcomes of other units.

**Mathematical Statement**:
```
Y_i(t_1, ..., t_i, ..., t_n) = Y_i(t_i)
```

**Violations**: Network effects, spillovers, general equilibrium effects

#### Treatment Variation Irrelevance
**Assumption**: There is only one version of each treatment level.

**Violations**: Multiple versions of "treatment" with different effects

## 4. Identification Strategies

### 4.1 Backdoor Criterion

The backdoor criterion provides a graphical method for identifying valid adjustment sets [20][22].

**Definition**: A set of variables Z satisfies the backdoor criterion relative to treatment T and outcome Y if:
1. Z blocks all backdoor paths from T to Y
2. Z contains no descendants of T

**Adjustment Formula**: If Z satisfies the backdoor criterion:
```
P(Y(t)) = ∑_z P(Y | T = t, Z = z) P(Z = z)
```

#### Algorithm for Finding Adjustment Sets

1. **Identify all paths** from treatment T to outcome Y
2. **Classify paths** as front-door (causal) or backdoor (confounding)
3. **Find blocking sets** that close all backdoor paths without blocking front-door paths
4. **Exclude descendants** of the treatment to avoid post-treatment bias

### 4.2 Instrumental Variables

When unconfoundedness fails, instrumental variables can provide identification under different assumptions.

**Definition**: An instrument Z for the effect of T on Y satisfies:
1. **Relevance**: Z affects T (correlation condition)
2. **Exclusion**: Z affects Y only through T (exogeneity condition)  
3. **Monotonicity**: Z affects T in the same direction for all units

**Two-Stage Least Squares (2SLS)**:
1. **First stage**: Predict T using Z
2. **Second stage**: Use predicted T̂ to estimate effect on Y

## 5. Estimation Approaches

### 5.1 Regression-Based Methods

**Simple Regression Adjustment**:
```
E[Y | T = 1, X] - E[Y | T = 0, X]
```

**Challenges**:
- Requires correct specification of E[Y | T, X]
- Sensitive to model misspecification
- Parametric assumptions may be violated

### 5.2 Propensity Score Methods

**Propensity Score**: The probability of treatment given covariates:
```
e(x) = P(T = 1 | X = x)
```

**Key Property**: If unconfoundedness holds given X, it also holds given e(X).

**Estimation Methods**:
1. **Matching**: Match treated and control units with similar propensity scores
2. **Stratification**: Group units into propensity score strata
3. **Weighting**: Weight by inverse propensity scores (IPW)

### 5.3 Modern Machine Learning Approaches

**Double/Debiased Machine Learning**: Uses cross-fitting to estimate nuisance functions while maintaining valid inference.

**Causal Forests**: Extends random forests to estimate heterogeneous treatment effects.

**Meta-learners**: Use any supervised learning algorithm to estimate treatment effects.

## 6. Threats to Causal Inference

### 6.1 Selection Bias

**Problem**: Systematic differences between treatment groups that affect outcomes.

**Sources**:
- Self-selection into treatment
- Non-random treatment assignment
- Unmeasured confounding

**Solutions**: Randomization, natural experiments, instrumental variables

### 6.2 Confounding

**Problem**: Variables that affect both treatment and outcome, creating spurious associations.

**Types**:
- **Measured confounding**: Can be addressed through adjustment
- **Unmeasured confounding**: Requires additional assumptions or design features

### 6.3 Post-Treatment Bias

**Problem**: Conditioning on variables affected by treatment can introduce bias.

**Example**: Studying effect of education on income while controlling for occupation (which education affects).

**Solution**: Only adjust for pre-treatment variables.

## 7. Sensitivity Analysis

Given the reliance on untestable assumptions, sensitivity analysis examines how conclusions change under assumption violations.

### 7.1 Unmeasured Confounding

**Rosenbaum Bounds**: Quantify how strong unmeasured confounding would need to be to explain away the estimated effect.

**Simulation-Based Approaches**: Model specific confounding scenarios and assess robustness.

### 7.2 Model Uncertainty

**Cross-Validation**: Assess stability across different model specifications.

**Ensemble Methods**: Combine multiple estimation approaches.

## 8. Study Design Considerations

### 8.1 Randomized Experiments

**Gold Standard**: Randomization ensures treatment assignment is independent of potential outcomes.

**Advantages**:
- Eliminates selection bias
- Balances measured and unmeasured confounders
- Provides valid causal inference under minimal assumptions

**Limitations**:
- May not be feasible or ethical
- External validity concerns
- Compliance issues

### 8.2 Natural Experiments

**Quasi-Randomization**: Exploit natural variation in treatment assignment.

**Examples**:
- Regression discontinuity (arbitrary cutoffs)
- Instrumental variables (plausibly exogenous variation)
- Difference-in-differences (before/after, treatment/control)

### 8.3 Observational Studies

**Challenge**: Must rely on stronger assumptions about selection mechanism.

**Best Practices**:
- Careful measurement of potential confounders
- Use of multiple identification strategies
- Extensive sensitivity analysis
- Theory-guided variable selection

## 9. Software Implementations

Modern causal inference benefits from specialized software that implements these methods:

- **DoWhy**: Provides a four-step workflow (model, identify, estimate, refute)
- **CausalML**: Focuses on machine learning approaches to heterogeneous effects
- **CausalInference**: Classical propensity score and matching methods
- **CausalPilot**: Unified framework for multiple estimation approaches

## Conclusion

Causal inference requires a careful combination of:
1. **Theoretical Understanding**: Potential outcomes, graphs, identification assumptions
2. **Design Considerations**: How to collect data that supports causal conclusions  
3. **Statistical Methods**: Appropriate estimation techniques for the identification strategy
4. **Sensitivity Analysis**: Assessment of robustness to assumption violations

Success in causal inference depends more on credible identification strategies than on sophisticated estimation methods. The most important question is not "What is the fanciest method?" but rather "What assumptions am I making, and how plausible are they?" [17][18].

## References

[16] Siebert, J. (2024). "Causal inference: An introduction." Fraunhofer IESE Blog.
[17] Pearl, J. (2010). "An Introduction to Causal Inference." Journal of Causal Inference.
[18] Pearl, J. (2009). "Causal inference in statistics: An overview." Statistical Science.
[19] Rubin, D. B. (2005). "Causal inference using potential outcomes." Journal of the American Statistical Association.
[20] Pearl, J. (1995). "Causal diagrams for empirical research." Biometrika.
[21] Holland, P. W. (1986). "Statistics and causal inference." Journal of the American Statistical Association.
[22] Shpitser, I. et al. "A complete identification procedure for causal effects." Proceedings of UAI.
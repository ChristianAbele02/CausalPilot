# Future Developments Roadmap

As the solo developer of CausalPilot, I have a clear vision for where I want to take this project. My goal is to bridge the gap between academic causal inference and practical, day-to-day data science.

Here is my realistic roadmap for the next 1-2 years.

## 1. Instrumental Variables (IV)
**Status: In Progress**

I am currently working on implementing **Instrumental Variables (IV2SLS)**. This is a critical feature because in many real-world scenarios, we have unobserved confounders that make standard methods (like DoubleML or Propensity Scores) biased.

- **Goal**: Allow users to specify an "instrument" (Z) that affects the treatment (T) but not the outcome (Y) directly.
- **Why**: To unlock causal inference in observational studies where we can't measure all confounders.

## 2. Causal Discovery
**Status: Planned**

Right now, CausalPilot assumes you know the causal graph. But often, we don't. I plan to integrate causal discovery algorithms to help users *learn* the graph from data.

- **PC Algorithm**: The classic constraint-based method.
- **NOTEARS**: A modern, optimization-based approach for learning DAGs.

I want to make this interactive: the algorithm suggests a graph, and the user refines it using their domain knowledge (perhaps via the Natural Language Interface).

## 3. Time Series Causal Inference
**Status: Planned**

Many business questions are time-dependent (e.g., "Did our marketing campaign last month cause sales to rise this month?").

- **Granger Causality**: A simple baseline.
- **Causal Impact**: A Bayesian structural time-series approach (similar to Google's CausalImpact).

I believe adding time-series support will make CausalPilot significantly more useful for marketing and finance use cases.

## 4. Uplift Modeling
**Status: Planned**

For marketing, we often care about *who* to target. Uplift modeling estimates the CATE (Conditional Average Treatment Effect) to identify persuadable customers.

- **Class Transformation Method**: A simple way to use any classifier for uplift.
- **Uplift Random Forests**: A specialized tree-based method.

I plan to add a dedicated `uplift` module to make these analyses easy.

---

*This roadmap reflects my current priorities. If you have other ideas or want to help with any of these, please open an issue!*
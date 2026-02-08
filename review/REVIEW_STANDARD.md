# Data Science & Engineering Review Standard

## 1. Purpose
This document defines the high-level criteria for code quality, reproducibility, and sustainability within this repository. It integrates industry-leading practices from Google’s Engineering Standards, Tim Hopper’s Data Science Guidelines, and the Reliable Data Science framework.

---

## 2. Engineering Foundations (Google Standards)
Code is an organizational asset. Every contribution should improve the overall "Code Health."

* **Small CLs (Change Lists):** Each Pull Request or Commit should have a single, clear purpose. Keep changes small (ideally ~100 lines) to reduce cognitive load and ensure thorough review.
* **Simplicity over Complexity:** Avoid "over-engineering." Choose the simplest solution that solves the current problem while remaining readable.
* **Self-Documenting Code:** Use intentional naming for variables and functions so the logic is clear without excessive commenting.

## 3. Data Science Reliability (Hopper & DrivenData)
Data science requires unique rigor due to stochastic behaviors and data-driven uncertainty.

* **Immutable Raw Data:** Raw data must never be overwritten. All transformations must be scripted and reproducible from the original source.
* **Notebook-to-Script Transition:** Use Jupyter Notebooks for exploration and visualization only. Transition core logic to modular `.py` files to enable unit testing and versioning.
* **Parameter Externalization:** Avoid hard-coding hyperparameters or file paths. Use configuration files (e.g., `config.yaml`) to manage experimental variables.

## 4. Interpretability & MLOps Governance
A model is only as good as its explainability and reliability in production.

* **XAI (Explainable AI):** Integrate SHAP or LIME to validate that model features align with domain expertise and ethical standards.
* **Data Validation:** Implement automated checks (e.g., Great Expectations) to detect data drift or schema inconsistencies early.
* **Fail Fast:** Design pipelines to fail explicitly when data quality or model performance drops below defined thresholds, preventing "silent failures."

---

## 5. Self-Review Checklist
- [ ] Does this change reduce or maintain system complexity?
- [ ] Is the core logic covered by automated tests or validation scripts?
- [ ] Are all magic numbers and parameters moved to a config file?
- [ ] Is the pipeline reproducible from raw data to final output?
- [ ] Is the model’s reasoning (feature importance/SHAP) visualized and documented?

---
*Reference: Google Engineering Practices, Tim Hopper's DS Review Guidelines, and Towards Data Science.*
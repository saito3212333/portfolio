Markdown
# EntropyInsight: Deciphering Model Uncertainty
### A Deep Dive into Cross-Entropy Dynamics via Titanic Survival Analysis

---

## 🚀 Project Overview
**EntropyInsight** is an interactive Streamlit application designed to demystify the inner workings of classification models. While most practitioners focus on "Accuracy," this project explores the **Loss Function**—the mathematical compass that guides a model during training.

By using the classic Titanic dataset, this app visualizes how adding informative features progressively reduces the model's **Cross-Entropy (Log-Loss)**, effectively "solving" the uncertainty within the data.



## 🧠 Theoretical Core: Cross-Entropy
At the heart of this project is the Cross-Entropy formula, which measures the "distance" between the true labels and the model's predicted probabilities:

$$H(P, Q) = -\sum_{i} P(i) \log Q(i)$$

- **$P(i)$**: The ground truth (Actual survival state).
- **$Q(i)$**: The predicted probability from the model.
- **The Logic**: By minimizing this value, the model reduces the "surprise" or "uncertainty" in its predictions. This application allows users to observe this reduction in real-time as the feature set expands.

## 🛠 Tech Stack
- **Environment & Package Manager**: [uv](https://github.com/astral-sh/uv) (Extremely fast, modern Python project manager)
- **ML Engine**: [LightGBM](https://lightgbm.readthedocs.io/) (Gradient Boosting Decision Trees)
- **Frontend Framework**: [Streamlit](https://streamlit.io/)
- **Data Science Stack**: Pandas, Seaborn, Scikit-learn, PyArrow

---

## 📊 Interactive Features
1. **Correlation Matrix**: Quantifying the statistical relationships between raw features and survival outcomes.
2. **Cross-Entropy Evolution**: An iterative training module that demonstrates how Log-Loss decreases as high-impact features are added.
3. **Model Interpretability**: Evaluating the final model through **Information Gain** (Cross-Entropy reduction) and Confusion Matrices.



---

## 💻 Installation & Usage

This project leverages `uv` for lightning-fast and reproducible environment setup.
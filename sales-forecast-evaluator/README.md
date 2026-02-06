# Sales Forecast Metrics Lab 📈

An interactive Streamlit dashboard designed to evaluate sales forecasting models through the lens of **practical business risk** and **operational performance**.

This project provides a "lab" environment where users can simulate forecast errors—such as systemic bias or massive outliers—and observe how different indicators respond.

---

## 🔬 The 4 Major Indicators
This lab implements the four essential metrics required to balance model quality with real-world risk management.

| Indicator | Business Role | Sensitivity & Focus |
| :--- | :--- | :--- |
| **WMAPE** | The **"Overall Pass Grade"** | Measures total error rate relative to total sales volume. |
| **RMSE** | The **"Fatal Risk Monitor"** | Highly sensitive to large outliers (imbalance risks). |
| **MAE** | The **"Standard Performance"** | Intuitive average drift for daily field reporting to managers. |
| **Max Error** | The **"Worst-Case Scenario"** | Identifies the largest prediction failure for anomaly detection. |

---

## 🛠️ Implementation Highlights
Following industry-standard best practices for data science development:

* **Optimized Calculations**: WMAPE is implemented as a high-performance, 1-line **NumPy** calculation.
* **Data Integrity**: A mandatory `dropna()` check is performed before any calculation to prevent sensor/system errors from biasing results.
* **Mathematical Precision**:
    * **WMAPE**: $$WMAPE = \frac{\sum |Actual - Predicted|}{\sum Actual}$$
    * **RMSE**: $$RMSE = \sqrt{\frac{1}{n} \sum (Actual - Predicted)^2}$$

---

## 💻 Tech Stack
* **Language**: Python 3.12+
* **Environment**: `uv`
* **Dashboard**: Streamlit
* **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib

---

## 🚀 Getting Started
To run this dashboard locally:

1. **Sync dependencies**:
   ```bash
   uv sync
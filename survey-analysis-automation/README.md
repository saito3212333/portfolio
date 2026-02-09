# Survey Data Analysis Automation Pipeline

A professional, modular, and memory-efficient Python pipeline for processing large-scale survey datasets. This project demonstrates best practices in data engineering: decoupling configuration from logic, optimizing memory, and ensuring data traceability.

---

## 🚀 Key Features

* **Modular Architecture**: Separation of logic (`.py`) and execution (`.ipynb`).
* **Config-Driven**: Management of cleaning rules and paths via `config.yaml`.
* **Memory Efficiency**: Usage of **Apache Parquet** and **Categorical dtypes** to minimize RAM footprint.
* **Traceability**: Automated logging of record loss at each cleaning step for analytical integrity.
* **Robust Pathing**: Cross-platform compatibility using `pathlib`.

---

## 🛠 Tech Stack

* **Python 3.12+**
* **Pandas**: Data manipulation and analysis.
* **PyYAML**: Configuration management.
* **PyArrow**: High-performance Parquet I/O engine.
* **Pathlib**: Object-oriented filesystem paths.

---

## 📁 Project Structure

```text
.
├── config.yaml          # Centralized analysis parameters and file paths
├── preprocess.py        # Reusable cleaning and optimization logic
├── analysis.ipynb       # Main execution notebook for EDA and results
├── data/
│   ├── raw/             # Source CSV files (Input)
│   └── processed/       # Optimized Parquet files (Output)
└── README.md            # Project documentation
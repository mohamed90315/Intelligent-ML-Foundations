# Assignment 12: Machine Learning & Data Analysis Algorithms

This repository contains a secure, refactored suite of machine learning algorithms (like Perceptrons) and dataset analysis tools developed for the AI course. The codebase has been extensively redesigned to ensure robust execution, environment separation, and professional-grade security.

## 🚀 Core Features
- **Environment Management**: Utilizes `.env` variables for sensitive configurations, preventing hardcoded absolute paths that leak internal system structures.
- **Robust Input Handling**: Replaced unsafe interactive prompts (`input()`) with precise command-line arguments using `argparse`.
- **Error Handling**: Implements `try-except` blocks to smoothly capture malformed data or missing files, averting silent failures or raw stack traces.
- **Headless Plotting**: Supports headless generation of ML plots for continuous integration and remote servers via `.env`.

## 🛠 Built With
- **Python 3.x**
- **NumPy**: Matrix and array computations.
- **Pandas**: CSV dataset ingestion and analysis.
- **Matplotlib**: Decision boundary visualization.
- **python-dotenv**: Environment configuration.

## 📁 Architecture & Directory Layout
```text
Assignment-12/
├── src/
│   ├── part1/
│   │   └── perceptron.py    # Refactored single perceptron classification algorithm
│   ├── part2/
│   │   └── kdd_analyzer.py  # Script for dataset ingestion and statistical summary
│   ├── q1/                  # Additional ML queries
│   └── q2/                  # Additional ML queries
├── data/
│   └── Sample_KDD.csv       # Local datastore for testing algorithms
├── docs/
│   └── Assignment2.pdf      # Original reference instructions
├── requirements.txt         # Explicit project dependencies
├── .env.example             # Template environment configurations
└── README.md                # Project documentation
```

## ⚙️ Prerequisites & Secure Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamed90315/AI-Assignment-12-ML.git
   cd Assignment-12
   ```
2. **Install dependencies**:
   It is recommended to run this inside a virtual environment (`venv`).
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to override `DATASET_PATH` or enable `HEADLESS` mode.*

## 📖 Usage

### Perceptron Classification
Run the classification algorithm by supplying custom weights, bias, and point coordinates via command-line arguments.

```bash
# Example with custom arguments
python src/part1/perceptron.py --weights "1.2,0.8" --bias 1.5 --points "0,0;1,1;0.5,0.5"
```

### KDD Dataset Analyzer
Run the data analyzer which automatically resolves relative paths or falls back on the `.env` settings.

```bash
# Reads Sample_KDD.csv from the default path or .env
python src/part2/kdd_analyzer.py

# Alternatively, pass an absolute path manually
python src/part2/kdd_analyzer.py --data_path "/absolute/path/to/data.csv"
```

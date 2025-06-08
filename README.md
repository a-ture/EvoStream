# 🧬 EvoStream: Concept Drift Detection in Viral Genomes

> A comprehensive and robust pipeline for tracking the real-time evolution of a virus by detecting concept drift in genomic data streams.

How can we identify the exact moment a new, dangerous viral variant emerges from a constant stream of genomic data? **EvoStream** is designed to answer this question. It simulates a realistic evolutionary scenario using SARS-CoV-2 variants (from Wuhan to Omicron) and benchmarks static (**batch**) versus adaptive (**online**) algorithms to determine the most effective approach for real-world surveillance.

---

## ✨ Key Features

EvoStream is built on three core pillars: a realistic experimental setup, advanced machine learning techniques, and a commitment to reproducible science.

#### A Robust Experimental Setup
* **Realistic Scenario**: Simulates a data stream with 120 real SARS-CoV-2 genomes, tracking the evolution across 4 major variants to create a complex, multi-drift environment.
* **Advanced Feature Engineering**: Utilizes a multi-scale data representation by combining vectors of multiple k-mer lengths (k=4, 5, 6), followed by a TF-IDF transformation to capture the unique genetic signatures of each variant.

#### Advanced Machine Learning
* **In-Depth Comparison**: Tests and compares 7 different algorithms, including anomaly detectors, change-point detectors, and an online clustering model to provide a holistic view of the available strategies.
* **Automatic Calibration**: Implements an automatic hyperparameter tuning phase on a dedicated validation set to find the optimal configuration for the online models, mirroring rigorous scientific methodology.

#### Comprehensive & Reproducible Analysis
* **Rich Analytics**: Generates a full suite of performance metrics (AUC, F1-Score, MCC, Detection Latency) and detailed visualizations, including PCA plots, time-series graphs, and heatmaps.
* **Research-Ready**: The code is structured to be extensible, reproducible, and provides a solid foundation for future research work.

---

## 🔬 How It Works

The EvoStream workflow is a systematic pipeline designed for clarity and reproducibility.

1.  **Data Acquisition**: The process begins by automatically downloading 120 complete genomic sequences for the four SARS-CoV-2 variants from the NCBI GenBank database.
2.  **Feature Creation**: Each genome is then transformed into a high-dimensional numerical vector. This is achieved by first counting k-mers of different lengths (4, 5, and 6) and then applying a TF-IDF transformation to weigh the most informative features.
3.  **Intelligent Hyperparameter Tuning**: Before the main analysis, the system uses the first two variants (Wuhan vs. Alpha) as a validation set. This step automatically tunes key parameters, like window size and quantile threshold, ensuring the online models are optimally configured.
4.  **Execution and Analysis**: With the best parameters identified, all seven algorithms are run on the entire data stream of 120 sequences for a comprehensive evaluation.
5.  **Output and Visualization**: Finally, the pipeline prints a detailed performance report to the console and saves a complete suite of publication-ready graphs for in-depth visual analysis in the `results_sars_cov_2_final` folder.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3 and the necessary libraries installed. You can set up your environment with a single command:
```bash
pip install biopython numpy scikit-learn matplotlib pandas seaborn scipy
```

### Execution

To launch the entire pipeline, simply execute the script from your terminal:
```bash
python virus_drift_detection.py
```
The script is fully automated, from data download to final result generation.

### What to Expect

Upon completion, a new folder named `results_sars_cov_2_final` will be created, containing:

* **PCA Plots**: Separate, high-resolution images for each algorithm, showing how the data is classified or clustered.
* **Temporal Plots**: Visualizations showing the trend of anomaly scores and adaptive thresholds over time.
* **Performance Curves**: ROC and Precision-Recall graphs to evaluate the trade-offs of the online models.
* **Matrices and Heatmaps**: Confusion matrices and a latency heatmap for a detailed quantitative analysis.
* **Console Output**: A complete report with all performance metrics for each model.

This project represents a rigorous application of data analytics principles to a real-world problem of significant impact, offering a robust framework for future research.

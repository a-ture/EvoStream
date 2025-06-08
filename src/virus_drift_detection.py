#!/usr/bin/env python3
"""
virus_drift_detection.py - Final Version for Examination

Pipeline to track the evolution of a virus (SARS-CoV-2) through its variants,
simulating a realistic concept drift scenario with an enlarged data sample.

Experimental Scenario:
- A data stream simulating the progressive appearance of SARS-CoV-2 variants is analyzed:
  1. Wuhan (Baseline) -> 2. Alpha -> 3. Delta -> 4. Omicron
- Each variant is represented by 30 genomic sequences, for a total of 120.

Objective:
- To compare batch and online models in their ability to detect and adapt.

Advanced Methodologies:
- Multi-scale feature engineering (multiple k-mers) and TF-IDF.
- ROBUST ONLINE THRESHOLD based on dynamic quantiles.
- Automatic hyperparameter calibration on a validation set, optimizing for F1-Score.
- Addition of a WEIGHTED ENSEMBLE based on individual model performance.
- Measurement of detection latency for online models.
- Analysis of Precision-Recall curves and a latency heatmap.
- Comparison of 7 algorithms (2 batch, 4 online anomaly, 1 online clustering).
- Generation of comprehensive performance metrics and detailed plots.

Prerequisites:
    pip install biopython numpy scikit-learn matplotlib pandas seaborn scipy

Usage:
    python virus_drift_detection.py
"""
import os
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from scipy.stats import trim_mean
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, adjusted_rand_score,
                             normalized_mutual_info_score, matthews_corrcoef, roc_curve, auc, v_measure_score,
                             precision_recall_curve, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# === EXPERT CONFIGURATION (LARGE SAMPLE) ===
VIRUS_VARIANTS = {
    "SARS-CoV-2 (Wuhan)": [
        "NC_045512.2", "MT291826.1", "MT291827.1", "MT291828.1", "MT263424.1",
        "MT259270.1", "MT259279.1", "MT263389.1", "MT281529.1", "MT281530.1",
        "LR757995.1", "LR757996.1", "LR757997.1", "LR757998.1", "LR757999.1",
        "LR758000.1", "LR758001.1", "LR758002.1", "LR758003.1", "MN908947.3",
        "MT345842.1", "MT345843.1", "MT345844.1", "MT345845.1", "MT345846.1",
        "MT345847.1", "MT345848.1", "MT345849.1", "MT345850.1", "MT345851.1"
    ],
    "SARS-CoV-2 (Alpha)": [
        "OV360434.1", "OV360435.1", "OV360436.1", "OV360437.1", "OV360438.1",
        "MW585539.1", "MW598433.1", "MW642251.1", "MW642252.1", "MW642253.1",
        "MW642254.1", "MW642255.1", "MW642256.1", "MW642257.1", "MW642258.1",
        "MW642259.1", "MW642260.1", "MW642261.1", "MW642262.1", "MW642263.1",
        "MZ356499.1", "MZ356500.1", "MZ356501.1", "MZ356502.1", "MZ356503.1",
        "MZ356504.1", "MZ356505.1", "MZ356506.1", "MZ356507.1", "MZ356508.1"
    ],
    "SARS-CoV-2 (Delta)": [
        "OQ829447.1", "OQ829448.1", "OQ829449.1", "OM487265.1", "OK058013.1",
        "MZ559986.1", "MZ559987.1", "MZ559988.1", "MZ559989.1", "MZ559990.1",
        "MZ559991.1", "MZ559992.1", "MZ559993.1", "MZ559994.1", "MZ559995.1",
        "MZ559996.1", "MZ559997.1", "MZ559998.1", "MZ559999.1", "MZ560000.1",
        "OK092523.1", "OK092524.1", "OK092525.1", "OK092526.1", "OK092527.1",
        "OK092528.1", "OK092529.1", "OK092530.1", "OK092531.1", "OK092532.1"
    ],
    "SARS-CoV-2 (Omicron)": [
        "OP011314.1", "OP011315.1", "OP011316.1", "OM287123.1", "ON939337.1",
        "ON939338.1", "ON939339.1", "ON939340.1", "ON939341.1", "ON939342.1",
        "ON939343.1", "ON939344.1", "ON939345.1", "ON939346.1", "ON939347.1",
        "ON939348.1", "ON939349.1", "ON939350.1", "ON939351.1", "ON939352.1",
        "OP073347.1", "OP073348.1", "OP073349.1", "OP073350.1", "OP073351.1",
        "OP073352.1", "OP073353.1", "OP073354.1", "OP073355.1", "OP073356.1"
    ]
}

KMER_SIZES = [4, 5, 6]
CONTAMINATION = 0.1
# Parameters for tuning
WINDOW_SIZES_TO_TUNE = [10, 15, 20]
QUANTILES_TO_TUNE = [0.85, 0.90, 0.95, 0.98]
# Default parameters (will be overwritten by tuning)
IPCA_COMPONENTS = 30
N_CLUSTERS = len(VIRUS_VARIANTS)
BATCH_SIZE = 15

# === PATHS ===
OUTPUT_DIR = "results_sars_cov_2_final"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
Entrez.email = "a.ture@studenti.unisa.it"


def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_sequences(variant_name, genbank_ids):
    out_path = os.path.join(DATA_DIR, f"{variant_name.replace(' ', '_')}.fasta")
    print(f"Downloading FASTA for {variant_name}...")
    try:
        handle = Entrez.efetch(db="nucleotide", id=','.join(genbank_ids), rettype="fasta", retmode="text")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        SeqIO.write(records, out_path, "fasta")
        print(f"Saved {len(records)} sequences.")
        return records
    except Exception as e:
        print(f"Error fetching sequences for {variant_name}: {e}")
        return []


def clean_sequences(records):
    clean_seqs = []
    for rec in records:
        seq = str(rec.seq).upper()
        seq = ''.join(b for b in seq if b in 'ACGT')
        if seq:
            clean_seqs.append(seq)
    return clean_seqs


def kmer_counts_matrix(sequences, k):
    all_kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    counts_matrix = np.zeros((len(sequences), len(all_kmers)), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) < k: continue
        counts = Counter(seq[j:j + k] for j in range(len(seq) - k + 1))
        for kmer, count in counts.items():
            if kmer in kmer_to_idx:
                counts_matrix[i, kmer_to_idx[kmer]] = count
    return counts_matrix


def build_multi_k_features(sequences, k_sizes):
    feature_matrices = []
    for k in k_sizes:
        print(f"  - Building k-mer counts for k={k}")
        feature_matrices.append(kmer_counts_matrix(sequences, k))
    return np.hstack(feature_matrices)


def run_batch_models(X_train, X_all):
    print("Running BATCH models...")
    iso = IsolationForest(contamination=CONTAMINATION, random_state=42, max_samples=0.5).fit(X_train)
    lof = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=CONTAMINATION).fit(X_train)
    return {"Isolation Forest": iso.predict(X_all), "Local Outlier Factor": lof.predict(X_all)}


def run_online_quantile_threshold(X_stream, score_func, window_size, quantile, **kwargs):
    """Generic function for online detection with quantile-based thresholding."""
    n_samples = X_stream.shape[0]
    scores = score_func(X_stream, window_size=window_size, **kwargs)
    thresholds = np.zeros(n_samples)
    anomalies = np.zeros(n_samples, dtype=bool)

    for i in range(1, n_samples):
        window_scores = scores[max(0, i - window_size):i]
        if len(window_scores) > 1:
            thresholds[i] = np.quantile(window_scores, quantile)
            if scores[i] > thresholds[i]:
                anomalies[i] = True
    return anomalies, scores, thresholds


def score_func_centroid(X_stream, window_size):
    n_samples = X_stream.shape[0]
    distances = np.zeros(n_samples)
    for i, x in enumerate(X_stream):
        window_data = X_stream[max(0, i - window_size):i]
        if window_data.shape[0] > 0:
            mean_vec = trim_mean(window_data, proportiontocut=0.1, axis=0)
            distances[i] = np.linalg.norm(x - mean_vec)
    return distances


def score_func_ipca(X_stream, window_size, n_components=IPCA_COMPONENTS):
    n_samples = X_stream.shape[0]
    recon_errors = np.zeros(n_samples)
    ipca = IncrementalPCA(n_components=n_components, batch_size=BATCH_SIZE)
    initial_train_size = max(n_components + 1, window_size)
    if initial_train_size >= n_samples: return recon_errors
    ipca.partial_fit(X_stream[:initial_train_size])
    for i in range(initial_train_size, n_samples):
        x = X_stream[i:i + 1]
        x_reconstructed = ipca.inverse_transform(ipca.transform(x))
        recon_errors[i] = np.linalg.norm(x - x_reconstructed)
        ipca.partial_fit(x)
    return recon_errors


def tune_online_hyperparameters(X_val, y_val, window_sizes, quantiles):
    print("\nTuning hyperparameters for online models on validation set (Wuhan vs Alpha)...")
    best_params = {'window_size': 0, 'quantile': 0}
    best_f1 = -1

    for ws in window_sizes:
        for q in quantiles:
            preds, _, _ = run_online_quantile_threshold(X_val, score_func_ipca, ws, q)
            # Optimize for F1-score on the drift (positive) class
            f1 = f1_score(y_val, preds, pos_label=True, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_params['window_size'] = ws
                best_params['quantile'] = q

    print(
        f"Best params found: Window Size = {best_params['window_size']}, Quantile = {best_params['quantile']} (F1-Score: {best_f1:.4f})")
    return best_params['window_size'], best_params['quantile']


def run_online_clustering(X_stream, n_clusters, batch_size):
    print("Running ONLINE Clustering (Mini-Batch K-Means)...")
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size, n_init='auto')
    cluster_labels = np.zeros(X_stream.shape[0], dtype=int)
    for i in range(0, X_stream.shape[0], batch_size):
        end = i + batch_size
        batch = X_stream[i:end]
        if len(batch) == 0: continue
        mbk.partial_fit(batch)
        cluster_labels[i:end] = mbk.predict(batch)
    return cluster_labels


def analyze_and_plot(X_all_tfidf, labels_true, variant_names, batch_preds, online_anomaly_preds, online_cluster_preds,
                     n_variant_counts, best_params):
    print("Generating final analysis and plots...")
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X_all_tfidf)
    print_summary_metrics(labels_true, variant_names, batch_preds, online_anomaly_preds, online_cluster_preds,
                          best_params, n_variant_counts)

    all_models = {**batch_preds, **{k: v['anomalies'] for k, v in online_anomaly_preds.items()},
                  "Mini-Batch K-Means": online_cluster_preds}
    cmap = plt.get_cmap('viridis', N_CLUSTERS)

    for name, preds in all_models.items():
        plt.figure(figsize=(14, 10))
        if name != "Mini-Batch K-Means":
            plot_preds = preds if name in batch_preds else [-1 if p else 1 for p in preds]
            for variant_idx, variant_name in enumerate(variant_names):
                mask = np.array(labels_true) == variant_idx
                plt.scatter(X2d[mask, 0], X2d[mask, 1], label=variant_name, color=cmap(variant_idx), alpha=0.8)
            anomaly_mask = np.array(plot_preds) == -1
            plt.scatter(X2d[anomaly_mask, 0], X2d[anomaly_mask, 1], facecolors='none', edgecolors='k', s=200,
                        linewidth=2.5, label='Anomaly Detected')
        else:
            for cluster_id in range(N_CLUSTERS):
                mask = preds == cluster_id
                plt.scatter(X2d[mask, 0], X2d[mask, 1], label=f'Online Cluster {cluster_id}', color=cmap(cluster_id),
                            alpha=0.8)
        plt.title(f"PCA Visualization - {name}", fontsize=18)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');
        plt.grid(True, linestyle='--')
        safe_name = name.replace(" ", "_").lower()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'1_pca_{safe_name}.png'))
        plt.close()

    fig, axes = plt.subplots(len(online_anomaly_preds), 1, figsize=(18, 22), sharex=True)
    fig.suptitle('Temporal Evolution of Online Algorithms', fontsize=20)
    drift_points = np.cumsum(n_variant_counts)
    for i, (name, data) in enumerate(online_anomaly_preds.items()):
        ax = axes[i]
        ax.plot(data['scores'], label='Score', color='dodgerblue')
        ax.plot(data['thresholds'], label='Adaptive Threshold (Quantile)', color='darkorange', linestyle='--')
        anomaly_idx = np.where(data['anomalies'])[0]
        ax.scatter(anomaly_idx, data['scores'][anomaly_idx], marker='o', facecolors='none', edgecolors='red', s=100,
                   linewidth=2, label='Drift Detected')
        for p_idx, p in enumerate(drift_points[:-1]): ax.axvline(x=p, color='red', linestyle='-.', linewidth=2,
                                                                 label=f'Actual Drift {p_idx + 1}' if i == 0 else "")
        ax.set_title(name, fontsize=14)
        ax.legend();
        ax.grid(True, linestyle='--');
        ax.set_ylabel("Score Value")
    axes[-1].set_xlabel("Sequence Index in Data Stream")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '2_online_temporal_comparison.png'))
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(18, 7))
    ax.plot(online_cluster_preds, drawstyle='steps-post', label='Assigned Cluster Label')
    ax.plot(labels_true, linestyle='--', color='gray', alpha=0.8, label='Ground Truth Label')
    for p in drift_points[:-1]: ax.axvline(x=p, color='red', linestyle='-.', linewidth=2)
    ax.set_title('Cluster Assignment Evolution (Mini-Batch K-Means)', fontsize=16)
    ax.set_xlabel("Sequence Index");
    ax.set_ylabel("Cluster ID")
    ax.set_yticks(range(N_CLUSTERS));
    ax.legend();
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_online_clustering_temporal.png'))
    plt.close()

    plot_roc_and_pr_curves(online_anomaly_preds, labels_true)
    run_latency_heatmap_analysis(X_all_tfidf, labels_true, n_variant_counts)


def plot_roc_and_pr_curves(online_anomaly_preds, labels_true):
    print("Generating ROC and Precision-Recall curves...")
    true_online_anomalies = np.array([l > 0 for l in labels_true])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Performance Curves for Online Detectors", fontsize=16)

    ax_roc = axes[0]
    for name, data in online_anomaly_preds.items():
        if "Ensemble" in name: continue
        fpr, tpr, _ = roc_curve(true_online_anomalies, data['scores'])
        auc_score = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax_roc.set_xlabel('False Positive Rate');
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve');
    ax_roc.legend();
    ax_roc.grid(True)

    ax_pr = axes[1]
    for name, data in online_anomaly_preds.items():
        if "Ensemble" in name: continue
        precision, recall, _ = precision_recall_curve(true_online_anomalies, data['scores'])
        ax_pr.plot(recall, precision, label=f"{name}")
    ax_pr.set_xlabel('Recall');
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve');
    ax_pr.legend();
    ax_pr.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, '5_performance_curves.png'))
    plt.close()


def run_latency_heatmap_analysis(X_all_tfidf, labels_true, n_variant_counts):
    print("\nRunning latency heatmap analysis...")
    latency_matrix = np.zeros((len(WINDOW_SIZES_TO_TUNE), len(QUANTILES_TO_TUNE)))

    for i, ws in enumerate(WINDOW_SIZES_TO_TUNE):
        for j, q in enumerate(QUANTILES_TO_TUNE):
            anomalies, _, _ = run_online_quantile_threshold(X_all_tfidf, score_func_ipca, ws, q)
            latencies = calculate_latency(labels_true, anomalies, n_variant_counts)
            valid_latencies = [l for l in latencies if isinstance(l, int)]
            avg_latency = np.mean(valid_latencies) if valid_latencies else np.inf
            latency_matrix[i, j] = avg_latency

    plt.figure(figsize=(12, 10))
    sns.heatmap(latency_matrix, annot=True, fmt='.2f', cmap='viridis_r',
                xticklabels=QUANTILES_TO_TUNE, yticklabels=WINDOW_SIZES_TO_TUNE)
    plt.xlabel("Threshold Quantile");
    plt.ylabel("Window Size")
    plt.title("Average Detection Latency Heatmap (Incremental PCA)", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, '6_latency_heatmap.png'))
    plt.close()


def calculate_latency(labels_true, anomalies, n_variant_counts):
    latencies = []
    drift_starts = np.cumsum(n_variant_counts)[:-1]
    for i, start in enumerate(drift_starts):
        end = start + n_variant_counts[i + 1]
        drift_zone_preds = anomalies[start:end]
        first_detection = np.where(drift_zone_preds)[0]
        latency = first_detection[0] + 1 if len(first_detection) > 0 else 'N/D'
        latencies.append(latency)
    return latencies


def print_summary_metrics(labels_true, variant_names, batch_preds, online_anomaly_preds, online_cluster_preds,
                          best_params, n_variant_counts):
    print("\n" + "=" * 80 + "\n||" + " " * 22 + "DETAILED PERFORMANCE METRICS" + " " * 22 + "||\n" + "=" * 80)
    print(
        f"\nOptimized Hyperparameters (from Validation Set): Window Size={best_params['window_size']}, Quantile={best_params['quantile']:.2f}\n")

    true_anomalies_batch = np.array([-1 if l > 0 else 1 for l in labels_true])
    true_online_anomalies = np.array([l > 0 for l in labels_true])

    print("\n" + "-" * 30 + " BATCH MODELS " + "-" * 34)
    for name, preds in batch_preds.items():
        print(f"\n--- {name} ---")
        print(classification_report(true_anomalies_batch, preds,
                                    target_names=[f'Drift ({N_CLUSTERS - 1} Variants)', 'Normal (Wuhan)'],
                                    zero_division=0))
        mcc = matthews_corrcoef(true_anomalies_batch, preds)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:+.4f}")

    print("\n" + "-" * 22 + " ONLINE ANOMALY/CHANGE-POINT MODELS " + "-" * 23)
    for name, data in online_anomaly_preds.items():
        print(f"\n--- {name} ---")
        preds = data['anomalies']
        scores = data['scores']
        print(classification_report(true_online_anomalies, preds,
                                    target_names=['Normal (Wuhan)', f'Drift ({N_CLUSTERS - 1} Variants)'],
                                    zero_division=0))
        mcc = matthews_corrcoef(true_online_anomalies, preds)
        fpr, tpr, _ = roc_curve(true_online_anomalies, scores);
        auc_score = auc(fpr, tpr)
        latency = calculate_latency(labels_true, preds, n_variant_counts)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:+.4f}")
        print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
        print(f"Detection Latency (per drift): {latency}")

    print("\n" + "-" * 27 + " ONLINE CLUSTERING MODEL " + "-" * 28)
    print(f"\n--- Mini-Batch K-Means ---")
    ari = adjusted_rand_score(labels_true, online_cluster_preds);
    nmi = normalized_mutual_info_score(labels_true, online_cluster_preds);
    v_measure = v_measure_score(labels_true, online_cluster_preds)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}");
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}");
    print(f"V-Measure: {v_measure:.4f}")

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(labels_true, online_cluster_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=[f'Cluster {i}' for i in range(N_CLUSTERS)],
                yticklabels=variant_names)
    plt.title('Confusion Matrix - Mini-Batch K-Means');
    plt.ylabel('True Label');
    plt.xlabel('Predicted Cluster')
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, '4_clustering_confusion_matrix.png'));
    plt.close()
    print("\n" + "=" * 80)


def main():
    setup_directories()

    variant_names = list(VIRUS_VARIANTS.keys())
    all_recs = [fetch_sequences(name, VIRUS_VARIANTS[name]) for name in variant_names]
    if any(not recs for recs in all_recs): return

    print("\nProcessing sequences and building k-mer count matrix...")
    all_seqs = [clean_sequences(recs) for recs in all_recs]
    if not any(all_seqs for s_list in all_seqs for s in s_list):
        print("No valid sequences found after cleaning. Aborting.");
        return

    n_variant_counts = [len(s) for s in all_seqs]
    labels_true = []
    for i, seqs in enumerate(all_seqs): labels_true.extend([i] * len(seqs))

    X_counts = build_multi_k_features(list(itertools.chain.from_iterable(all_seqs)), KMER_SIZES)

    print("\nApplying TF-IDF transformation...")
    tfidf_transformer = TfidfTransformer()
    X_all_tfidf = tfidf_transformer.fit_transform(X_counts).toarray()

    val_set_end_idx = n_variant_counts[0] + n_variant_counts[1]
    X_val = X_all_tfidf[:val_set_end_idx]
    y_val = np.array([l > 0 for l in labels_true[:val_set_end_idx]])
    best_ws, best_q = tune_online_hyperparameters(X_val, y_val, WINDOW_SIZES_TO_TUNE, QUANTILES_TO_TUNE)

    print("\nRunning final analysis with tuned hyperparameters...")
    X_train_tfidf = X_all_tfidf[:n_variant_counts[0]]
    batch_preds = run_batch_models(X_train_tfidf, X_all_tfidf)

    online_anomaly_preds = {}
    anom_centroid, scores_centroid, thresh_centroid = run_online_quantile_threshold(X_all_tfidf, score_func_centroid,
                                                                                    best_ws, best_q)
    online_anomaly_preds['TrimmedMean Centroid'] = {'anomalies': anom_centroid, 'scores': scores_centroid,
                                                    'thresholds': thresh_centroid}

    anom_ipca, scores_ipca, thresh_ipca = run_online_quantile_threshold(X_all_tfidf, score_func_ipca, best_ws, best_q)
    online_anomaly_preds['Incremental PCA'] = {'anomalies': anom_ipca, 'scores': scores_ipca, 'thresholds': thresh_ipca}

    auc_centroid = auc(*roc_curve(y_val, score_func_centroid(X_val, best_ws))[:2])
    auc_ipca = auc(*roc_curve(y_val, score_func_ipca(X_val, best_ws))[:2])
    weights = np.array([auc_centroid, auc_ipca])
    weights /= weights.sum()
    print(f"\nEnsemble weights (Centroid, IPCA): {weights[0]:.2f}, {weights[1]:.2f}")

    scaler = MinMaxScaler()
    combined_scores = (scaler.fit_transform(scores_centroid.reshape(-1, 1)) * weights[0] +
                       scaler.fit_transform(scores_ipca.reshape(-1, 1)) * weights[1]).flatten()
    ensemble_anomalies, _, ensemble_thresholds = run_online_quantile_threshold(X_all_tfidf,
                                                                               lambda x, **kw: combined_scores, best_ws,
                                                                               best_q)
    online_anomaly_preds['Weighted Ensemble'] = {'anomalies': ensemble_anomalies, 'scores': combined_scores,
                                                 'thresholds': ensemble_thresholds}

    online_cluster_preds = run_online_clustering(X_all_tfidf, N_CLUSTERS, BATCH_SIZE)

    analyze_and_plot(X_all_tfidf, labels_true, variant_names, batch_preds, online_anomaly_preds, online_cluster_preds,
                     n_variant_counts, {'window_size': best_ws, 'quantile': best_q})

    print(f"\nOperation completed! Check the '{OUTPUT_DIR}' folder for results.")


if __name__ == '__main__':
    main()

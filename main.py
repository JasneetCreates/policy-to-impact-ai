"""
=============================================================================
Policy-to-Impact Intelligence System: Master Execution Pipeline
=============================================================================
From Government Policy to Ground Reality: An Evidence-Driven AI Framework
for Monitoring Public Infrastructure Delivery and Detecting Emerging Service Failures.

Runs all 12 Machine Learning syllabus algorithms end-to-end:
1. Linear Regression
2. Batch, Stochastic, and Mini-Batch Gradient Descent
3. Polynomial Regression & Learning Curves
4. Ridge and Lasso Regularization
5. Logistic Regression & Decision Boundaries
6. KNN Classification on MNIST
7. Naive Bayes with ROC & Precision-Recall curves
8. Linear and Non-Linear SVM (Polynomial & RBF kernels)
9. CART Decision Tree (Gini Impurity vs. Entropy)
10. Random Forest & Feature Importance
11. Boosting (AdaBoost & Gradient Boosting)
12. K-Means Spatial Hotspot Clustering
=============================================================================
"""

import os
import sys
import time

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_loader import generate_curated_datasets
from stage1_regression import run_all_regression_experiments
from stage2_citizen_nlp import run_all_classification_clustering_experiments
from mnist_knn_benchmark import evaluate_knn_on_mnist
from stage3_triangulation_ml import run_all_advanced_ml_experiments


def print_banner():
    banner = r"""
  ____       _ _                 _          ___                            _   
 |  _ \ ___ | (_) ___ _   _     | |_ ___   |_ _|_ __ ___  _ __   __ _  ___| |_ 
 | |_) / _ \| | |/ __| | | |____| __/ _ \   | || '_ ` _ \| '_ \ / _` |/ __| __|
 |  __/ (_) | | | (__| |_| |____| || (_) |  | || | | | | | |_) | (_| | (__| |_ 
 |_|   \___/|_|_|\___|\__, |     \__\___/  |___|_| |_| |_| .__/ \__,_|\___|\__|
                      |___/                              |_|                   
               POLICY-TO-IMPACT INTELLIGENCE SYSTEM
    Evidence-Driven AI Framework for Public Infrastructure Auditing
    """
    print(banner)


def generate_final_intelligence_dossier():
    """Produce the final policy-to-impact intelligence summary report."""
    import pandas as pd
    data_path = os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv")
    df = pd.read_csv(data_path)

    print("\n" + "=" * 80)
    print("FINAL AUDIT DOSSIER: FLAGGED INFRASTRUCTURE CORRIDORS")
    print("=" * 80)

    discrepancies = df[df["is_discrepancy"] == 1].copy()
    print(f"Total Monitored Projects : {len(df)}")
    print(f"Total Discrepancies Flagged : {len(discrepancies)} ({len(discrepancies)/len(df)*100:.1f}% of portfolio)")
    print("-" * 80)

    for idx, row in discrepancies.head(8).iterrows():
        print(f"Project ID : {row['project_id']} | City: {row['city']}")
        print(f"Title      : {row['project_name']}")
        print(f"Budget     : Sanctioned: ₹{row['sanctioned_budget_cr']} Cr | Disbursed: {row['disbursed_funds_pct']}%")
        print(f"Timeline   : Sched: {row['scheduled_duration_months']}m | Actual: {row['actual_duration_months']}m (Delay: +{row['delay_months']}m)")
        print(f"Paper Claim: Reported Status: '{row['reported_status']}' ({row['reported_progress_pct']}% completed)")
        print(f"Ground Truth: Defect Density: {row['ground_defect_density']} /km | Citizen Complaints: {row['citizen_complaint_count']}")
        print(f"FLAGGED AS : [!] {row['discrepancy_type']}")
        print(f"Action     : Depute field verification team to inspect corridor coordinates ({row['latitude']}, {row['longitude']}).")
        print("-" * 80)


def list_generated_figures():
    """List all figures generated in outputs/figures."""
    fig_dir = os.path.join("outputs", "figures")
    print("\n" + "=" * 80)
    print("GENERATED EXPERIMENTAL FIGURES & VISUALIZATIONS (outputs/figures/)")
    print("=" * 80)
    if os.path.exists(fig_dir):
        files = sorted(os.listdir(fig_dir))
        for f in files:
            if f.endswith(".png"):
                full_path = os.path.join(fig_dir, f)
                size_kb = os.path.getsize(full_path) / 1024
                print(f"  [x] {f:48s} ({size_kb:6.1f} KB)")
    print("=" * 80)


def main():
    start_time = time.time()
    print_banner()

    print("\n[PHASE 1] Initializing and Verifying Multi-Tier Datasets...")
    generate_curated_datasets()

    print("\n[PHASE 2] Executing Regression Suite (Syllabus Items 1, 2, 3, 4)...")
    run_all_regression_experiments()

    print("\n[PHASE 3] Executing Classification & Clustering Suite (Syllabus Items 5, 7, 12)...")
    run_all_classification_clustering_experiments()

    print("\n[PHASE 4] Executing MNIST Digit KNN Classification Benchmark (Syllabus Item 6)...")
    evaluate_knn_on_mnist()

    print("\n[PHASE 5] Executing Advanced Classifiers & Ensembles (Syllabus Items 8, 9, 10, 11)...")
    run_all_advanced_ml_experiments()

    generate_final_intelligence_dossier()
    list_generated_figures()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Complete Policy-to-Impact ML Pipeline executed successfully in {elapsed:.2f} seconds.\n")


if __name__ == "__main__":
    main()

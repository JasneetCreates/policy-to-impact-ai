"""
Script to generate a comprehensive, self-contained Google Colab Jupyter Notebook
for the Policy-to-Impact Intelligence System.
"""

import json
import os


def build_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip().splitlines(keepends=True)
        })

    def code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip().splitlines(keepends=True)
        })

    md("""
# Policy-to-Impact Intelligence System
### *From Government Policy to Ground Reality: An Evidence-Driven AI Framework for Monitoring Public Infrastructure Delivery and Detecting Emerging Service Failures*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasneetCreates/policy-to-impact-ai/blob/main/notebooks/01_policy_to_impact_complete_demo.ipynb)
**GitHub Repository:** [JasneetCreates/policy-to-impact-ai](https://github.com/JasneetCreates/policy-to-impact-ai)

---

### Project Overview
This project presents an evidence-based Machine Learning framework for public infrastructure monitoring. It triangulates three critical tiers of data:
1. **Policy Commitments (Macro):** Budget sanctions, timelines, and physical corridor specifications (National Infrastructure Pipeline / State Budgets).
2. **Delivery Records (Meso):** Tenders, reported completion %, and contractor milestone logs (CPPP / State Procurement).
3. **Citizen Realities (Micro):** Unstructured grievances, hazard signals, and ground defect concentrations.

#### Core Predictive & Triangulation Engines:
1. **Timeline & Cost Escalation Modeling:** Multi-variable regularized regression and optimization convergence analysis.
2. **Citizen Signal Mining & NLP:** Text categorization of complaints and precision-recall tradeoff analysis.
3. **Geospatial Hotspot Clustering:** Unsupervised GPS clustering identifying physical service failure corridors.
4. **Triangulation Discrepancy Detection:** Support Vector Machines, Decision Trees, Random Forest, and Gradient Boosting detecting ghost completions and severe delays.
    """)

    code("""
# Setup Environment and Dependencies
import sys
import os

if 'google.colab' in str(get_ipython()):
    print("Running on Google Colab! Cloning repository...")
    !git clone https://github.com/JasneetCreates/policy-to-impact-ai.git
    %cd policy-to-impact-ai
    !pip install -q -r requirements.txt
else:
    print("Running in local environment.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 10

print("Environment verified!")
    """)

    md("""
---
## Stage 1: Multi-Tier Data Ingestion & Triangulation
We load authentic public infrastructure projects, delivery logs, and citizen grievance streams.
    """)

    code("""
from src.data_loader import generate_curated_datasets

df_projects, df_complaints = generate_curated_datasets()
print(f"Projects shape: {df_projects.shape}")
print(f"Complaints shape: {df_complaints.shape}")

df_projects[['project_id', 'city', 'project_name', 'sanctioned_budget_cr', 'length_km', 'delay_months', 'is_discrepancy']].head()
    """)

    md("""
---
## Stage 2: Timeline & Cost Escalation Predictive Modeling
Evaluating multi-variable linear regression on infrastructure delays and optimization convergence.
    """)

    code("""
from src.stage1_regression import load_regression_data, evaluate_linear_regression, compare_gradient_descent

X, y, df = load_regression_data()
eval_res = evaluate_linear_regression(X, y)
bgd, sgd, mbgd = compare_gradient_descent(X, y, n_epochs=120, lr=0.01)

from IPython.display import Image
Image(filename='outputs/figures/01_gradient_descent_convergence.png')
    """)

    md("""
### Non-Linearity Modeling & Regularization Paths
Evaluating polynomial learning curves and L1/L2 regularization feature selection.
    """)

    code("""
from src.stage1_regression import analyze_polynomial_and_learning_curves, analyze_ridge_and_lasso

poly_models = analyze_polynomial_and_learning_curves(df)
best_ridge, best_lasso = analyze_ridge_and_lasso(X, y)

Image(filename='outputs/figures/02_polynomial_regression_learning_curves.png')
    """)

    md("""
---
## Stage 3: Citizen Signal Mining & Geospatial Hotspot Clustering
Classifying unstructured citizen complaints into civic hazard categories with ROC/PR curves and clustering GPS coordinates.
    """)

    code("""
from src.stage2_citizen_nlp import evaluate_logistic_regression, evaluate_naive_bayes, evaluate_kmeans_clustering

clf_log, scaler = evaluate_logistic_regression()
nb_model, vectorizer = evaluate_naive_bayes()
final_kmeans, optimal_k = evaluate_kmeans_clustering()

Image(filename='outputs/figures/06_kmeans_hotspots_elbow_silhouette.png')
    """)

    md("""
---
## Stage 4: Multi-Source Triangulation & Discrepancy Detection Engine
Benchmarking Support Vector Machines, interpretable CART Decision Trees, Random Forest (MDI feature ranking), and Gradient Boosting.
    """)

    code("""
from src.stage3_triangulation_ml import load_triangulated_features, evaluate_svm_kernels, evaluate_decision_tree, evaluate_random_forest, evaluate_boosting_and_comparison

X_full, y_full, df_full, feature_names = load_triangulated_features()
svm_res = evaluate_svm_kernels(df_full)
tree_g, tree_e = evaluate_decision_tree(X_full, y_full, feature_names)
rf, sorted_features, sorted_importances = evaluate_random_forest(X_full, y_full, feature_names)
df_metrics = evaluate_boosting_and_comparison(X_full, y_full, feature_names)

Image(filename='outputs/figures/10_random_forest_feature_importance.png')
    """)

    md("""
---
## Final Synthesis: Automated Triangulation Intelligence Dossier
The AI framework connects policy commitments, delivery records, and citizen complaints to flag discrepancy cases requiring field audits.
    """)

    code("""
from main import generate_final_intelligence_dossier

generate_final_intelligence_dossier()
    """)

    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "01_policy_to_impact_complete_demo.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[OK] Successfully built Colab Notebook at: {nb_path}")


if __name__ == "__main__":
    build_notebook()

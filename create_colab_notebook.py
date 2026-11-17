"""
Script to generate a comprehensive, self-contained Google Colab Jupyter Notebook
for the Policy-to-Impact Intelligence System covering all 12 syllabus algorithms.
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

    # Cell 1: Title and Colab Badge
    md("""
# Policy-to-Impact Intelligence System
### *From Government Policy to Ground Reality: An Evidence-Driven AI Framework for Monitoring Public Infrastructure Delivery and Detecting Emerging Service Failures*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasneetCreates/policy-to-impact-ai/blob/main/notebooks/01_policy_to_impact_complete_demo.ipynb)
**GitHub Repository:** [JasneetCreates/policy-to-impact-ai](https://github.com/JasneetCreates/policy-to-impact-ai)

---

### Project Overview & Syllabus Alignment
This project demonstrates an evidence-based Machine Learning framework for public infrastructure auditing. It cross-examines:
1. **Policy Commitments (Macro):** Budget sanctions, timelines, and road lengths.
2. **Delivery Records (Meso):** Tenders, reported completion %, contractor delays.
3. **Citizen Realities (Micro):** Grievances, accident hazard signals, and ground defect density.

#### The 12 Syllabus Machine Learning Algorithms Covered:
1. **Linear Regression:** Infrastructure delay prediction & performance evaluation ($R^2$, MSE, RMSE, MAE).
2. **Gradient Descent Convergence:** Comparative analysis of Batch, Stochastic (SGD), and Mini-Batch GD.
3. **Polynomial Regression & Learning Curves:** Modeling compounding delays; analyzing underfitting vs. overfitting.
4. **Ridge ($L_2$) & Lasso ($L_1$) Regularization:** Coefficient shrinkage & Lasso automatic feature selection.
5. **Logistic Regression:** Binary classification of high-risk discrepancy projects with 2D Decision Boundary.
6. **KNN Classification on MNIST:** Handwritten digit recognition with Confusion Matrix, Precision, and Recall.
7. **Naïve Bayes Text Classifier:** Citizen complaint classification with ROC Curve and Precision–Recall tradeoff.
8. **Support Vector Machines (SVM):** Linear, Polynomial, and RBF kernels compared with support vector counts.
9. **Decision Tree (CART Algorithm):** Visual audit tree comparing Gini Impurity vs. Entropy (Information Gain).
10. **Random Forest & Feature Importance:** MDI ranking of the top mathematical drivers of infrastructure failure.
11. **Boosting Ensemble:** AdaBoost & Gradient Boosting benchmarked against Single Tree and Random Forest.
12. **K-Means Spatial Hotspot Clustering:** Unsupervised GPS clustering of citizen complaints with Elbow & Silhouette analysis.
    """)

    # Cell 2: Setup and Clone / Local check
    code("""
# Setup Environment and Dependencies
import sys
import os

# If running in Google Colab, clone the repository
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

# Inline plotting setup
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 10

print("Environment verified!")
    """)

    # Cell 3: Data Ingestion
    md("""
---
## Stage 0: Multi-Tier Data Ingestion & Dataset Preparation
We load real-world inspired public infrastructure projects, delivery logs, citizen grievances, and the MNIST benchmark dataset.
    """)

    code("""
from src.data_loader import generate_curated_datasets

df_projects, df_complaints, df_mnist = generate_curated_datasets()
print(f"Projects shape: {df_projects.shape}")
print(f"Complaints shape: {df_complaints.shape}")
print(f"MNIST sample shape: {df_mnist.shape}")

df_projects[['project_id', 'city', 'project_name', 'sanctioned_budget_cr', 'length_km', 'delay_months', 'is_discrepancy']].head()
    """)

    # Cell 4: Linear Regression & Gradient Descent
    md("""
---
## 1. Linear Regression (Syllabus Item 1)
Evaluating project delay predictions using $R^2$, MSE, RMSE, and MAE.
    """)

    code("""
from src.stage1_regression import load_regression_data, evaluate_linear_regression

X, y, df = load_regression_data()
eval_res = evaluate_linear_regression(X, y)
    """)

    # Cell 5: Gradient Descent Convergence
    md("""
---
## 2. Gradient Descent Convergence: Batch vs. SGD vs. Mini-Batch (Syllabus Item 2)
Comparing the convergence behavior and stability of Batch, Stochastic, and Mini-Batch Gradient Descent from scratch.
    """)

    code("""
from src.stage1_regression import compare_gradient_descent

bgd, sgd, mbgd = compare_gradient_descent(X, y, n_epochs=120, lr=0.01)

# Display the generated figure
from IPython.display import Image
Image(filename='outputs/figures/01_gradient_descent_convergence.png')
    """)

    # Cell 6: Polynomial Regression & Learning Curves
    md("""
---
## 3. Polynomial Regression & Learning Curves (Syllabus Item 3)
Analyzing underfitting (Degree 1) vs. optimal fit (Degree 2) vs. overfitting (Degree 5) using learning curves.
    """)

    code("""
from src.stage1_regression import analyze_polynomial_and_learning_curves

poly_models = analyze_polynomial_and_learning_curves(df)
Image(filename='outputs/figures/02_polynomial_regression_learning_curves.png')
    """)

    # Cell 7: Ridge & Lasso Regularization
    md("""
---
## 4. Ridge & Lasso Regularization (Syllabus Item 4)
Evaluating coefficient shrinkage ($L_2$) and automatic feature selection ($L_1$) across regularization strengths $\alpha$.
    """)

    code("""
from src.stage1_regression import analyze_ridge_and_lasso

best_ridge, best_lasso = analyze_ridge_and_lasso(X, y)
Image(filename='outputs/figures/03_ridge_lasso_regularization.png')
    """)

    # Cell 8: Logistic Regression
    md("""
---
## 5. Logistic Regression: Binary Discrepancy Classification & Decision Boundary (Syllabus Item 5)
Classifying high-risk projects and plotting the 2D decision boundary contour.
    """)

    code("""
from src.stage2_citizen_nlp import evaluate_logistic_regression

clf_log, scaler = evaluate_logistic_regression()
Image(filename='outputs/figures/04_logistic_regression_decision_boundary.png')
    """)

    # Cell 9: Naïve Bayes on Citizen Complaints
    md("""
---
## 7. Naïve Bayes Classifier on Citizen Grievances (Syllabus Item 7)
Classifying unstructured complaints and analyzing the ROC Curve and Precision–Recall Tradeoff.
    """)

    code("""
from src.stage2_citizen_nlp import evaluate_naive_bayes

nb_model, vectorizer = evaluate_naive_bayes()
Image(filename='outputs/figures/05_naive_bayes_roc_precision_recall.png')
    """)

    # Cell 10: K-Means Hotspot Clustering
    md("""
---
## 12. K-Means Spatial Hotspot Clustering (Syllabus Item 12)
Discovering geographic failure hotspots using the Elbow Method and Silhouette Analysis.
    """)

    code("""
from src.stage2_citizen_nlp import evaluate_kmeans_clustering

final_kmeans, optimal_k = evaluate_kmeans_clustering()
Image(filename='outputs/figures/06_kmeans_hotspots_elbow_silhouette.png')
    """)

    # Cell 11: KNN on MNIST
    md("""
---
## 6. KNN Classification on the MNIST Dataset (Syllabus Item 6)
Evaluating handwritten digit recognition across neighborhood sizes ($k$) with Confusion Matrix, Precision, and Recall.
    """)

    code("""
from src.mnist_knn_benchmark import evaluate_knn_on_mnist

final_knn, cm, report = evaluate_knn_on_mnist()
Image(filename='outputs/figures/07_knn_mnist_confusion_matrix.png')
    """)

    # Cell 12: SVM Kernels
    md("""
---
## 8. Linear & Non-Linear SVM: Polynomial & RBF Kernels (Syllabus Item 8)
Comparing Linear, Polynomial, and RBF kernels on project discrepancy classification.
    """)

    code("""
from src.stage3_triangulation_ml import load_triangulated_features, evaluate_svm_kernels

X_full, y_full, df_full, feature_names = load_triangulated_features()
svm_res = evaluate_svm_kernels(df_full)
Image(filename='outputs/figures/08_svm_linear_poly_rbf_comparison.png')
    """)

    # Cell 13: Decision Tree CART Gini vs Entropy
    md("""
---
## 9. Decision Tree (CART): Gini Impurity vs. Entropy (Syllabus Item 9)
Constructing interpretable decision trees for government auditors and comparing splitting criteria.
    """)

    code("""
from src.stage3_triangulation_ml import evaluate_decision_tree

tree_g, tree_e = evaluate_decision_tree(X_full, y_full, feature_names)
Image(filename='outputs/figures/09_decision_tree_cart_gini_vs_entropy.png')
    """)

    # Cell 14: Random Forest & Feature Importance
    md("""
---
## 10. Random Forest & Feature Importance Ranking (Syllabus Item 10)
Measuring Out-Of-Bag score and ranking the top mathematical drivers of public service failure.
    """)

    code("""
from src.stage3_triangulation_ml import evaluate_random_forest

rf, sorted_features, sorted_importances = evaluate_random_forest(X_full, y_full, feature_names)
Image(filename='outputs/figures/10_random_forest_feature_importance.png')
    """)

    # Cell 15: Boosting Ensemble Comparison
    md("""
---
## 11. Boosting Technique & Ensemble Benchmark (Syllabus Item 11)
Benchmarking AdaBoost and Gradient Boosting against a single CART tree and Random Forest.
    """)

    code("""
from src.stage3_triangulation_ml import evaluate_boosting_and_comparison

df_metrics = evaluate_boosting_and_comparison(X_full, y_full, feature_names)
Image(filename='outputs/figures/11_ensemble_models_comparison.png')
    """)

    # Cell 16: Final Intelligence Dossier
    md("""
---
## Final Synthesis: Automated Triangulation Intelligence Dossier
The AI framework connects policy commitments, delivery records, and citizen complaints to flag discrepancy cases requiring field audits.
    """)

    code("""
from main import generate_final_intelligence_dossier

generate_final_intelligence_dossier()
    """)

    # Write notebook file
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "01_policy_to_impact_complete_demo.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[OK] Successfully built Colab Notebook at: {nb_path}")


if __name__ == "__main__":
    build_notebook()

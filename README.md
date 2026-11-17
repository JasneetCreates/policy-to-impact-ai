# Policy-to-Impact Intelligence System

### *From Government Policy to Ground Reality: An Evidence-Driven AI Framework for Monitoring Public Infrastructure Delivery and Detecting Emerging Service Failures*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasneetCreates/policy-to-impact-ai/blob/main/notebooks/01_policy_to_impact_complete_demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

---

## 1. Executive Summary & Research Motivation

Public infrastructure investments (roads, bridges, water supply corridors, and drainage systems) involve billions of dollars in taxpayer expenditure. However, traditional governance monitoring suffers from **severe information asymmetry**:
* **Macro-Level Policy Commitments:** Ambitious schemes, budget sanctions, and deadlines declared on paper.
* **Meso-Level Delivery Records:** Contractor progress logs, work orders, and completion certificates that frequently claim "100% completion" on departmental portals.
* **Micro-Level Ground Realities:** Citizen complaints, localized service disruptions, and news investigations revealing craters, unpaved trenches, and persistent service failures.

The **Policy-to-Impact Intelligence System** bridges this divide. It is an evidence-based Machine Learning framework that ingests data from all three tiers, resolves project entities across non-standardized records, and detects discrepancies (e.g., **"Ghost Completions"** and **"Milestone Slippages"**) without making speculative accusations—providing public auditors with verifiable evidence dossiers.

```
+---------------------------------------------------------------------------------------------------+
|                               POLICY-TO-IMPACT INTELLIGENCE PIPELINE                              |
+---------------------------------------------------------------------------------------------------+
|  [Tier 1: Policy Commitments]   ---> Budget Sanctions, Corridor Length, Deadlines (NIP / Portals) |
|                                                      |                                            |
|  [Tier 2: Delivery Auditing]    ---> Tenders, Contractor Delays, Claimed Completion % (CPPP)      |
|                                                      |                                            |
|  [Tier 3: Ground-Level Sensing] ---> Citizen Complaints, GPS Defect Hotspots, Civic News Feeds   |
|                                                      |                                            |
|                                                      v                                            |
|  [Machine Learning Triangulation]                                                                 |
|  - Regression Suite: Timeline & Cost Overrun Modeling                                            |
|  - Classification & Clustering: Hazard NLP & Spatial Hotspots (HDBSCAN / K-Means)                 |
|  - Ensemble Anomaly Detection: SVM, CART Trees, Random Forest (MDI), Boosting                     |
|                                                      |                                            |
|                                                      v                                            |
|  [Actionable Intelligence Output]                                                                 |
|  - Flagged Discrepancy Dossiers (Ghost Completions vs. Verified Projects)                         |
|  - Non-Accusatory Field Audit Prompts for Government Inspectors                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Academic Research Questions (RQ1 – RQ4)

* **RQ1 (Policy Extraction & Regression Modeling):** Can machine learning model project timelines and cost overruns from initial policy declarations, corridor lengths, and contractor histories?
* **RQ2 (Cross-Source Record Linkage):** Can algorithms reliably link disparate government project titles to tender notices and informal citizen complaints?
* **RQ3 (Citizen Signal Mining & Clustering):** Can NLP classifiers and unsupervised spatial clustering isolate genuine service failure hotspots from noisy public grievances?
* **RQ4 (Multi-Source Triangulation & Ablation Value):** Does combining Policy + Delivery + Citizen feedback detect project delivery failures significantly better than monitoring each source in isolation?

---

## 3.  Machine Learning Algorithms Covered

The system implements and benchmarks the following **12 foundational Machine Learning algorithms**:

| # | s Algorithm | System Component & Role | Primary Evaluation Metric / Output |
| :-: | :--- | :--- | :--- |
| **1** | **Linear Regression** | Predict project completion delay (months) from budget, length, junctions, and contractor track record. | $R^2$, MSE (`4.7686`), RMSE (`2.1837m`), MAE (`1.7609m`). |
| **2** | **Batch, Stochastic (SGD) & Mini-Batch Gradient Descent** | Optimization convergence analysis comparing full batch vs. single sample vs. mini-batch updates. | Epoch vs. MSE loss convergence curves on infrastructure dataset. |
| **3** | **Polynomial Regression & Learning Curves** | Modeling non-linear project cost escalations; analyzing underfitting (Degree 1) vs. overfitting (Degree 5). | Learning curves (Training vs. Validation RMSE over sample sizes). |
| **4** | **Ridge ($L_2$) & Lasso ($L_1$) Regularization** | Multicollinearity reduction & automatic feature selection across $\alpha \in [10^{-2}, 10^3]$. | Regularization path plots; Lasso drives non-essential features to zero. |
| **5** | **Logistic Regression** | Binary classification of high-risk project discrepancies based on defect density & funds paid. | Confusion matrix, Classification report, and **2D Decision Boundary**. |
| **6** | **KNN Classification on MNIST** | Handwritten digit recognition benchmark across neighborhood sizes ($k \in [1, 15]$). | **`98.44%` Accuracy**, **10x10 Confusion Matrix**, Precision & Recall. |
| **7** | **Naïve Bayes Text Classifier** | Classify unstructured citizen complaints into urgent hazards vs. routine grievances. | **ROC Curve (AUC = 0.707)** and **Precision–Recall Tradeoff Curve**. |
| **8** | **Linear & Non-Linear SVM (Poly & RBF Kernels)** | Classify project discrepancy risk across varying kernel margin geometries. | Linear (`83.3%`), Poly (`88.9%`), **RBF (`100.0%` Accuracy)**. |
| **9** | **Decision Tree (CART Algorithm)** | Human-auditable rule tree comparing **Gini Impurity** vs. **Entropy (Info Gain)**. | Visual decision tree splits (`max_depth=3`, Accuracy `94.44%`). |
| **10** | **Random Forest & Feature Importance** | Ensemble bagging model; compute Mean Decrease in Impurity (MDI) feature ranking. | OOB Score (`90.48%`), Accuracy (`94.44%`), **Feature Importance Chart**. |
| **11** | **Boosting (AdaBoost & Gradient Boosting)** | Sequential ensemble boosting to maximize discrepancy detection on edge cases. | Gradient Boosting achieves **`94.44%` Accuracy** and **`0.981` ROC-AUC**. |
| **12** | **K-Means Spatial Hotspot Clustering** | Unsupervised GPS clustering of citizen complaints to discover road failure corridors. | **Elbow Method (Inertia curve)** and **Silhouette Score ($k=7$)**. |

---

## 4. Real-World Data Sources

The project utilizes data reflecting authentic Indian public infrastructure monitoring:
1. **Tier 1 (Policy):** National Infrastructure Pipeline (NIP) reports (Ministry of Finance / Dept. of Economic Affairs) and municipal master plans (e.g., Pune Municipal Corporation ₹1,866 Cr road corridor plan).
2. **Tier 2 (Delivery):** Central Public Procurement Portal (CPPP) and state e-tender notices (e.g., Maharashtra and Punjab e-procurement logs) tracking contractor awards, milestone delays, and disbursed funds.
3. **Tier 3 (Citizen Ground Signals):** Municipal grievance logs (PMC Care, CPGRAMS public records) and investigative local civic news reports (e.g., *Hindustan Times* Ludhiana dug-up road reports, *Times of India* civic beats).
4. **Benchmark Dataset (MNIST):** 1,797 samples of 64 pixel features for the dedicated KNN digit recognition lab module.

---

## 5. Experimental Results & Visualizations

All experiments save publication-quality figures to `outputs/figures/`:

| Figure File | Description | Key Insight |
| :--- | :--- | :--- |
| `01_gradient_descent_convergence.png` | Batch vs. SGD vs. Mini-Batch GD | Mini-batch (size 8) converges rapidly while avoiding stochastic oscillations. |
| `02_polynomial_regression_learning_curves.png` | Poly fits (d=1, 2, 5) & Learning Curves | Degree 2 captures quadratic compounding project complexity without overfitting. |
| `03_ridge_lasso_regularization.png` | Ridge vs. Lasso Regularization Paths | Lasso automatically eliminates `budget` and `length` to highlight contractor delays. |
| `04_logistic_regression_decision_boundary.png` | 2D Logistic Decision Boundary | Clean separation of discrepancy projects at $P=0.5$ probability contour. |
| `05_naive_bayes_roc_precision_recall.png` | Naïve Bayes ROC & PR Tradeoff | Demonstrates the trade-off between sensitivity and false alarms in civic complaints. |
| `06_kmeans_hotspots_elbow_silhouette.png` | Elbow Method, Silhouette & Hotspots | Optimal $k=7$ reveals distinct geographical road failure hotspots. |
| `07_knn_mnist_confusion_matrix.png` | MNIST 10x10 Confusion Matrix & Digits | Achieves `98.44%` test accuracy on handwritten digit recognition. |
| `08_svm_linear_poly_rbf_comparison.png` | SVM Linear vs. Poly vs. RBF Kernels | RBF kernel achieves `100%` separation on the non-linear failure boundary. |
| `09_decision_tree_cart_gini_vs_entropy.png` | CART Tree: Gini vs. Entropy | Both criteria produce identical, interpretable audit rules at depth 3. |
| `10_random_forest_feature_importance.png` | Random Forest MDI Feature Ranking | `ground_defect_density` (35.7%) and `complaint_count` (24.8%) are top failure predictors. |
| `11_ensemble_models_comparison.png` | Single Tree vs. RF vs. Boosting | Gradient Boosting achieves the highest ROC-AUC (`0.981`). |

---

## 6. How to Run

### Option A: Interactive Google Colab (One-Click)
Open the fully populated, ready-to-run notebook directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasneetCreates/policy-to-impact-ai/blob/main/notebooks/01_policy_to_impact_complete_demo.ipynb)

### Option B: Run Locally in Terminal

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JasneetCreates/policy-to-impact-ai.git
   cd policy-to-impact-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the complete master pipeline:**
   ```bash
   python main.py
   ```
   *Runs all 12 algorithms, prints metrics and audit dossiers, and saves all 11 plots to `outputs/figures/` in ~6 seconds.*

4. **Run individual syllabus modules:**
   ```bash
   python src/stage1_regression.py         # Items 1, 2, 3, 4
   python src/stage2_citizen_nlp.py        # Items 5, 7, 12
   python src/mnist_knn_benchmark.py       # Item 6
   python src/stage3_triangulation_ml.py   # Items 8, 9, 10, 11
   ```

---

## 7. Repository Structure

```
policy-to-impact-ai/
├── README.md                          <- Comprehensive project documentation & syllabus guide
├── FLOW_PAPER.md                      <- Academic research paper flow specification & blueprint
├── requirements.txt                   <- Python package dependencies (scikit-learn, numpy, pandas, matplotlib)
├── main.py                            <- Master pipeline runner executing all 12 algorithms
├── create_colab_notebook.py           <- Programmatic builder for the Google Colab notebook
├── data/
│   ├── raw/
│   │   ├── infrastructure_projects.json    <- 60 infrastructure project records
│   │   ├── citizen_complaints.csv          <- 1,105 citizen grievance records
│   │   └── mnist_digits_sample.csv         <- 1,797 MNIST digit samples
│   └── processed/
│       └── policy_delivery_ground_triangulated.csv <- Triangulated multi-tier dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 <- Data generation, ingestion & feature engineering
│   ├── stage1_regression.py           <- Items 1, 2, 3, 4 (Linear, Batch/SGD/Mini-batch GD, Poly, Ridge/Lasso)
│   ├── stage2_citizen_nlp.py          <- Items 5, 7, 12 (Logistic Reg, Naïve Bayes, K-Means Hotspots)
│   ├── mnist_knn_benchmark.py         <- Item 6 (KNN on MNIST with Confusion Matrix)
│   └── stage3_triangulation_ml.py     <- Items 8, 9, 10, 11 (SVM, Decision Tree, Random Forest, Boosting)
├── notebooks/
│   └── 01_policy_to_impact_complete_demo.ipynb  <- Self-contained Google Colab interactive notebook
└── outputs/
    └── figures/                       <- All 11 generated publication-grade figures (.png)
```

---

## 8. License & Attribution

This project is open-source under the [MIT License](LICENSE).  
Designed as an academic and practical AI framework for evidence-based governance and public infrastructure auditing.

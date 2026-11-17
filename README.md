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
|  - Timeline & Cost Escalation Modeling: Multi-variable Regularized Regression                    |
|  - Citizen Signal Mining & NLP: Civic Hazard Categorization & Precision-Recall Optimization       |
|  - Geospatial Pattern Recognition: K-Means Spatial Defect Hotspot Clustering                     |
|  - Discrepancy Detection: Random Forest (MDI Feature Importance) & Gradient Boosting (0.981 AUC)  |
|                                                      |                                            |
|                                                      v                                            |
|  [Actionable Intelligence Output]                                                                 |
|  - Interactive Web Dashboard & Geospatial Map (dashboard.html & Streamlit app.py)                 |
|  - Flagged Discrepancy Dossiers (Ghost Completions vs. Verified Projects)                         |
|  - Non-Accusatory Field Audit Prompts for Technical Inspection Teams                             |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Research Questions (RQ1 – RQ4)

* **RQ1 (Policy Extraction & Regression Modeling):** Can machine learning model project timelines and cost overruns from initial policy declarations, corridor lengths, and contractor histories?
* **RQ2 (Cross-Source Record Linkage):** Can algorithms reliably link disparate government project titles to tender notices and informal citizen complaints?
* **RQ3 (Citizen Signal Mining & Clustering):** Can NLP classifiers and unsupervised spatial clustering isolate genuine service failure hotspots from noisy public grievances?
* **RQ4 (Multi-Source Triangulation & Ablation Value):** Does combining Policy + Delivery + Citizen feedback detect project delivery failures significantly better than monitoring each source in isolation?

---

## 3. Core Machine Learning Architecture

The system utilizes specialized algorithms tailored to each analytical layer of the problem:

| Analytical Layer | Machine Learning Model | Functional Role in System | Key Metric / Performance |
| :--- | :--- | :--- | :--- |
| **Timeline Prediction** | **Multi-Variable Linear Regression & Gradient Descent** | Models project completion delays from budget, corridor length, junctions, and past contractor track records. | MSE: `4.7686`, RMSE: `2.1837 months`, MAE: `1.7609 months`. |
| **Cost Non-Linearity** | **Polynomial Regression with Learning Curves** | Models compounding non-linear cost escalations without overfitting. | Quadratic model captures compounding physical complexity ($R^2 = 0.99+$). |
| **Feature Selection** | **Ridge ($L_2$) & Lasso ($L_1$) Regularization** | Prevents multi-collinearity and eliminates non-essential variables to isolate true delay drivers. | Lasso eliminates budget/length to isolate contractor track record and junctions. |
| **Binary Triage** | **Logistic Regression (2D Decision Boundary)** | Separates high-risk discrepancy corridors based on funds disbursed vs. ground defect density. | **`83.33%` Accuracy**, **`86.0%` Discrepancy Recall** at $P=0.5$ boundary. |
| **Civic NLP** | **Multinomial Naïve Bayes Classifier** | Categorizes unstructured complaints into hazard categories (*Potholes*, *Dug-up Trenches*, *Waterlogging*). | **ROC-AUC: `0.707`**, Average Precision (PR-AUC): **`0.888`**. |
| **Geospatial Hotspots**| **K-Means Spatial Density Clustering** | Groups complaint GPS coordinates into distinct geographical failure corridors with centroid markers. | Optimal $k=7$ identified via **Elbow Curve** and **Silhouette Score (`0.6262`)**. |
| **Failure Triangulation**| **Random Forest Classifier (100 Trees)** | Classifies corridors into *Verified Delivery*, *Milestone Slippage*, and *Suspected Ghost Completion*. | **`94.44%` Accuracy**, **`90.48%` Out-Of-Bag Score**, MDI Feature Importance. |
| **Ensemble Detection**| **Gradient Boosting & AdaBoost** | Sequential boosting optimizing detection on complex, non-linear edge cases. | **`94.44%` Accuracy**, **`0.981` ROC-AUC** against single tree baseline. |

---

## 4. Real-World Data Sources

The project utilizes multi-tier data reflecting authentic public infrastructure monitoring:
1. **Tier 1 (Policy):** National Infrastructure Pipeline (NIP) reports (Ministry of Finance / Dept. of Economic Affairs) and municipal master plans (e.g., Pune Municipal Corporation ₹1,866 Cr road corridor plan).
2. **Tier 2 (Delivery):** Central Public Procurement Portal (CPPP) and state e-tender notices (e.g., Maharashtra and Punjab e-procurement logs) tracking contractor awards, milestone delays, and disbursed funds.
3. **Tier 3 (Citizen Ground Signals):** Municipal grievance logs (PMC Care, CPGRAMS public records) and investigative local civic news reports (e.g., *Hindustan Times* Ludhiana dug-up road reports, *Times of India* civic beats).

---

## 5. Visualizing Web Dashboard

The project includes an interactive web application:

### Standalone Web Dashboard (`dashboard.html`)
Open [dashboard.html](file:///c:/Users/ASUS/OneDrive/Documents/GitHub/policy-to-impact-ai/dashboard.html) directly in any web browser without running commands:
* **Interactive GIS Map (Leaflet.js):** Corridors across Pune, Ludhiana, Bengaluru, Delhi, Ahmedabad, and Mumbai color-coded by audit status (🔴 Ghost Completion, 🟡 Milestone Slippage, 🟢 Clean).
* **Live Filtering:** Instant filtering by city jurisdiction and audit status.
* **Interactive Dossiers:** Click on any project to view tripartite evidence (Policy vs Tender vs Citizen Reality) and the AI recommendation prompt.
* **Model Visualization Gallery:** Displays all empirical figures and evaluation plots.

### Live Streamlit Application (`app.py`)
To run the reactive Python server:
```bash
streamlit run app.py
```

---

## 6. How to Run Locally

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

4. **Run individual modules:**
   ```bash
   python src/stage1_regression.py         # Timeline & Cost Escalation Modeling
   python src/stage2_citizen_nlp.py        # Citizen NLP & Geospatial Hotspot Clustering
   python src/stage3_triangulation_ml.py   # Triangulation & Discrepancy Detection Engine
   ```

---

## 7. License & Attribution

This project is open-source under the [MIT License](LICENSE).  
Designed as an academic and practical AI framework for evidence-based governance and public infrastructure auditing.

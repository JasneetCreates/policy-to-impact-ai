# Flow Paper: Policy-to-Impact Intelligence System

**Research Title:**  
*From Government Policy to Ground Reality: An Evidence-Driven AI Framework for Monitoring Public Infrastructure Delivery and Detecting Emerging Service Failures*

**Principal Architecture & System Flow Specification**  
**Repository:** [policy-to-impact-ai](https://github.com/JasneetCreates/policy-to-impact-ai)  
**Status:** Architecture Specification & Project Flow Blueprint (Pre-Implementation Stage)

---

## 1. Executive Summary & Research Motivation

Governments allocate trillions of dollars to public infrastructure (e.g., roads, drainage, water supply, bridges). However, a chronic information asymmetry persists between:
1. **Macro-Level Policy Commitments:** High-level strategic targets, budgetary sanctions, and scheme timelines.
2. **Meso-Level Project Delivery:** Departmental progress reports, contractor work orders, and completion certificates.
3. **Micro-Level Ground Reality:** Lived experiences, chronic service disruptions, and complaints logged by citizens on civic portals and news beats.

Traditional monitoring systems rely on self-reported departmental auditing, creating blind spots for "ghost completions" (projects marked completed on paper while remaining defunct or sub-standard on the ground), milestone slippages, and localized infrastructure collapse.

**The Policy-to-Impact Intelligence System** bridges this chasm. By combining **Advanced Natural Language Processing (NLP)**, **Multimodal/Generative AI**, and **Machine Learning**, the framework automatically extracts commitments, links disparate administrative records via entity resolution, mines citizen feedback, and flags statistically significant discrepancies with an auditable evidence chain.

```
+---------------------------------------------------------------------------------------------------+
|                               POLICY-TO-IMPACT INTELLIGENCE PIPELINE                              |
+---------------------------------------------------------------------------------------------------+
|  [Stage 1: Policy Ingestion]                                                                      |
|  Government PDFs / NIP / State Budgets  ---> [LayoutParser + LLM/NER] ---> Structured Commitments |
|                                                                                    |              |
|  [Stage 2: Delivery Auditing]                                                      v              |
|  E-Tenders / Work Orders / Progress Logs ---> [Entity Resolution & Linking] <------+              |
|                                                      |                                            |
|                                                      v                                            |
|  [Stage 3: Ground Sensing]             [Unified Project State]                                    |
|  Civic Grievances / News / Social     ---> [NLP Classif. + HDBSCAN]                               |
|                                                      |                                            |
|                                                      v                                            |
|  [Stage 4: Triangulation & Discrepancy Engine]                                                    |
|  Discrepancy Matrix (Paper vs Ground) ---> [Anomaly Detection + Cross-Encoder Verification]        |
|                                                      |                                            |
|                                                      v                                            |
|  [Stage 5: Explainable Dashboard & Evidence Dossier]                                              |
|  Streamlit UI + GIS Heatmaps + Auditable Evidence Dossiers (RAG Citations)                        |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Research Questions (RQ1 – RQ4)

The framework is architected around four core research questions:

* **RQ1 (Policy Information Extraction):** Can multimodal and transformer-based NLP accurately parse non-standardized, unstructured government policy documents to extract structured entities (targets, budgets, deadlines, locations, and nodal agencies) with high precision and recall?
* **RQ2 (Cross-Document Entity Resolution):** Can semantic embedding and record-linkage algorithms reliably link policy commitments to e-procurement tenders, work orders, and completion reports across inconsistent project naming conventions?
* **RQ3 (Citizen Signal & Civic Anomaly Mining):** Can NLP classification and unsupervised spatio-temporal clustering of citizen complaints isolate reliable signals of recurring public service failures from noisy public text?
* **RQ4 (Multi-Source Triangulation & Ablation Value):** Does an evidence-linked triangulation model (Policy + Delivery + Citizen) detect project failures and discrepancies with significantly higher precision and actionable context than single-source or dual-source baselines?

---

## 3. End-to-End System Workflow

### Stage 1: Policy Intelligence (Understanding Commitments)
* **Inputs:** Government policy PDFs, National Infrastructure Pipeline (NIP) reports, municipal budget sanction documents, urban development vision plans.
* **Processing:**
  1. Document ingestion and OCR fallback using `PyMuPDF` and `pytesseract`.
  2. Document layout parsing: separating tables, metadata, and body paragraphs.
  3. Structured Extraction Engine: Transformer-based Named Entity Recognition (NER) + Few-Shot Structured Output extraction via LLM/Regex anchors.
* **Output Schema (`PolicyCommitment`):**
  - `policy_id`, `project_name`, `domain` (Roads/Water/Drainage), `sanctioned_budget_inr`, `target_deadline`, `milestones`, `nodal_department`, `target_geography`.

### Stage 2: Delivery Auditing (Tracking Documented Execution)
* **Inputs:** E-procurement tenders (CPPP/State portal formats), contractor work orders, contractor payment records, completion certificates.
* **Processing:**
  1. Extraction of tender value, contractor details, agreed completion dates, and reported milestone completion status.
  2. Temporal tracking: calculating planned duration vs reported duration.
* **Output Schema (`DeliveryRecord`):**
  - `tender_id`, `linked_project_name`, `contractor_name`, `awarded_amount_inr`, `commencement_date`, `contracted_completion_date`, `reported_completion_status` (In-Progress / Delayed / Completed), `milestones_claimed`.

### Stage 3: Ground-Level Civic Sensing (Mining Citizen Reality)
* **Inputs:** Citizen grievances (Municipal grievance portals, CPGRAMS public records), local civic news beats, community reports.
* **Processing:**
  1. **Text Preprocessing & Translation:** Cleaning municipal transliterated complaints and multi-language normalization.
  2. **Topic & Issue Classification:** Zero-shot / Fine-tuned Transformer classifier (e.g. Potholes, Waterlogging, Dug-up Road, Sewer Overflow, Traffic Bottleneck).
  3. **Severity & Sentiment Scoring:** Detecting acute distress, accident hazard, and repeated failure signals.
  4. **Spatio-Temporal Hotspot Discovery:** Density-based spatial/temporal clustering (`HDBSCAN` / `DBSCAN`) to group individual complaints into distinct physical ground events.
* **Output Schema (`GroundSignal`):**
  - `signal_id`, `issue_category`, `severity_score` (0.0–1.0), `location_entity`, `geo_coordinates`, `timestamp`, `cluster_id`, `recurring_frequency`.

### Stage 4: Multi-Source Entity Resolution & Triangulation Engine
* **Entity Resolution (Connecting the Disparate Sources):**
  - Hybrid matching using **Dense Semantic Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) + **Fuzzy String Matching** (Levenshtein/Jaro-Winkler) + **Geographic Proximity Filtering**.
  - Maps `PolicyCommitment <---> DeliveryRecord <---> GroundSignal Cluster`.
* **Discrepancy Detection Matrix:**
  The system computes four primary discrepancy flags:
  1. **Flag A: "Ghost Completion" Alert:** Delivery record reports `Status: 100% Completed`, but Ground Signal cluster reports persistent, acute failures (e.g., "road dug up", "unpaved crater", "water supply disconnected") in the post-completion window.
  2. **Flag B: "Severe Milestone Slippage":** Tender/Policy deadline passed or imminent, but zero progress documented while citizen complaints cite abandoned construction sites.
  3. **Flag C: "Fiscal-Physical Decoupling":** >80% funds disbursed, but physical completion or ground verification indicates minimal actual construction.
  4. **Flag D: "Repeated Defect Post-Maintenance":** Road repaired under warranty/Defect Liability Period (DLP), but citizen complaints re-emerge within 30-90 days.
* **Discrepancy Confidence Score ($S_{disc}$):**
  $$S_{disc} = w_1 \cdot \text{StatusMismatch} + w_2 \cdot \text{ComplaintIntensity} + w_3 \cdot \text{TemporalPersistence} + w_4 \cdot \text{SemanticRelevance}$$

### Stage 5: Explainable Dashboard & Evidence Dossier
* **Interactive Streamlit Web Dashboard:**
  - **Macro Policy View:** Budget allocation vs delivery status across districts/sectors.
  - **Interactive GIS Map:** Projects color-coded by Discrepancy Risk (Green: Verified; Amber: Milestone Slippage; Red: Suspected Ghost/Failed Delivery).
  - **Evidence Dossier Card:** Displays the full chain of evidence:
    - *Policy promise:* Quote from budget document with page citation.
    - *Tender claim:* Work order number, contractor, and reported completion date.
    - *Citizen evidence:* Timeline of timestamped citizen complaints and local news headlines.
    - *AI Recommendation:* Non-accusatory, objective inspection prompt (e.g., *"Recommended Action: Depute third-party field auditor to verify kilometer stretch 12 to 16 on Sector 4 Road."*).

---

## 4. Machine Learning & Algorithm Matrix

| Pipeline Component | Primary Algorithm / Model | Baseline / Fallback Algorithm | Evaluation Metric |
| :--- | :--- | :--- | :--- |
| **Doc Layout & Text** | PyMuPDF + LayoutParser | Pytesseract OCR | Character Error Rate (CER), Extraction Yield |
| **Policy/Tender NER** | Fine-tuned DeBERTa / Few-Shot LLM | Regex + Rule-based Spans | Precision, Recall, Macro-F1 |
| **Entity Resolution** | Sentence-Transformers (`all-MiniLM-L6-v2`) + Cosine Thresholding | Jaccard / Fuzzy Token Sort Ratio | Linkage Precision, Recall, Pairwise F1 |
| **Complaint Classification** | Transformer Classifier (RoBERTa / DistilBERT) | TF-IDF + Logistic Regression / SVM | Accuracy, Macro-F1 across 6 Civic Classes |
| **Spatial-Temporal Clustering** | HDBSCAN (Haversine distance + Time window) | Grid-based Geo-hashing | Silhouette Score, Cluster Purity |
| **Discrepancy Detection** | Hybrid Multi-Vector Discrepancy Scoring | Rule-based Heuristic Matrix | Precision@K, ROC-AUC against Ground Truth |
| **Evidence Dossier Generation** | RAG (Retrieval-Augmented Generation) with strict citation verification | Template-based Narrative Formatter | Citation Accuracy, Faithfulness Score |

---

## 5. Data Strategy & Selected Domain

### Selected Primary Domain: Road Construction & Urban Mobility Infrastructure
* **Why this domain?**
  1. High public expenditure and frequent policy declarations.
  2. Clear, verifiable physical milestones (tender award, commencement, surfacing, completion).
  3. Plentiful citizen feedback signals (potholes, dug-up stretches, asphalt degradation, incomplete junctions).
  4. Real-world precedents (e.g., NIP volume, Pune Municipal Corporation road master plans, Ludhiana road/water project sites).

### Data Ingestion Strategy:
1. **Real-world Official Documents:** Public NIP project briefs, state/municipal road infrastructure project sheets, e-procurement tender notices.
2. **Real-world Civic Complaints & News:** Scraped and open-access municipal grievance reports (e.g., PMC Care grievance records, public municipal portals), regional civic news archives (Hindustan Times, Times of India civic reporting).
3. **Calibrated Synthetic & Semi-Supervised Corpus:** For rigorous evaluation of RQ1–RQ4, we also construct an auditable evaluation benchmark containing positive matches, delayed projects, verified completions, and synthetic "ghost" edge cases to calculate standard ML metrics (ROC-AUC, F1, Precision@K).

---

## 6. Execution Roadmap & Phased Implementation

To maintain complete transparency and precision, the project is structured in 6 sequential phases. **Each phase requires explicit user sign-off before proceeding to the next.**

```
Phase 1: Environment, Directory Scaffolding & Data Architecture Setup [CURRENT]
   │
   ▼ (User Approval Required)
Phase 2: Stage 1 (Policy Engine) & Stage 2 (Delivery Auditing Engine)
   │
   ▼ (User Approval Required)
Phase 3: Stage 3 (Citizen Ground Sensing NLP & Spatio-Temporal Clustering Engine)
   │
   ▼ (User Approval Required)
Phase 4: Stage 4 (Entity Resolution & Multi-Source Triangulation Discrepancy Engine)
   │
   ▼ (User Approval Required)
Phase 5: Evaluation Benchmark & Comparative Ablation Experiments (RQ1 - RQ4)
   │
   ▼ (User Approval Required)
Phase 6: Interactive Streamlit Dashboard, Colab Notebook & Research Paper / Report
```

---

## 7. Dual Execution Model: Local Repository + Google Colab

The project is engineered to work seamlessly in both modes:
1. **Modular Local Package:** Production-ready Python structure inside this Git repository (`src/`, `data/`, `tests/`, `dashboard/`) with command-line runners and Streamlit dashboard.
2. **Google Colab Notebooks:** Self-contained, zero-setup Colab notebooks (`notebooks/01_policy_to_impact_demo.ipynb`) allowing anyone to run the full pipeline in the cloud with free GPU support.

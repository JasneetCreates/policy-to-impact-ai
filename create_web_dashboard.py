"""
Generates a standalone, beautiful HTML5 + Tailwind CSS + Leaflet.js interactive web dashboard.
Can be opened directly in any web browser without needing any terminal or server commands.
"""

import os
import json
import pandas as pd


def generate_html_dashboard():
    data_path = os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv")
    complaints_path = os.path.join("data", "raw", "citizen_complaints.csv")
    
    df_proj = pd.read_csv(data_path)
    df_comp = pd.read_csv(complaints_path)

    projects_json = df_proj.to_json(orient="records")
    complaints_json = df_comp.head(100).to_json(orient="records")

    total_projects = len(df_proj)
    total_budget = df_proj["sanctioned_budget_cr"].sum()
    total_ghost = len(df_proj[df_proj["discrepancy_type"] == "Suspected Ghost Completion"])
    total_delayed = len(df_proj[df_proj["discrepancy_type"] == "Severe Milestone Slippage"])
    total_verified = len(df_proj[df_proj["is_discrepancy"] == 0])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Policy-to-Impact Intelligence System | Public Infrastructure Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Leaflet GIS Map -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC;
        }}
        .custom-scrollbar::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb {{
            background: #CBD5E1;
            border-radius: 4px;
        }}
        #map {{
            height: 480px;
            border-radius: 12px;
            z-index: 10;
        }}
    </style>
</head>
<body class="text-slate-800">

    <!-- Top Navigation Header -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-lg bg-gradient-to-tr from-blue-700 to-indigo-600 flex items-center justify-center text-white text-xl shadow-md">
                    <i class="fa-solid fa-landmark"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold text-slate-900 leading-tight">Policy-to-Impact Intelligence System</h1>
                    <p class="text-xs text-slate-500 font-medium">Evidence-Driven AI Framework for Public Infrastructure Auditing</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
                    <span class="w-2 h-2 mr-1.5 bg-emerald-500 rounded-full animate-pulse"></span> Active Intelligence Feed
                </span>
                <a href="https://github.com/JasneetCreates/policy-to-impact-ai" target="_blank" class="px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-slate-900 text-white hover:bg-slate-800 transition flex items-center gap-1.5 shadow">
                    <i class="fa-brands fa-github text-sm"></i> GitHub Repo
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        <!-- Executive Summary Banner -->
        <div class="bg-gradient-to-r from-blue-900 via-indigo-900 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-xl relative overflow-hidden">
            <div class="relative z-10 max-w-3xl">
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 text-xs font-semibold mb-3 border border-blue-400/20">
                    <i class="fa-solid fa-satellite-dish"></i> Triangulation Engine: Macro Policy + Meso Tenders + Micro Citizen Realities
                </div>
                <h2 class="text-2xl sm:text-3xl font-extrabold tracking-tight">From Government Policy to Ground Reality</h2>
                <p class="mt-2 text-sm sm:text-base text-slate-300 leading-relaxed">
                    Automatically cross-examining public road infrastructure schemes against contractor e-procurement progress logs and timestamped citizen grievance signals to flag emerging service failures and <b>"Ghost Completions"</b> with verifiable evidence dossiers.
                </p>
            </div>
            <div class="absolute right-0 top-0 bottom-0 w-1/3 bg-cover opacity-10 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-400 to-transparent"></div>
        </div>

        <!-- 5 Key KPI Metric Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition">
                <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Monitored Portfolio</p>
                <p class="text-2xl font-bold text-slate-900 mt-2">{total_projects} Corridors</p>
                <p class="text-xs text-slate-500 mt-1 flex items-center gap-1"><i class="fa-solid fa-map-location-dot text-blue-500"></i> Across 6 Metros</p>
            </div>
            <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition">
                <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Sanctioned Budget</p>
                <p class="text-2xl font-bold text-slate-900 mt-2">₹{total_budget:,.0f} Cr</p>
                <p class="text-xs text-slate-500 mt-1 flex items-center gap-1"><i class="fa-solid fa-money-bill-transfer text-emerald-500"></i> NIP / State Sanctions</p>
            </div>
            <div class="bg-white p-5 rounded-xl border border-rose-200 bg-rose-50/40 shadow-sm hover:shadow-md transition">
                <p class="text-xs font-semibold text-rose-600 uppercase tracking-wider">Ghost Completions</p>
                <p class="text-2xl font-bold text-rose-700 mt-2">{total_ghost} Flagged</p>
                <p class="text-xs text-rose-600 mt-1 flex items-center gap-1"><i class="fa-solid fa-triangle-exclamation"></i> Paper 100% != Reality</p>
            </div>
            <div class="bg-white p-5 rounded-xl border border-amber-200 bg-amber-50/40 shadow-sm hover:shadow-md transition">
                <p class="text-xs font-semibold text-amber-700 uppercase tracking-wider">Milestone Slippages</p>
                <p class="text-2xl font-bold text-amber-800 mt-2">{total_delayed} Delayed</p>
                <p class="text-xs text-amber-700 mt-1 flex items-center gap-1"><i class="fa-solid fa-clock-rotate-left"></i> Delay Delta > 6 Months</p>
            </div>
            <div class="bg-white p-5 rounded-xl border border-emerald-200 bg-emerald-50/40 shadow-sm hover:shadow-md transition">
                <p class="text-xs font-semibold text-emerald-700 uppercase tracking-wider">Verified Clean</p>
                <p class="text-2xl font-bold text-emerald-800 mt-2">{total_verified} Clean</p>
                <p class="text-xs text-emerald-700 mt-1 flex items-center gap-1"><i class="fa-solid fa-circle-check"></i> Low Defect Density</p>
            </div>
        </div>

        <!-- Interactive GIS Map & Priority Alert Section -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div>
                        <h3 class="text-base font-bold text-slate-900 flex items-center gap-2">
                            <i class="fa-solid fa-earth-asia text-indigo-600"></i> Geospatial Infrastructure Delivery Map
                        </h3>
                        <p class="text-xs text-slate-500">Interactive corridor GPS coordinates color-coded by AI audit risk</p>
                    </div>
                    <div class="flex items-center gap-2 text-xs">
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-100 text-rose-800 font-semibold"><span class="w-2 h-2 rounded-full bg-rose-600"></span> Ghost</span>
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-semibold"><span class="w-2 h-2 rounded-full bg-amber-600"></span> Delayed</span>
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-semibold"><span class="w-2 h-2 rounded-full bg-emerald-600"></span> Clean</span>
                    </div>
                </div>
                <div id="map"></div>
            </div>

            <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                <h3 class="text-base font-bold text-slate-900 flex items-center gap-2">
                    <i class="fa-solid fa-clipboard-check text-rose-600"></i> Priority Audit Cases
                </h3>
                <p class="text-xs text-slate-500">Corridors flagged by the Triangulation Anomaly Detector for immediate human field inspection.</p>
                
                <div class="space-y-3 max-h-[420px] overflow-y-auto pr-1 custom-scrollbar" id="priorityList">
                </div>
            </div>
        </div>

        <!-- Filterable Project Records Table -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h3 class="text-base font-bold text-slate-900">Infrastructure Project Portfolio Records</h3>
                    <p class="text-xs text-slate-500">Search and filter across public infrastructure corridors</p>
                </div>
                <div class="flex items-center gap-3">
                    <select id="cityFilter" class="text-xs font-semibold rounded-lg border border-slate-300 px-3 py-1.5 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="ALL">All Cities</option>
                        <option value="Pune">Pune</option>
                        <option value="Ludhiana">Ludhiana</option>
                        <option value="Bengaluru">Bengaluru</option>
                        <option value="Delhi">Delhi</option>
                        <option value="Ahmedabad">Ahmedabad</option>
                        <option value="Mumbai">Mumbai</option>
                    </select>
                    <select id="statusFilter" class="text-xs font-semibold rounded-lg border border-slate-300 px-3 py-1.5 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="ALL">All Audit Statuses</option>
                        <option value="Ghost">Ghost Completions Only</option>
                        <option value="Delayed">Milestone Slippages</option>
                        <option value="Clean">Verified Clean</option>
                    </select>
                </div>
            </div>

            <div class="overflow-x-auto custom-scrollbar">
                <table class="min-w-full text-left text-xs divide-y divide-slate-200">
                    <thead class="bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider">
                        <tr>
                            <th class="py-3 px-3">Project ID</th>
                            <th class="py-3 px-3">City</th>
                            <th class="py-3 px-3">Corridor Scope</th>
                            <th class="py-3 px-3">Sanctioned (₹ Cr)</th>
                            <th class="py-3 px-3">Length</th>
                            <th class="py-3 px-3">Paper Status</th>
                            <th class="py-3 px-3">Citizen Defects/km</th>
                            <th class="py-3 px-3">AI Audit Flag</th>
                            <th class="py-3 px-3 text-right">Inspect</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 text-slate-700" id="projectTableBody">
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Core Machine Learning Laboratory Gallery -->
        <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6">
            <div class="border-b border-slate-200 pb-4">
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-indigo-100 text-indigo-800">
                    <i class="fa-solid fa-flask"></i> ML Evaluation Suite
                </span>
                <h3 class="text-xl font-bold text-slate-900 mt-2">Core Machine Learning Models & Predictive Visualizations</h3>
                <p class="text-xs text-slate-500">Empirical model outputs and evaluations powering the Policy-to-Impact Intelligence System.</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-blue-600 uppercase">Optimization Engine</span>
                        <h4 class="font-bold text-sm text-slate-900">Gradient Descent Convergence</h4>
                        <p class="text-xs text-slate-500">Batch vs Stochastic vs Mini-Batch GD</p>
                    </div>
                    <img src="outputs/figures/01_gradient_descent_convergence.png" alt="Gradient Descent Convergence" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-blue-600 uppercase">Non-Linearity Modeling</span>
                        <h4 class="font-bold text-sm text-slate-900">Polynomial Regression Fits</h4>
                        <p class="text-xs text-slate-500">Cost Escalation Learning Curves</p>
                    </div>
                    <img src="outputs/figures/02_polynomial_regression_learning_curves.png" alt="Polynomial Regression Fits" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-blue-600 uppercase">Regularization Engine</span>
                        <h4 class="font-bold text-sm text-slate-900">Ridge & Lasso Shrinkage Paths</h4>
                        <p class="text-xs text-slate-500">Lasso Automatic Feature Selection</p>
                    </div>
                    <img src="outputs/figures/03_ridge_lasso_regularization.png" alt="Ridge & Lasso" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-indigo-600 uppercase">Classification Engine</span>
                        <h4 class="font-bold text-sm text-slate-900">Logistic 2D Decision Boundary</h4>
                        <p class="text-xs text-slate-500">Separating Hyperplane at P(y=1)=0.5</p>
                    </div>
                    <img src="outputs/figures/04_logistic_regression_decision_boundary.png" alt="Logistic Regression" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-indigo-600 uppercase">Civic Grievance NLP</span>
                        <h4 class="font-bold text-sm text-slate-900">Naïve Bayes ROC & PR Curves</h4>
                        <p class="text-xs text-slate-500">ROC-AUC 0.707 | Avg Precision 0.888</p>
                    </div>
                    <img src="outputs/figures/05_naive_bayes_roc_precision_recall.png" alt="Naive Bayes" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-indigo-600 uppercase">Geospatial Clustering</span>
                        <h4 class="font-bold text-sm text-slate-900">K-Means Spatial Hotspots</h4>
                        <p class="text-xs text-slate-500">Elbow Curve & Silhouette Score (k=7)</p>
                    </div>
                    <img src="outputs/figures/06_kmeans_hotspots_elbow_silhouette.png" alt="K-Means Hotspots" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-emerald-600 uppercase">Support Vector Machine</span>
                        <h4 class="font-bold text-sm text-slate-900">Linear vs Poly vs RBF Kernels</h4>
                        <p class="text-xs text-slate-500">RBF Achieves 100% Separation</p>
                    </div>
                    <img src="outputs/figures/08_svm_linear_poly_rbf_comparison.png" alt="SVM Kernels" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-emerald-600 uppercase">Audit Rule Tree</span>
                        <h4 class="font-bold text-sm text-slate-900">CART Tree: Gini vs Entropy</h4>
                        <p class="text-xs text-slate-500">Human-Auditable Inspection Rules</p>
                    </div>
                    <img src="outputs/figures/09_decision_tree_cart_gini_vs_entropy.png" alt="CART Decision Tree" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-emerald-600 uppercase">Triangulation Model</span>
                        <h4 class="font-bold text-sm text-slate-900">Random Forest Feature Ranking</h4>
                        <p class="text-xs text-slate-500">Top Drivers: Defect Density & Complaints</p>
                    </div>
                    <img src="outputs/figures/10_random_forest_feature_importance.png" alt="Random Forest Feature Importance" class="w-full h-52 object-contain bg-white p-2">
                </div>

                <div class="border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow transition bg-slate-50/50">
                    <div class="p-4 border-b border-slate-200 bg-white">
                        <span class="text-xs font-bold text-emerald-600 uppercase">Ensemble Benchmark</span>
                        <h4 class="font-bold text-sm text-slate-900">AdaBoost & Gradient Boosting</h4>
                        <p class="text-xs text-slate-500">ROC-AUC: 0.981 vs Single Tree</p>
                    </div>
                    <img src="outputs/figures/11_ensemble_models_comparison.png" alt="Ensemble Models Comparison" class="w-full h-52 object-contain bg-white p-2">
                </div>

            </div>
        </div>

    </main>

    <!-- Inspection Modal Dialog -->
    <div id="auditModal" class="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-white rounded-2xl max-w-2xl w-full shadow-2xl border border-slate-200 overflow-hidden transform transition-all">
            <div class="p-6 border-b border-slate-100 flex items-center justify-between bg-slate-50">
                <div class="flex items-center gap-2">
                    <span id="modalBadge" class="px-2.5 py-1 rounded text-xs font-bold"></span>
                    <h4 id="modalTitle" class="text-base font-bold text-slate-900"></h4>
                </div>
                <button onclick="closeModal()" class="text-slate-400 hover:text-slate-600 text-lg">&times;</button>
            </div>
            <div class="p-6 space-y-4 text-xs text-slate-700">
                <div class="grid grid-cols-3 gap-3">
                    <div class="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <p class="font-bold text-blue-700 uppercase">1. Policy Sanction</p>
                        <p id="modalBudget" class="mt-1 font-semibold"></p>
                        <p id="modalLength" class="text-slate-500"></p>
                    </div>
                    <div class="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <p class="font-bold text-amber-700 uppercase">2. Tender Claim</p>
                        <p id="modalClaim" class="mt-1 font-semibold"></p>
                        <p id="modalDisbursed" class="text-slate-500"></p>
                    </div>
                    <div class="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <p class="font-bold text-rose-700 uppercase">3. Ground Reality</p>
                        <p id="modalComplaints" class="mt-1 font-semibold"></p>
                        <p id="modalDefects" class="text-slate-500"></p>
                    </div>
                </div>
                <div class="bg-blue-50 border-l-4 border-blue-600 p-3 rounded-r-lg">
                    <p class="font-bold text-blue-900">Recommended Audit Action:</p>
                    <p id="modalAction" class="text-blue-800 mt-1"></p>
                </div>
            </div>
            <div class="p-4 bg-slate-50 border-t border-slate-100 text-right">
                <button onclick="closeModal()" class="px-4 py-1.5 rounded-lg bg-slate-900 text-white font-semibold text-xs hover:bg-slate-800">Close Dossier</button>
            </div>
        </div>
    </div>

    <!-- Script Data Ingestion & Logic -->
    <script>
        const projects = {projects_json};

        const map = L.map('map').setView([20.5937, 78.9629], 5);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            maxZoom: 18
        }}).addTo(map);

        let markers = [];

        function renderMapAndTable() {{
            const cityVal = document.getElementById('cityFilter').value;
            const statusVal = document.getElementById('statusFilter').value;

            markers.forEach(m => map.removeLayer(m));
            markers = [];

            const tbody = document.getElementById('projectTableBody');
            tbody.innerHTML = '';

            const priorityContainer = document.getElementById('priorityList');
            priorityContainer.innerHTML = '';

            const filtered = projects.filter(p => {{
                if (cityVal !== 'ALL' && p.city !== cityVal) return false;
                if (statusVal === 'Ghost' && p.discrepancy_type !== 'Suspected Ghost Completion') return false;
                if (statusVal === 'Delayed' && p.discrepancy_type !== 'Severe Milestone Slippage') return false;
                if (statusVal === 'Clean' && p.is_discrepancy !== 0) return false;
                return true;
            }});

            filtered.forEach((p, idx) => {{
                let color = '#059669';
                let badgeClass = 'bg-emerald-100 text-emerald-800';
                if (p.discrepancy_type === 'Suspected Ghost Completion') {{
                    color = '#DC2626';
                    badgeClass = 'bg-rose-100 text-rose-800';
                }} else if (p.discrepancy_type === 'Severe Milestone Slippage') {{
                    color = '#D97706';
                    badgeClass = 'bg-amber-100 text-amber-800';
                }}

                const marker = L.circleMarker([p.latitude, p.longitude], {{
                    radius: 7,
                    fillColor: color,
                    color: '#FFFFFF',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.85
                }}).addTo(map);

                marker.bindPopup(`
                    <div class="text-xs p-1">
                        <b>${{p.project_name}}</b><br>
                        City: ${{p.city}}<br>
                        Sanctioned: ₹${{p.sanctioned_budget_cr}} Cr<br>
                        Paper Status: ${{p.reported_status}} (${{p.reported_progress_pct}}%)<br>
                        Defects: ${{p.ground_defect_density}}/km<br>
                        <span class="font-bold text-rose-600">${{p.discrepancy_type}}</span>
                    </div>
                `);
                markers.push(marker);

                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-50 transition';
                tr.innerHTML = `
                    <td class="py-2.5 px-3 font-mono text-[11px] font-bold text-slate-800">${{p.project_id}}</td>
                    <td class="py-2.5 px-3 font-semibold">${{p.city}}</td>
                    <td class="py-2.5 px-3 font-medium text-slate-900">${{p.project_name}}</td>
                    <td class="py-2.5 px-3 font-semibold">₹${{p.sanctioned_budget_cr}} Cr</td>
                    <td class="py-2.5 px-3">${{p.length_km}} km</td>
                    <td class="py-2.5 px-3">${{p.reported_status}} (${{p.reported_progress_pct}}%)</td>
                    <td class="py-2.5 px-3 font-semibold text-rose-600">${{p.ground_defect_density}}</td>
                    <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded font-bold text-[10px] ${{badgeClass}}">${{p.discrepancy_type}}</span></td>
                    <td class="py-2.5 px-3 text-right">
                        <button onclick="openModal('${{p.project_id}}')" class="px-2.5 py-1 text-[11px] font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 rounded transition">Dossier</button>
                    </td>
                `;
                tbody.appendChild(tr);

                if (p.is_discrepancy === 1 && priorityContainer.children.length < 5) {{
                    const card = document.createElement('div');
                    card.className = 'p-3 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white transition cursor-pointer';
                    card.onclick = () => openModal(p.project_id);
                    card.innerHTML = `
                        <div class="flex items-center justify-between text-[11px]">
                            <span class="font-bold text-slate-900">${{p.city}}</span>
                            <span class="px-2 py-0.5 rounded font-bold ${{badgeClass}}">${{p.discrepancy_type}}</span>
                        </div>
                        <p class="font-medium text-slate-800 mt-1 line-clamp-1">${{p.project_name}}</p>
                        <p class="text-slate-500 text-[11px] mt-1">₹${{p.sanctioned_budget_cr}} Cr | ${{p.citizen_complaint_count}} Citizen Complaints</p>
                    `;
                    priorityContainer.appendChild(card);
                }}
            }});

            if (markers.length > 0 && cityVal !== 'ALL') {{
                const group = new L.featureGroup(markers);
                map.fitBounds(group.getBounds().pad(0.3));
            }}
        }}

        function openModal(projectId) {{
            const p = projects.find(item => item.project_id === projectId);
            if (!p) return;

            document.getElementById('modalTitle').innerText = p.project_name;
            document.getElementById('modalBadge').innerText = p.discrepancy_type;
            document.getElementById('modalBadge').className = p.is_discrepancy === 1 
                ? (p.discrepancy_type.includes('Ghost') ? 'px-2.5 py-1 rounded text-xs font-bold bg-rose-100 text-rose-800' : 'px-2.5 py-1 rounded text-xs font-bold bg-amber-100 text-amber-800')
                : 'px-2.5 py-1 rounded text-xs font-bold bg-emerald-100 text-emerald-800';

            document.getElementById('modalBudget').innerText = `Sanctioned: ₹${{p.sanctioned_budget_cr}} Cr`;
            document.getElementById('modalLength').innerText = `Corridor: ${{p.length_km}} km (${{p.num_major_junctions}} junctions)`;

            document.getElementById('modalClaim').innerText = `Reported: ${{p.reported_status}} (${{p.reported_progress_pct}}%)`;
            document.getElementById('modalDisbursed').innerText = `Disbursed Funds: ${{p.disbursed_funds_pct}}%`;

            document.getElementById('modalComplaints').innerText = `Complaints: ${{p.citizen_complaint_count}} reports`;
            document.getElementById('modalDefects').innerText = `Defect Density: ${{p.ground_defect_density}} /km`;

            document.getElementById('modalAction').innerText = p.discrepancy_type.includes('Ghost')
                ? `Depute third-party technical audit team to verify physical kilometer markers at (${{p.latitude}}, ${{p.longitude}}) and drill core samples to test bitumen thickness.`
                : (p.is_discrepancy === 1 
                    ? `Request CPM/PERT revised milestone schedule and conduct joint inspection of open utility trenches.`
                    : `Routine monitoring verified. Project metrics conform to scheduled warranty lifecycle.`);

            document.getElementById('auditModal').classList.remove('hidden');
        }}

        function closeModal() {{
            document.getElementById('auditModal').classList.add('hidden');
        }}

        document.getElementById('cityFilter').addEventListener('change', renderMapAndTable);
        document.getElementById('statusFilter').addEventListener('change', renderMapAndTable);

        renderMapAndTable();
    </script>
</body>
</html>
"""
    output_path = os.path.join("dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Generated standalone interactive web dashboard at: {output_path}")


if __name__ == "__main__":
    generate_html_dashboard()

"""
=============================================================================
Policy-to-Impact Intelligence System: Interactive Web Dashboard
=============================================================================
From Government Policy to Ground Reality: An Evidence-Driven AI Framework
for Monitoring Public Infrastructure Delivery and Detecting Emerging Service Failures.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

st.set_page_config(
    page_title="Policy-to-Impact Intelligence System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E5E7EB;
        text-align: left;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .flag-badge-ghost {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .flag-badge-delay {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .flag-badge-verified {
        background-color: #D1FAE5;
        color: #047857;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .audit-card {
        background: #F9FAFB;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data_path = os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv")
    complaints_path = os.path.join("data", "raw", "citizen_complaints.csv")
    
    if not os.path.exists(data_path) or not os.path.exists(complaints_path):
        from src.data_loader import generate_curated_datasets
        generate_curated_datasets()
        
    df_proj = pd.read_csv(data_path)
    df_comp = pd.read_csv(complaints_path)
    return df_proj, df_comp


df_projects, df_complaints = load_data()

st.sidebar.image("https://img.icons8.com/color/96/government.png", width=70)
st.sidebar.title("Auditor Controls")

selected_view = st.sidebar.radio(
    "Navigation View",
    ["🏛️ Executive Audit Dashboard", "🗺️ GIS Infrastructure Map", "🔍 Evidence Dossier & Triangulation", "🧠 Machine Learning Laboratory", "📄 Research Flow Paper"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Portfolio")
all_cities = ["All Cities"] + sorted(list(df_projects["city"].unique()))
city_filter = st.sidebar.selectbox("Geographic Jurisdiction", all_cities)

all_types = ["All Projects", "Discrepancies Only", "Ghost Completions Only", "Severe Slippages Only", "Verified Clean Only"]
status_filter = st.sidebar.selectbox("Audit Status Filter", all_types)

budget_range = st.sidebar.slider(
    "Sanctioned Budget (₹ Crore)",
    min_value=float(df_projects["sanctioned_budget_cr"].min()),
    max_value=float(df_projects["sanctioned_budget_cr"].max()),
    value=(float(df_projects["sanctioned_budget_cr"].min()), float(df_projects["sanctioned_budget_cr"].max()))
)

filtered_df = df_projects.copy()
if city_filter != "All Cities":
    filtered_df = filtered_df[filtered_df["city"] == city_filter]

if status_filter == "Discrepancies Only":
    filtered_df = filtered_df[filtered_df["is_discrepancy"] == 1]
elif status_filter == "Ghost Completions Only":
    filtered_df = filtered_df[filtered_df["discrepancy_type"] == "Suspected Ghost Completion"]
elif status_filter == "Severe Slippages Only":
    filtered_df = filtered_df[filtered_df["discrepancy_type"] == "Severe Milestone Slippage"]
elif status_filter == "Verified Clean Only":
    filtered_df = filtered_df[filtered_df["is_discrepancy"] == 0]

filtered_df = filtered_df[
    (filtered_df["sanctioned_budget_cr"] >= budget_range[0]) &
    (filtered_df["sanctioned_budget_cr"] <= budget_range[1])
]

st.markdown('<div class="main-title">Policy-to-Impact Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Government Policy to Ground Reality: An Evidence-Driven AI Framework for Monitoring Public Infrastructure Delivery</div>', unsafe_allow_html=True)


if selected_view == "🏛️ Executive Audit Dashboard":
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_budget = df_projects["sanctioned_budget_cr"].sum()
    total_projects = len(df_projects)
    total_ghost = len(df_projects[df_projects["discrepancy_type"] == "Suspected Ghost Completion"])
    total_delayed = len(df_projects[df_projects["discrepancy_type"] == "Severe Milestone Slippage"])
    total_verified = len(df_projects[df_projects["is_discrepancy"] == 0])

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Monitored Portfolio</div><div class="metric-value">{total_projects} Projects</div><small style="color:#6B7280;">Across 6 Metropolitan Hubs</small></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sanctioned Outlay</div><div class="metric-value">₹{total_budget:,.0f} Cr</div><small style="color:#6B7280;">NIP & State Budgets</small></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#B91C1C;">Ghost Completions</div><div class="metric-value" style="color:#DC2626;">{total_ghost} Flagged</div><small style="color:#DC2626;">Paper Complete != Ground</small></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#D97706;">Milestone Slippages</div><div class="metric-value" style="color:#D97706;">{total_delayed} Delayed</div><small style="color:#D97706;">Duration Delta > 6 Months</small></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#047857;">Verified Clean</div><div class="metric-value" style="color:#059669;">{total_verified} Clean</div><small style="color:#059669;">Complaints Match Progress</small></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([6, 4])

    with c1:
        st.subheader("📊 Portfolio Discrepancy Breakdown by City")
        city_summary = df_projects.groupby(["city", "discrepancy_type"]).size().unstack(fill_value=0)
        st.bar_chart(city_summary)

    with c2:
        st.subheader("⚠️ High-Risk Discrepancy Matrix")
        st.markdown("""
        **How the AI Triangulation Engine Detects Risk:**
        * **Ghost Completion:** Disbursed Funds > 95% AND Reported Progress = 100%, BUT Citizen Defect Density > 1.5 complaints/km.
        * **Severe Slippage:** Calendar Deadline Passed, Actual Delay > 6 Months, AND Stalled Labor on Ground.
        * **Verified Delivery:** Reported Progress aligns with low ground complaints (< 0.5/km).
        """)
        
        disc_counts = df_projects["discrepancy_type"].value_counts()
        st.dataframe(disc_counts, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 Monitored Infrastructure Corridors ({len(filtered_df)} projects matching filters)")

    display_df = filtered_df[[
        "project_id", "city", "project_name", "sanctioned_budget_cr", "length_km", 
        "delay_months", "reported_progress_pct", "ground_defect_density", "discrepancy_type"
    ]].copy()
    display_df.columns = [
        "Project ID", "City", "Corridor Name", "Sanctioned (₹ Cr)", "Length (km)", 
        "Delay (m)", "Paper Progress (%)", "Ground Defects/km", "AI Audit Flag"
    ]
    st.dataframe(display_df, use_container_width=True, height=350)


elif selected_view == "🗺️ GIS Infrastructure Map":
    st.subheader("🗺️ Geospatial Infrastructure Delivery Map: Corridors & Defect Concentrations")
    st.markdown("Interactive corridor coordinates for Pune, Ludhiana, Bengaluru, Delhi, Ahmedabad, and Mumbai.")

    map_df = filtered_df[["latitude", "longitude", "project_name", "city", "discrepancy_type", "sanctioned_budget_cr"]].copy()
    st.map(map_df, latitude="latitude", longitude="longitude", size=25, color="#DC2626")

    st.markdown("### Corridor Geographic Details")
    st.dataframe(map_df, use_container_width=True)


elif selected_view == "🔍 Evidence Dossier & Triangulation":
    st.subheader("🔍 Project Evidence Dossiers: Multi-Tier Verification")
    st.markdown("Click on any infrastructure project below to inspect the tripartite evidence chain (Policy Commitments vs Tender Delivery vs Citizen Grievances).")

    proj_options = filtered_df["project_name"].tolist()
    if not proj_options:
        st.warning("No projects match the current filter criteria.")
    else:
        selected_pname = st.selectbox("Select Corridor to Audit", proj_options)
        p_row = filtered_df[filtered_df["project_name"] == selected_pname].iloc[0]

        badge_style = "flag-badge-ghost" if p_row["discrepancy_type"] == "Suspected Ghost Completion" else ("flag-badge-delay" if p_row["is_discrepancy"] == 1 else "flag-badge-verified")
        
        st.markdown(f"""
        <div class="audit-card">
            <span class="{badge_style}">{p_row['discrepancy_type']}</span>
            <h3 style="margin-top:10px; margin-bottom:5px;">{p_row['project_name']}</h3>
            <p style="color:#6B7280; font-size:0.9rem;">Project Code: <b>{p_row['project_id']}</b> | City Jurisdiction: <b>{p_row['city']}</b> | GPS: ({p_row['latitude']}, {p_row['longitude']})</p>
        </div>
        """, unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("#### 1. Macro Policy Promise")
            st.info(f"""
            * **Sanctioned Outlay:** ₹{p_row['sanctioned_budget_cr']:,.2f} Crore
            * **Corridor Scope:** {p_row['length_km']} km four-lane expansion
            * **Key Junctions:** {p_row['num_major_junctions']} major traffic nodes
            * **Scheduled Timeline:** {p_row['scheduled_duration_months']} Months
            * **Mandate Authority:** Municipal Urban Development Dept.
            """)

        with colB:
            st.markdown("#### 2. Meso Delivery Record")
            st.warning(f"""
            * **Reported Status:** **{p_row['reported_status']}** ({p_row['reported_progress_pct']}% complete)
            * **Funds Disbursed:** {p_row['disbursed_funds_pct']}% of total budget
            * **Contractor Experience:** {p_row['contractor_experience_yrs']} Years
            * **Contractor Past Delays:** {p_row['past_contractor_delays']} recorded incidents
            * **Timeline Delay:** **+{p_row['delay_months']} Months** beyond contract
            """)

        with colC:
            st.markdown("#### 3. Micro Ground Reality")
            st.error(f"""
            * **Citizen Complaints:** **{p_row['citizen_complaint_count']} reports logged**
            * **Defect Density:** **{p_row['ground_defect_density']} defects/km**
            * **Dominant Signals:** Unpaved trenches, missing bitumen, asphalt degradation.
            * **Local Beat Status:** Stalled excavation, safety hazards.
            """)

        st.markdown("### 🤖 Objective AI Inspection Prompt")
        if p_row["discrepancy_type"] == "Suspected Ghost Completion":
            st.error(f"""
            **ATTENTION AUDITOR:** System detected divergence between documented paper claim and ground evidence.  
            * **Paper Claim:** Contractor reported 100% completion and collected {p_row['disbursed_funds_pct']}% of funds.  
            * **Ground Signal:** Unusually dense concentration of citizen failure reports ({p_row['ground_defect_density']}/km).  
            * **Recommended Action:** Depute a third-party technical quality verification team to inspect kilometer marks along {p_row['project_name']} and audit core drill asphalt thickness samples.
            """)
        elif p_row["discrepancy_type"] == "Severe Milestone Slippage":
            st.warning(f"""
            **ATTENTION AUDITOR:** Project has experienced a significant chronological delay (+{p_row['delay_months']} months).  
            * **Recommended Action:** Request revised CPM/PERT milestone schedule from the contractor and conduct joint survey on open junction trenches.
            """)
        else:
            st.success("""
            **PORTFOLIO ON TRACK:** Reported progress matches low citizen defect signals. Recommended for standard warranty lifecycle audit.
            """)

        st.markdown("#### Sample Citizen Complaints Tagged to this Corridor:")
        p_complaints = df_complaints[df_complaints["project_id"] == p_row["project_id"]]
        if len(p_complaints) > 0:
            st.dataframe(p_complaints[["complaint_id", "category", "complaint_text", "is_urgent", "days_since_reported"]], use_container_width=True)
        else:
            st.write("No citizen complaints logged for this corridor.")


elif selected_view == "🧠 Machine Learning Laboratory":
    st.subheader("🧠 Machine Learning Laboratory: Core Predictive & Triangulation Engines")
    st.markdown("Interactive visualizations generated from the core machine learning models:")

    tab1, tab2, tab3 = st.tabs([
        "📈 Timeline & Cost Escalation Models", 
        "💬 Citizen Signal NLP & Hotspots", 
        "🌲 Triangulation Discrepancy & Ensembles"
    ])

    with tab1:
        st.markdown("### Timeline & Cost Escalation Predictive Modeling")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.image("outputs/figures/01_gradient_descent_convergence.png", caption="Gradient Descent Optimization Convergence (Batch vs. SGD vs. Mini-Batch)")
        with rc2:
            st.image("outputs/figures/02_polynomial_regression_learning_curves.png", caption="Cost Escalation Modeling & Learning Curves")

        st.markdown("---")
        st.image("outputs/figures/03_ridge_lasso_regularization.png", caption="L1/L2 Regularization Paths & Feature Selection")

    with tab2:
        st.markdown("### Citizen Signal Mining & Spatial Hotspot Clustering")
        nc1, nc2 = st.columns(2)
        with nc1:
            st.image("outputs/figures/04_logistic_regression_decision_boundary.png", caption="Binary Discrepancy Decision Boundary")
        with nc2:
            st.image("outputs/figures/05_naive_bayes_roc_precision_recall.png", caption="Naïve Bayes ROC Curve & Precision-Recall Tradeoff")

        st.markdown("---")
        st.image("outputs/figures/06_kmeans_hotspots_elbow_silhouette.png", caption="K-Means Geospatial Hotspots & Silhouette Analysis")

    with tab3:
        st.markdown("### Triangulation Discrepancy & Service Failure Detection")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.image("outputs/figures/08_svm_linear_poly_rbf_comparison.png", caption="Support Vector Machines: Linear vs. Polynomial vs. RBF Kernels")
        with sc2:
            st.image("outputs/figures/09_decision_tree_cart_gini_vs_entropy.png", caption="CART Decision Tree Splits for Government Auditors")

        st.markdown("---")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.image("outputs/figures/10_random_forest_feature_importance.png", caption="Random Forest Feature Importance Ranking")
        with ec2:
            st.image("outputs/figures/11_ensemble_models_comparison.png", caption="Ensemble Classifier Performance Benchmark")


elif selected_view == "📄 Research Flow Paper":
    st.subheader("📄 Research Flow Paper & Architecture Specification")
    paper_path = "FLOW_PAPER.md"
    if os.path.exists(paper_path):
        with open(paper_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("FLOW_PAPER.md not found in the workspace root.")

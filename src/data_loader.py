"""
Data Loader & Dataset Generator for Policy-to-Impact Intelligence System.
Curates real-world inspired public infrastructure projects, delivery audits,
and citizen complaints to train and evaluate the 12 syllabus ML algorithms.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        os.path.join("outputs", "figures"),
        "notebooks",
        "src"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_curated_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic multi-tier infrastructure data:
    1. Policy Commitments (NIP / State Budgets)
    2. Tender & Delivery Progress Logs (CPPP / PWD)
    3. Citizen Grievance Streams & Civic News Beats (Municipal Portals)
    """
    ensure_directories()
    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. Policy & Delivery Data (60 Real-World Infrastructure Projects)
    # ---------------------------------------------------------
    project_archetypes = [
        {"name": "Pune High-Capacity Mass Transit Road Upgradation", "city": "Pune", "base_len": 64.0, "base_cost": 1866.0, "duration": 24},
        {"name": "Krishna Mandir to Ishmeet Chowk Corridor Improvement", "city": "Ludhiana", "base_len": 8.5, "base_cost": 42.0, "duration": 12},
        {"name": "Outer Ring Road Junction & Flyover Decongestion Scheme", "city": "Bengaluru", "base_len": 28.0, "base_cost": 840.0, "duration": 30},
        {"name": "Chandigarh-Ludhiana Highway Smart Corridor Widening", "city": "Ludhiana", "base_len": 45.0, "base_cost": 620.0, "duration": 36},
        {"name": "Shivaji Nagar to Hinjewadi Arterial Road Strengthening", "city": "Pune", "base_len": 18.0, "base_cost": 310.0, "duration": 18},
        {"name": "Old Delhi Heritage Corridor Pavement Rehabilitation", "city": "Delhi", "base_len": 6.2, "base_cost": 55.0, "duration": 10},
        {"name": "Sardar Patel Ring Road Stormwater & Surface Overlay", "city": "Ahmedabad", "base_len": 32.0, "base_cost": 490.0, "duration": 22},
        {"name": "Bandra-Kurla Complex Elevated Connector Link Road", "city": "Mumbai", "base_len": 4.8, "base_cost": 720.0, "duration": 28},
        {"name": "Gill Road Industrial Area Asphalt Reconstruction", "city": "Ludhiana", "base_len": 12.0, "base_cost": 88.0, "duration": 14},
        {"name": "Katraj-Kondhwa Bypass 6-Lane Expansion Project", "city": "Pune", "base_len": 9.5, "base_cost": 215.0, "duration": 20},
    ]

    records = []
    project_id_counter = 101

    for i in range(60):
        archetype = project_archetypes[i % len(project_archetypes)]
        noise_factor = np.random.uniform(0.75, 1.35)
        
        pid = f"PROJ-INFRA-{project_id_counter}"
        project_id_counter += 1
        
        name = f"{archetype['name']} Phase {((i // len(project_archetypes)) + 1)}"
        city = archetype["city"]
        length_km = round(archetype["base_len"] * noise_factor, 1)
        sanctioned_budget_cr = round(archetype["base_cost"] * noise_factor, 2)
        scheduled_duration_months = int(np.round(archetype["duration"] * noise_factor))
        contractor_experience_yrs = int(np.random.randint(2, 25))
        past_contractor_delays = int(np.random.poisson(lam=2.5))
        num_major_junctions = int(max(1, np.round(length_km / 3.5)))
        
        # Real-world actual cost and completion dynamics
        # Feature relationship: cost overrun depends on length, contractor delay history, and duration
        cost_overrun_pct = (
            0.05 * (past_contractor_delays) +
            0.002 * (length_km) +
            np.random.normal(0.08, 0.05)
        )
        cost_overrun_pct = max(-0.05, cost_overrun_pct)
        actual_cost_cr = round(sanctioned_budget_cr * (1.0 + cost_overrun_pct), 2)

        # Actual completion time in months
        delay_months = int(np.round(
            0.6 * past_contractor_delays +
            0.15 * num_major_junctions +
            np.random.normal(3.0, 2.0)
        ))
        delay_months = max(0, delay_months)
        actual_duration_months = scheduled_duration_months + delay_months

        # Reported progress on government portal (0 to 100%)
        # Some projects are legitimately finished, some in-progress, some "ghost completed"
        status_category = np.random.choice(["Completed", "In-Progress", "Ghost-Completed"], p=[0.45, 0.35, 0.20])
        
        if status_category == "Completed":
            reported_progress_pct = 100.0
            reported_status = "Completed"
            disbursed_funds_pct = round(np.random.uniform(92.0, 100.0), 1)
            ground_defect_density = round(np.random.uniform(0.05, 0.4), 2)  # Low complaints
            is_discrepancy = 0
            discrepancy_type = "None"
        elif status_category == "In-Progress":
            reported_progress_pct = round(np.random.uniform(30.0, 85.0), 1)
            reported_status = "In-Progress"
            disbursed_funds_pct = round(reported_progress_pct + np.random.uniform(-5.0, 12.0), 1)
            ground_defect_density = round(np.random.uniform(0.3, 1.2), 2)
            # High delay is a discrepancy
            is_discrepancy = 1 if delay_months > 6 else 0
            discrepancy_type = "Severe Milestone Slippage" if is_discrepancy else "Normal Ongoing"
        else:
            # Ghost Completion: Government claims 100% completed, funds disbursed, but high ground failure
            reported_progress_pct = 100.0
            reported_status = "Completed"
            disbursed_funds_pct = round(np.random.uniform(96.0, 105.0), 1)
            ground_defect_density = round(np.random.uniform(1.8, 4.5), 2)  # Severe complaints
            is_discrepancy = 1
            discrepancy_type = "Suspected Ghost Completion"

        # Coordinates around Indian cities
        city_coords = {
            "Pune": (18.5204, 73.8567),
            "Ludhiana": (30.9010, 75.8573),
            "Bengaluru": (12.9716, 77.5946),
            "Delhi": (28.6139, 77.2090),
            "Ahmedabad": (23.0225, 72.5714),
            "Mumbai": (19.0760, 72.8777)
        }
        base_lat, base_lon = city_coords.get(city, (20.5937, 78.9629))
        lat = round(base_lat + np.random.normal(0, 0.04), 5)
        lon = round(base_lon + np.random.normal(0, 0.04), 5)

        citizen_complaint_count = int(np.round(ground_defect_density * 15 + np.random.poisson(2)))

        records.append({
            "project_id": pid,
            "project_name": name,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "length_km": length_km,
            "sanctioned_budget_cr": sanctioned_budget_cr,
            "actual_cost_cr": actual_cost_cr,
            "cost_overrun_cr": round(actual_cost_cr - sanctioned_budget_cr, 2),
            "cost_overrun_pct": round(cost_overrun_pct * 100, 2),
            "scheduled_duration_months": scheduled_duration_months,
            "actual_duration_months": actual_duration_months,
            "delay_months": delay_months,
            "contractor_experience_yrs": contractor_experience_yrs,
            "past_contractor_delays": past_contractor_delays,
            "num_major_junctions": num_major_junctions,
            "reported_progress_pct": reported_progress_pct,
            "reported_status": reported_status,
            "disbursed_funds_pct": disbursed_funds_pct,
            "ground_defect_density": ground_defect_density,
            "citizen_complaint_count": citizen_complaint_count,
            "is_discrepancy": is_discrepancy,
            "discrepancy_type": discrepancy_type
        })

    df_projects = pd.DataFrame(records)

    # ---------------------------------------------------------
    # 2. Citizen Grievance & Civic Complaints (Text Dataset)
    # ---------------------------------------------------------
    complaint_templates = [
        ("Potholes and Surface Damage", [
            "Severe craters and broken bitumen on the newly tarred stretch near junction.",
            "Deep potholes after first rain shower. Bikers losing balance and skidding daily.",
            "Substandard asphalt used; the top gravel layer has completely washed off.",
            "Massive crater opposite the bus stop causing severe traffic jams.",
            "Uneven road leveling with open pits; driving here at night is extremely dangerous."
        ]),
        ("Dug-up Road and Incomplete Work", [
            "Contractor dug up the entire 2km road for pipeline laying and abandoned work for 3 months.",
            "Unfinished drainage trench left open without warning barricades or safety signage.",
            "Heaps of gravel and construction debris blocking two full lanes of the main road.",
            "Water project contractor dug up the newly paved road; zero coordination between departments.",
            "Dug-up stretch at water project site turns accident-prone; no laborers seen on site."
        ]),
        ("Waterlogging and Drainage Failure", [
            "Rainwater pooling over 2 feet deep due to missing or blocked stormwater drains.",
            "Severe inundation across both carriageways; water entered roadside commercial shops.",
            "Stormwater culvert collapsed under traffic load, completely choking the drainage line.",
            "Drainage covers missing; overflow of filthy runoff water across pedestrian walkway.",
            "Water stagnation for over 4 days after light drizzle, attracting mosquitoes."
        ])
    ]

    complaints_data = []
    comp_counter = 1001

    for idx, row in df_projects.iterrows():
        n_comp = row["citizen_complaint_count"]
        for _ in range(n_comp):
            cat_choice = np.random.choice(len(complaint_templates))
            cat_name, texts = complaint_templates[cat_choice]
            raw_text = np.random.choice(texts)
            text = f"[{row['city']} - {row['project_name']}] {raw_text}"
            
            # Severity: High (1) if project has severe ground discrepancy, else 0/1
            is_urgent = 1 if (row["is_discrepancy"] == 1 or "accident" in raw_text or "danger" in raw_text) else 0

            # Geo jitter near project
            c_lat = round(row["latitude"] + np.random.normal(0, 0.008), 5)
            c_lon = round(row["longitude"] + np.random.normal(0, 0.008), 5)

            complaints_data.append({
                "complaint_id": f"CMP-{comp_counter}",
                "project_id": row["project_id"],
                "city": row["city"],
                "complaint_text": text,
                "category": cat_name,
                "is_urgent": is_urgent,
                "latitude": c_lat,
                "longitude": c_lon,
                "days_since_reported": int(np.random.randint(1, 180))
            })
            comp_counter += 1

    df_complaints = pd.DataFrame(complaints_data)

    # ---------------------------------------------------------
    # 3. MNIST Digits Sample (For Syllabus Item 6: KNN on MNIST)
    # ---------------------------------------------------------
    from sklearn.datasets import load_digits
    digits = load_digits()
    mnist_df = pd.DataFrame(digits.data)
    mnist_df["target"] = digits.target

    # Save to disk
    df_projects.to_csv(os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv"), index=False)
    df_projects.to_json(os.path.join("data", "raw", "infrastructure_projects.json"), orient="records", indent=2)
    df_complaints.to_csv(os.path.join("data", "raw", "citizen_complaints.csv"), index=False)
    mnist_df.to_csv(os.path.join("data", "raw", "mnist_digits_sample.csv"), index=False)

    print(f"[OK] Generated {len(df_projects)} infrastructure project records.")
    print(f"[OK] Generated {len(df_complaints)} citizen complaints across {len(complaint_templates)} categories.")
    print(f"[OK] Generated MNIST benchmark dataset with {len(mnist_df)} samples and {mnist_df.shape[1]-1} pixel features.")
    return df_projects, df_complaints, mnist_df


if __name__ == "__main__":
    generate_curated_datasets()

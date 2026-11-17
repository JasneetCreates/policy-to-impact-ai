"""
Stage 2: Classification & Clustering Suite (Syllabus Items 5, 7, 12)
---------------------------------------------------------------------
5. Logistic Regression for binary classification & decision boundary evaluation.
7. Naive Bayes Classifier on citizen grievances with ROC curve and Precision-Recall tradeoff.
12. K-Means Clustering on spatial citizen complaints with Elbow Method & Silhouette analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler


def ensure_output_dir():
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)


# --------------------------------------------------------------------------
# Item 5: Logistic Regression for Binary Classification & Decision Boundary
# --------------------------------------------------------------------------
def evaluate_logistic_regression():
    """
    Train Logistic Regression to classify infrastructure projects as
    Discrepancy (1) vs Normal (0) based on Ground Defect Density and
    Disbursed Funds Percentage, and plot the 2D decision boundary.
    """
    ensure_output_dir()
    data_path = os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv")
    df = pd.read_csv(data_path)

    # 2 features for 2D decision boundary visualization
    features = ["ground_defect_density", "disbursed_funds_pct"]
    X = df[features].values
    y = df["is_discrepancy"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=======================================================")
    print("5. LOGISTIC REGRESSION: BINARY DISCREPANCY CLASSIFICATION")
    print("=======================================================")
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(f"  [TN: {cm[0,0]}  FP: {cm[0,1]}]")
    print(f"  [FN: {cm[1,0]}  TP: {cm[1,1]}]")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Discrepancy (1)"]))

    # Plotting 2D Decision Boundary
    x_min, x_max = X_scaled[:, 0].min() - 0.8, X_scaled[:, 0].max() + 0.8
    y_min, y_max = X_scaled[:, 1].min() - 0.8, X_scaled[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, Z, levels=25, cmap="coolwarm", alpha=0.6)
    plt.colorbar(contour, label="Discrepancy Probability P(y=1)")
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")

    # Plot actual points
    scatter = plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="k",
        s=60,
        label="Project Records"
    )

    plt.title("Logistic Regression Decision Boundary (Separating Hyperplane at P=0.5)", fontsize=11, fontweight="bold")
    plt.xlabel("Standardized Ground Defect Density (Citizen Signal)", fontsize=10)
    plt.ylabel("Standardized Disbursed Funds % (Delivery Record)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)

    plot_path = os.path.join("outputs", "figures", "04_logistic_regression_decision_boundary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Logistic Regression decision boundary plot to: {plot_path}")

    return clf, scaler


# --------------------------------------------------------------------------
# Item 7: Naive Bayes Classifier with ROC Curve & Precision-Recall Tradeoff
# --------------------------------------------------------------------------
def evaluate_naive_bayes():
    """
    Classify unstructured citizen complaints using Multinomial Naive Bayes (Bag of Words)
    and analyze ROC curve and Precision-Recall tradeoff curve.
    """
    ensure_output_dir()
    complaints_path = os.path.join("data", "raw", "citizen_complaints.csv")
    df = pd.read_csv(complaints_path)

    # Classification task: Predict whether a citizen report indicates an urgent hazard (is_urgent)
    X_text = df["complaint_text"].values
    y = df["is_urgent"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_vec, y_train)

    y_pred = nb_model.predict(X_test_vec)
    y_prob = nb_model.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print("\n=======================================================")
    print("7. NAIVE BAYES CLASSIFIER ON CITIZEN COMPLAINTS")
    print("=======================================================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Area Under ROC Curve (ROC-AUC) : {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC)     : {avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Routine Complaint (0)", "Urgent Hazard (1)"]))

    # Plot Side-by-Side: ROC Curve and Precision-Recall Tradeoff Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance Baseline")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    axes[0].set_title("Naïve Bayes ROC Curve", fontsize=11, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Precision-Recall Tradeoff Curve
    axes[1].plot(recall, precision, color="forestgreen", lw=2.5, label=f"PR curve (Avg Precision = {avg_precision:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Coverage of Genuine Service Hazards)", fontsize=10)
    axes[1].set_ylabel("Precision (Accuracy of Flagged Hazards)", fontsize=10)
    axes[1].set_title("Precision–Recall Tradeoff Curve", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "05_naive_bayes_roc_precision_recall.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Naive Bayes ROC and Precision-Recall plot to: {plot_path}")

    return nb_model, vectorizer


# --------------------------------------------------------------------------
# Item 12: K-Means Spatial Hotspot Clustering & Elbow / Silhouette Analysis
# --------------------------------------------------------------------------
def evaluate_kmeans_clustering():
    """
    Perform K-Means clustering on citizen complaints using GPS coordinates
    to discover geographical failure hotspots. Evaluate optimal k using
    both the Elbow Method (Inertia) and Silhouette Score.
    """
    ensure_output_dir()
    complaints_path = os.path.join("data", "raw", "citizen_complaints.csv")
    df = pd.read_csv(complaints_path)

    # Focus on Pune or Ludhiana cluster for clear spatial visualization
    city_df = df[df["city"] == "Pune"].copy()
    if len(city_df) < 50:
        city_df = df.copy()

    coords = city_df[["latitude", "longitude"]].values

    k_range = range(2, 9)
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(coords, kmeans.labels_))

    optimal_k = k_range[np.argmax(silhouettes)]

    # Fit final model with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(coords)
    centroids = final_kmeans.cluster_centers_

    print("\n=======================================================")
    print("12. K-MEANS SPATIAL HOTSPOT CLUSTERING")
    print("=======================================================")
    print(f"Evaluated k from {k_range[0]} to {k_range[-1]}")
    for k, inert, sil in zip(k_range, inertias, silhouettes):
        print(f"  k={k} -> Inertia: {inert:.6f} | Silhouette Score: {sil:.4f}")
    print(f"Optimal Cluster Count (max Silhouette): k = {optimal_k}")

    # Plot 1x3 panel: Elbow Curve, Silhouette Curve, Spatial Cluster Map
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Elbow Method (Inertia)
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=7)
    axes[0].set_title("Elbow Method: Inertia vs k", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=10)
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Silhouette Analysis
    axes[1].plot(list(k_range), silhouettes, "gs-", linewidth=2, markersize=7)
    axes[1].axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    axes[1].set_title("Silhouette Analysis for Optimal k", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=10)
    axes[1].set_ylabel("Silhouette Score", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Spatial Clusters with Centroids
    scatter = axes[2].scatter(
        coords[:, 1], coords[:, 0], c=final_labels, cmap="tab10", alpha=0.65, s=30, label="Citizen Complaints"
    )
    axes[2].scatter(
        centroids[:, 1], centroids[:, 0], c="red", marker="*", s=220, edgecolors="black", label="Hotspot Centroids"
    )
    axes[2].set_title(f"Discovered Civic Failure Hotspots (k={optimal_k})", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Longitude (°E)", fontsize=10)
    axes[2].set_ylabel("Latitude (°N)", fontsize=10)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "06_kmeans_hotspots_elbow_silhouette.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved K-Means hotspot & evaluation plot to: {plot_path}")

    return final_kmeans, optimal_k


def run_all_classification_clustering_experiments():
    """Run all Step 3 experiments."""
    clf, scaler = evaluate_logistic_regression()
    nb, vec = evaluate_naive_bayes()
    kmeans, k = evaluate_kmeans_clustering()
    print("\n[SUCCESS] Stage 2 Classification & Clustering Suite fully executed and verified.")


if __name__ == "__main__":
    run_all_classification_clustering_experiments()

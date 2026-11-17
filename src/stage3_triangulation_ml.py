"""
Stage 4: Advanced Classifiers & Ensembles (Syllabus Items 8, 9, 10, 11)
------------------------------------------------------------------------
8. Linear and Non-linear SVM using Polynomial and RBF kernels.
9. Decision Tree using CART algorithm: Gini Impurity vs Entropy comparison and tree visualization.
10. Random Forest classifier: Performance evaluation and Feature Importance ranking.
11. Boosting technique (AdaBoost & Gradient Boosting): Model performance comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def ensure_output_dir():
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)


def load_triangulated_features():
    """Load the full multi-tier infrastructure dataset."""
    data_path = os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv")
    df = pd.read_csv(data_path)

    feature_cols = [
        "sanctioned_budget_cr",
        "length_km",
        "num_major_junctions",
        "past_contractor_delays",
        "contractor_experience_yrs",
        "delay_months",
        "disbursed_funds_pct",
        "ground_defect_density",
        "citizen_complaint_count"
    ]
    X = df[feature_cols]
    y = df["is_discrepancy"]
    return X, y, df, feature_cols


# --------------------------------------------------------------------------
# Item 8: Linear and Nonlinear SVM (Polynomial & RBF Kernels)
# --------------------------------------------------------------------------
def evaluate_svm_kernels(df):
    """
    Train Linear, Polynomial, and RBF kernel SVMs on two primary indicators:
    Ground Defect Density and Delay Months, and plot decision boundaries.
    """
    ensure_output_dir()
    X_2d = df[["ground_defect_density", "delay_months"]].values
    y = df["is_discrepancy"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42, stratify=y
    )

    kernels = [
        ("Linear SVM", SVC(kernel="linear", C=1.0, probability=True, random_state=42)),
        ("Polynomial Kernel (d=3)", SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=42)),
        ("RBF Kernel (Gaussian)", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_min, x_max = X_scaled[:, 0].min() - 0.7, X_scaled[:, 0].max() + 0.7
    y_min, y_max = X_scaled[:, 1].min() - 0.7, X_scaled[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    print("\n=======================================================")
    print("8. SUPPORT VECTOR MACHINES: LINEAR VS NONLINEAR KERNELS")
    print("=======================================================")

    results = {}
    for idx, (name, clf) in enumerate(kernels):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        n_sv = len(clf.support_)
        results[name] = {"accuracy": acc, "n_support_vectors": n_sv, "model": clf}

        print(f"{name:25s} -> Accuracy: {acc*100:.2f}% | Support Vectors: {n_sv}")

        # Decision boundary plot
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        axes[idx].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

        # Plot training points & support vectors
        axes[idx].scatter(
            X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40, alpha=0.8
        )
        axes[idx].scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=120,
            facecolors="none",
            edgecolors="gold",
            linewidths=1.8,
            label=f"Support Vectors ({n_sv})"
        )
        axes[idx].set_title(f"{name}\nAcc: {acc*100:.1f}% | SVs: {n_sv}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Std. Ground Defect Density", fontsize=9)
        axes[idx].set_ylabel("Std. Project Delay Months", fontsize=9)
        axes[idx].grid(True, linestyle=":", alpha=0.5)
        axes[idx].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "08_svm_linear_poly_rbf_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved SVM kernels comparison plot to: {plot_path}")

    return results


# --------------------------------------------------------------------------
# Item 9: Decision Tree (CART) - Gini Impurity vs Entropy Comparison
# --------------------------------------------------------------------------
def evaluate_decision_tree(X, y, feature_names):
    """
    Train CART Decision Tree using both Gini Impurity and Entropy (Information Gain),
    compare their splitting behavior, and visualize the tree structures.
    """
    ensure_output_dir()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)

    tree_gini.fit(X_train, y_train)
    tree_entropy.fit(X_train, y_train)

    acc_gini = accuracy_score(y_test, tree_gini.predict(X_test))
    acc_entropy = accuracy_score(y_test, tree_entropy.predict(X_test))

    print("\n=======================================================")
    print("9. DECISION TREE (CART): GINI IMPURITY VS ENTROPY")
    print("=======================================================")
    print(f"Gini Impurity Tree Accuracy  : {acc_gini * 100:.2f}% (Tree depth: {tree_gini.get_depth()}, Leaves: {tree_gini.get_n_leaves()})")
    print(f"Entropy Tree Accuracy        : {acc_entropy * 100:.2f}% (Tree depth: {tree_entropy.get_depth()}, Leaves: {tree_entropy.get_n_leaves()})")

    # Plot both trees side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    plot_tree(
        tree_gini,
        ax=axes[0],
        feature_names=feature_names,
        class_names=["Normal (0)", "Discrepancy (1)"],
        filled=True,
        rounded=True,
        fontsize=8
    )
    axes[0].set_title(f"CART Decision Tree (Gini Impurity, Acc: {acc_gini*100:.1f}%)", fontsize=11, fontweight="bold")

    plot_tree(
        tree_entropy,
        ax=axes[1],
        feature_names=feature_names,
        class_names=["Normal (0)", "Discrepancy (1)"],
        filled=True,
        rounded=True,
        fontsize=8
    )
    axes[1].set_title(f"CART Decision Tree (Entropy / Info Gain, Acc: {acc_entropy*100:.1f}%)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "09_decision_tree_cart_gini_vs_entropy.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Decision Tree Gini vs Entropy visualization to: {plot_path}")

    return tree_gini, tree_entropy


# --------------------------------------------------------------------------
# Item 10: Random Forest & Feature Importance
# --------------------------------------------------------------------------
def evaluate_random_forest(X, y, feature_names):
    """
    Train Random Forest ensemble and evaluate Out-Of-Bag score, accuracy,
    and Mean Decrease in Impurity (MDI) Feature Importance.
    """
    ensure_output_dir()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    oob = rf.oob_score_

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    print("\n=======================================================")
    print("10. RANDOM FOREST & FEATURE IMPORTANCE RANKING")
    print("=======================================================")
    print(f"Random Forest Test Accuracy : {acc * 100:.2f}%")
    print(f"Out-of-Bag (OOB) Score      : {oob * 100:.2f}%")
    print("\nFeature Importance Ranking (Mean Decrease in Impurity):")
    for feat, imp in zip(sorted_features, sorted_importances):
        print(f"  - {feat:26s}: {imp * 100:.2f}%")

    # Plot Horizontal Bar Chart
    plt.figure(figsize=(10, 5))
    y_pos = np.arange(len(sorted_features))
    plt.barh(y_pos, sorted_importances[::-1] * 100, color="teal", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, sorted_features[::-1], fontsize=9)
    plt.xlabel("Feature Importance (%)", fontsize=10)
    plt.title("Random Forest: Top Mathematical Drivers of Infrastructure Discrepancies", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    plot_path = os.path.join("outputs", "figures", "10_random_forest_feature_importance.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Random Forest Feature Importance plot to: {plot_path}")

    return rf, sorted_features, sorted_importances


# --------------------------------------------------------------------------
# Item 11: Boosting Technique & Ensemble Model Comparison
# --------------------------------------------------------------------------
def evaluate_boosting_and_comparison(X, y, feature_names):
    """
    Train AdaBoost and Gradient Boosting classifiers and benchmark performance
    against a single Decision Tree and Random Forest.
    """
    ensure_output_dir()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        "Single CART Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost Classifier": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    }

    metrics = []

    print("\n=======================================================")
    print("11. BOOSTING & ENSEMBLE PERFORMANCE BENCHMARK")
    print("=======================================================")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_prob)

        metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc_score
        })
        print(f"{name:26s} | Acc: {acc*100:5.2f}% | Prec: {prec:5.3f} | Rec: {rec:5.3f} | F1: {f1:5.3f} | AUC: {auc_score:5.3f}")

    df_metrics = pd.DataFrame(metrics)

    # Plot Comparative Bar Chart
    plt.figure(figsize=(11, 5))
    x = np.arange(len(df_metrics))
    width = 0.18

    plt.bar(x - 1.5 * width, df_metrics["Accuracy"] * 100, width, label="Accuracy (%)", color="navy")
    plt.bar(x - 0.5 * width, df_metrics["Precision"] * 100, width, label="Precision (%)", color="forestgreen")
    plt.bar(x + 0.5 * width, df_metrics["Recall"] * 100, width, label="Recall (%)", color="coral")
    plt.bar(x + 1.5 * width, df_metrics["F1-Score"] * 100, width, label="F1-Score (%)", color="crimson")

    plt.xticks(x, df_metrics["Model"], fontsize=9, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=10)
    plt.title("Comparative Performance: Single Tree vs. Bagging (RF) vs. Boosting (AdaBoost & Gradient Boosting)", fontsize=11, fontweight="bold")
    plt.ylim(0, 115)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    plot_path = os.path.join("outputs", "figures", "11_ensemble_models_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Ensemble models comparison plot to: {plot_path}")

    return df_metrics


def run_all_advanced_ml_experiments():
    """Run all Step 5 experiments."""
    X, y, df, feature_names = load_triangulated_features()
    svm_res = evaluate_svm_kernels(df)
    tree_g, tree_e = evaluate_decision_tree(X, y, feature_names)
    rf, feats, imps = evaluate_random_forest(X, y, feature_names)
    df_metrics = evaluate_boosting_and_comparison(X, y, feature_names)
    print("\n[SUCCESS] Stage 4 Advanced Classifiers & Ensembles fully executed and verified.")


if __name__ == "__main__":
    run_all_advanced_ml_experiments()

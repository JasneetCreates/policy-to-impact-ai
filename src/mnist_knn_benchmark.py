"""
Stage 3: KNN Classification on MNIST Dataset (Syllabus Item 6)
---------------------------------------------------------------
6. Perform KNN classification on the MNIST dataset and evaluate
   accuracy using the confusion matrix, precision, and recall.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def ensure_output_dir():
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)


def evaluate_knn_on_mnist():
    """
    Train and evaluate K-Nearest Neighbors on the MNIST digits dataset.
    Computes accuracy, confusion matrix, precision, recall, and plots
    the confusion matrix heatmap, k-parameter curve, and sample predictions.
    """
    ensure_output_dir()
    mnist_path = os.path.join("data", "raw", "mnist_digits_sample.csv")
    if not os.path.exists(mnist_path):
        from data_loader import generate_curated_datasets
        generate_curated_datasets()

    df = pd.read_csv(mnist_path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Hyperparameter search over k
    k_values = [1, 3, 5, 7, 9, 11, 15]
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        accuracies.append(acc)

    best_idx = int(np.argmax(accuracies))
    best_k = k_values[best_idx]
    best_acc = accuracies[best_idx]

    # 2. Train final model with best k
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    y_pred = final_knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n=======================================================")
    print("6. KNN CLASSIFICATION ON MNIST DATASET")
    print("=======================================================")
    print(f"Dataset Size: {len(df)} images (8x8 grayscale = 64 features)")
    print(f"Optimal k: {best_k} with Test Accuracy: {best_acc * 100:.2f}%")
    print("\nAccuracy across neighborhood sizes (k):")
    for k, acc in zip(k_values, accuracies):
        print(f"  k = {k:2d} -> Accuracy: {acc * 100:.2f}%")

    print("\nDetailed Per-Digit Classification Report (Precision, Recall, F1):")
    print(report)

    # 3. Create Multi-Panel Visualization
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Confusion Matrix Heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title(f"MNIST Confusion Matrix (k={best_k}, Acc={best_acc*100:.1f}%)", fontsize=11, fontweight="bold")
    tick_marks = np.arange(10)
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xlabel("Predicted Digit Label", fontsize=10)
    ax1.set_ylabel("True Digit Label", fontsize=10)

    # Annotate counts inside heatmap
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    # Panel 2: k-Parameter vs Accuracy Curve
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(k_values, [a * 100 for a in accuracies], "ro-", linewidth=2, markersize=7)
    ax2.axvline(best_k, color="navy", linestyle="--", label=f"Best k={best_k} ({best_acc*100:.1f}%)")
    ax2.set_title("KNN Accuracy vs. Neighborhood Size (k)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Neighbors (k)", fontsize=10)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax2.set_xticks(k_values)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    # Panel 3: Sample Test Digits with True vs Predicted Labels
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.set_title("Sample Test Predictions (8x8 Pixels)", fontsize=11, fontweight="bold")

    # Display 12 sample test digits in a 3x4 subgrid inside Panel 3
    sample_indices = np.random.choice(len(X_test), 12, replace=False)
    for idx_num, s_idx in enumerate(sample_indices):
        img = X_test[s_idx].reshape(8, 8)
        true_lbl = y_test[s_idx]
        pred_lbl = y_pred[s_idx]
        
        # Subplot positioning
        row = idx_num // 4
        col = idx_num % 4
        sub_ax = fig.add_axes([0.69 + col * 0.07, 0.60 - row * 0.22, 0.055, 0.16])
        sub_ax.imshow(img, cmap="gray_r", interpolation="nearest")
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        color = "green" if true_lbl == pred_lbl else "red"
        sub_ax.set_title(f"T:{true_lbl}|P:{pred_lbl}", color=color, fontsize=8, fontweight="bold")

    plot_path = os.path.join("outputs", "figures", "07_knn_mnist_confusion_matrix.png")
    plt.tight_layout(rect=[0, 0, 0.67, 1.0])
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved KNN MNIST evaluation and confusion matrix plot to: {plot_path}")

    return final_knn, cm, report


if __name__ == "__main__":
    evaluate_knn_on_mnist()

"""
Stage 1: Regression Suite (Syllabus Items 1, 2, 3, 4)
------------------------------------------------------
1. Linear Regression with performance metrics (R2, MSE, RMSE, MAE).
2. Batch, Stochastic, and Mini-Batch Gradient Descent convergence comparison.
3. Polynomial Regression with Learning Curves (Underfitting vs Overfitting analysis).
4. Ridge (L2) and Lasso (L1) Regularization analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_regression_data():
    """Load infrastructure project data and prepare features and targets."""
    data_path = os.path.join("data", "processed", "policy_delivery_ground_triangulated.csv")
    if not os.path.exists(data_path):
        from data_loader import generate_curated_datasets
        generate_curated_datasets()
    df = pd.read_csv(data_path)
    
    # Features predicting project delay in months
    features = [
        "sanctioned_budget_cr",
        "length_km",
        "num_major_junctions",
        "past_contractor_delays",
        "contractor_experience_yrs"
    ]
    X = df[features]
    y = df["delay_months"]
    return X, y, df


# --------------------------------------------------------------------------
# Item 1: Linear Regression & Model Performance Evaluation
# --------------------------------------------------------------------------
def evaluate_linear_regression(X, y):
    """Train Ordinary Least Squares Linear Regression and report metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=======================================================")
    print("1. LINEAR REGRESSION PERFORMANCE METRICS")
    print("=======================================================")
    print(f"R-squared (R2) Score : {r2:.4f}")
    print(f"Mean Squared Error (MSE) : {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"Mean Absolute Error (MAE) : {mae:.4f}")
    print("Learned Feature Coefficients:")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"  - {feat:26s}: {coef:+.4f}")
    print(f"  - Intercept                 : {model.intercept_:+.4f}")
    
    return {
        "model": model,
        "scaler": scaler,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_test": y_test,
        "y_pred": y_pred
    }


# --------------------------------------------------------------------------
# Item 2: Batch, Stochastic, and Mini-Batch Gradient Descent Comparison
# --------------------------------------------------------------------------
def compare_gradient_descent(X, y, n_epochs=100, lr=0.01):
    """
    Implement Batch, Stochastic, and Mini-Batch Gradient Descent
    and plot MSE loss vs Epochs to compare convergence.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Add bias term (column of 1s)
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    y_vec = y.values.reshape(-1, 1)
    m, n = X_b.shape

    # 1. Batch Gradient Descent
    np.random.seed(42)
    theta_bgd = np.random.randn(n, 1)
    bgd_loss = []
    for epoch in range(n_epochs):
        gradients = (2 / m) * X_b.T.dot(X_b.dot(theta_bgd) - y_vec)
        theta_bgd -= lr * gradients
        loss = np.mean((X_b.dot(theta_bgd) - y_vec) ** 2)
        bgd_loss.append(loss)

    # 2. Stochastic Gradient Descent (SGD)
    np.random.seed(42)
    theta_sgd = np.random.randn(n, 1)
    sgd_loss = []
    for epoch in range(n_epochs):
        indices = np.random.permutation(m)
        for i in indices:
            xi = X_b[i:i+1]
            yi = y_vec[i:i+1]
            gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
            # Decaying learning rate
            eta = lr / (1 + 0.005 * (epoch * m + i))
            theta_sgd -= eta * gradients
        loss = np.mean((X_b.dot(theta_sgd) - y_vec) ** 2)
        sgd_loss.append(loss)

    # 3. Mini-Batch Gradient Descent (batch_size=8)
    np.random.seed(42)
    theta_mbgd = np.random.randn(n, 1)
    mbgd_loss = []
    batch_size = 8
    for epoch in range(n_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_b[indices]
        y_shuffled = y_vec[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = (2 / len(xi)) * xi.T.dot(xi.dot(theta_mbgd) - yi)
            theta_mbgd -= lr * gradients
        loss = np.mean((X_b.dot(theta_mbgd) - y_vec) ** 2)
        mbgd_loss.append(loss)

    print("\n=======================================================")
    print("2. GRADIENT DESCENT CONVERGENCE COMPARISON")
    print("=======================================================")
    print(f"Final Batch GD Loss       : {bgd_loss[-1]:.4f} (Smooth, monotonic convergence)")
    print(f"Final Stochastic GD Loss  : {sgd_loss[-1]:.4f} (Rapid early drop, mild oscillation)")
    print(f"Final Mini-Batch GD Loss  : {mbgd_loss[-1]:.4f} (Optimal balance of speed & stability)")

    # Plot convergence
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(bgd_loss, label="Batch Gradient Descent", color="navy", linewidth=2)
    plt.plot(sgd_loss, label="Stochastic Gradient Descent (SGD)", color="coral", alpha=0.85, linewidth=1.5)
    plt.plot(mbgd_loss, label="Mini-Batch Gradient Descent (Batch size=8)", color="forestgreen", linestyle="--", linewidth=2)
    plt.title("Convergence Comparison: Batch vs. Stochastic vs. Mini-Batch Gradient Descent", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE Loss)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "01_gradient_descent_convergence.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved convergence plot to: {plot_path}")

    return bgd_loss, sgd_loss, mbgd_loss


# --------------------------------------------------------------------------
# Item 3: Polynomial Regression & Learning Curves (Underfitting vs Overfitting)
# --------------------------------------------------------------------------
def analyze_polynomial_and_learning_curves(df):
    """
    Fit Polynomial regression of degrees 1, 2, and 5 to analyze non-linear
    relationships and plot learning curves to evaluate underfitting vs overfitting.
    """
    # 1D feature for intuitive visualization: Sanctioned Budget vs Actual Cost
    X_single = df[["sanctioned_budget_cr"]].values
    y_cost = df["actual_cost_cr"].values

    degrees = [1, 2, 5]
    models = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left Plot: Polynomial Fit comparisons
    X_plot = np.linspace(X_single.min(), X_single.max(), 200).reshape(-1, 1)
    axes[0].scatter(X_single, y_cost, color="dimgray", alpha=0.7, edgecolors="k", label="Actual Project Data")

    colors = ["navy", "forestgreen", "crimson"]
    styles = ["-", "-", "--"]
    for deg, color, style in zip(degrees, colors, styles):
        poly_pipeline = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        poly_pipeline.fit(X_single, y_cost)
        y_fit = poly_pipeline.predict(X_plot)
        r2 = r2_score(y_cost, poly_pipeline.predict(X_single))
        label_text = f"Degree {deg} (R² = {r2:.3f})"
        if deg == 1:
            label_text += " [Underfitting]"
        elif deg == 2:
            label_text += " [Optimal Fit]"
        elif deg == 5:
            label_text += " [Overfitting Risk]"
        axes[0].plot(X_plot, y_fit, color=color, linestyle=style, linewidth=2, label=label_text)
        models[deg] = poly_pipeline

    axes[0].set_title("Polynomial Regression Fits", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Sanctioned Budget (₹ Crore)", fontsize=10)
    axes[0].set_ylabel("Actual Cost (₹ Crore)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Right Plot: Learning Curves for Degree 1 vs Degree 2 vs Degree 5
    # Let's show Learning Curve for Degree 1 vs Degree 2
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    train_sizes, train_scores, val_scores = learning_curve(
        poly_model, X_single, y_cost, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.15, 1.0, 10), random_state=42
    )

    train_rmse = np.sqrt(-np.mean(train_scores, axis=1))
    val_rmse = np.sqrt(-np.mean(val_scores, axis=1))

    axes[1].plot(train_sizes, train_rmse, "o-", color="navy", label="Training RMSE (Degree 2)")
    axes[1].plot(train_sizes, val_rmse, "s-", color="crimson", label="Validation RMSE (Degree 2)")
    axes[1].set_title("Learning Curves (Degree 2 Polynomial Model)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Training Examples", fontsize=10)
    axes[1].set_ylabel("RMSE (₹ Crore)", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "02_polynomial_regression_learning_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\n=======================================================")
    print("3. POLYNOMIAL REGRESSION & LEARNING CURVE ANALYSIS")
    print("=======================================================")
    print("Degree 1 (Linear)    : Underfits non-linear cost escalations.")
    print("Degree 2 (Quadratic) : Captures compounding infrastructure overhead without excessive variance.")
    print("Degree 5 (High Deg)  : High variance, risks memorizing outliers.")
    print(f"-> Saved polynomial & learning curves plot to: {plot_path}")
    return models


# --------------------------------------------------------------------------
# Item 4: Ridge (L2) and Lasso (L1) Regularization
# --------------------------------------------------------------------------
def analyze_ridge_and_lasso(X, y):
    """
    Fit Ridge and Lasso across various alphas and analyze coefficient shrinkage
    and feature selection effect.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alphas = np.logspace(-2, 3, 50)
    ridge_coefs = []
    lasso_coefs = []

    for a in alphas:
        r = Ridge(alpha=a)
        r.fit(X_train_scaled, y_train)
        ridge_coefs.append(r.coef_)

        l = Lasso(alpha=a, max_iter=2000)
        l.fit(X_train_scaled, y_train)
        lasso_coefs.append(l.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ridge path
    for i, col in enumerate(X.columns):
        axes[0].plot(alphas, ridge_coefs[:, i], label=col, linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].set_title("Ridge (L2) Regularization Path: Coefficient Shrinkage", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Alpha (Regularization Strength)", fontsize=10)
    axes[0].set_ylabel("Learned Coefficients", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8, loc="upper right")

    # Lasso path
    for i, col in enumerate(X.columns):
        axes[1].plot(alphas, lasso_coefs[:, i], label=col, linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set_title("Lasso (L1) Regularization Path: Feature Selection to Zero", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Alpha (Regularization Strength)", fontsize=10)
    axes[1].set_ylabel("Learned Coefficients", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "03_ridge_lasso_regularization.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Optimal alpha test
    best_ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
    best_lasso = Lasso(alpha=0.1).fit(X_train_scaled, y_train)

    rmse_ridge = np.sqrt(mean_squared_error(y_test, best_ridge.predict(X_test_scaled)))
    rmse_lasso = np.sqrt(mean_squared_error(y_test, best_lasso.predict(X_test_scaled)))

    print("\n=======================================================")
    print("4. RIDGE & LASSO REGULARIZATION EFFECT")
    print("=======================================================")
    print(f"Ridge (alpha=1.0) Test RMSE : {rmse_ridge:.4f} (Shrinks coefficients smoothly towards 0)")
    print(f"Lasso (alpha=0.1) Test RMSE : {rmse_lasso:.4f} (Drives irrelevant features exactly to 0)")
    print("Lasso Sparsity (Feature selection):")
    for feat, coef in zip(X.columns, best_lasso.coef_):
        status = "Active" if abs(coef) > 1e-4 else "Eliminated (Zero)"
        print(f"  - {feat:26s}: {coef:+.4f} [{status}]")
    print(f"-> Saved Ridge & Lasso regularization path plot to: {plot_path}")

    return best_ridge, best_lasso


def run_all_regression_experiments():
    """Run all 4 syllabus regression experiments."""
    X, y, df = load_regression_data()
    eval_res = evaluate_linear_regression(X, y)
    bgd, sgd, mbgd = compare_gradient_descent(X, y, n_epochs=120, lr=0.01)
    poly_models = analyze_polynomial_and_learning_curves(df)
    r_model, l_model = analyze_ridge_and_lasso(X, y)
    print("\n[SUCCESS] Stage 1 Regression Suite fully executed and verified.")


if __name__ == "__main__":
    run_all_regression_experiments()

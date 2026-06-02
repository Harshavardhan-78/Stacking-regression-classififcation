"""
train_model_regression.py
──────────────────────────
Run directly to train and save the stacking regressor:
    python train_model_regression.py
"""

import os
import pickle
import numpy as np
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Path constants ─────────────────────────────────────────────────────────────
MODELS_DIR      = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "regression_best.pkl")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_regression_data(test_size=0.2):
    """
    Load Breast Cancer data for regression.
    Target: radius_mean (continuous)
    """
    df = pd.read_csv(os.path.join("data", "data.csv"))
    df = df.dropna(axis=1, how="all")

    target_col = "radius_mean"
    X = df.drop(columns=["id", "diagnosis", target_col])
    y = df[target_col]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# ── Pickle helpers ─────────────────────────────────────────────────────────────
def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, filename=None):
    """Save a fitted model to models/. Returns the filepath."""
    _ensure_models_dir()
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"regression_{timestamp}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[Regression] Saved  → {filepath}")
    return filepath


def load_model(filename="regression_best.pkl"):
    """Load a saved model from models/."""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No model found at '{filepath}'. "
            "Run train_model_regression.py first."
        )
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[Regression] Loaded ← {filepath}")
    return model


def list_saved_models():
    """Return sorted list of regression .pkl files in models/."""
    _ensure_models_dir()
    return sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and f.startswith("regression")
    )


# ── Core training function ─────────────────────────────────────────────────────
def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    knn_neighbors=5,
    tree_depth=5,
    rf_estimators=100,
    alpha=1.0,
    save=True,
    model_filename=None,
):
    """
    Train a Stacking Regressor and optionally pickle it.

    Base learners : Ridge, Lasso, Decision Tree, KNN, SVR
    Meta-learner  : Random Forest

    Returns
    -------
    model, y_pred, metrics  (metrics = dict with MAE, RMSE, R2)
    """
    base_learners = [
        ("ridge", Ridge(alpha=alpha)),
        ("lasso", Lasso(alpha=alpha, max_iter=5000)),
        ("dt",    DecisionTreeRegressor(max_depth=tree_depth)),
        ("knn",   KNeighborsRegressor(n_neighbors=knn_neighbors)),
        ("svr",   SVR()),
    ]

    meta_learner = RandomForestRegressor(
        n_estimators=rf_estimators, random_state=42
    )

    model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False,
    )

    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
    }

    if save:
        save_model(model, model_filename)

    return model, y_pred, metrics


def get_base_learner_metrics(
    X_train, y_train, X_test, y_test, tree_depth, knn_neighbors, alpha
):
    """Train each base learner individually and return their R² scores."""
    learners = {
        "Ridge":         Ridge(alpha=alpha),
        "Lasso":         Lasso(alpha=alpha, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=tree_depth),
        "KNN":           KNeighborsRegressor(n_neighbors=knn_neighbors),
        "SVR":           SVR(),
    }
    results = {}
    for name, reg in learners.items():
        reg.fit(X_train, y_train)
        results[name] = r2_score(y_test, reg.predict(X_test))
    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data ...")
    X_train, X_test, y_train, y_test = load_regression_data(test_size=0.2)

    print("Training stacking regressor ...")
    model, y_pred, metrics = train_model(
        X_train, y_train, X_test, y_test,
        knn_neighbors=5,
        tree_depth=5,
        rf_estimators=100,
        alpha=1.0,
        save=True,
        model_filename="regression_best.pkl",
    )

    print(f"R²   : {metrics['R2']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"Saved to : {BEST_MODEL_PATH}")

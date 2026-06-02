"""
train_model_classification.py
──────────────────────────────
Run directly to train and save the stacking classifier:
    python train_model_classification.py
"""

import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── Path constants ─────────────────────────────────────────────────────────────
MODELS_DIR      = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "classification_best.pkl")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_classification_data(test_size=0.2):
    """
    Load Breast Cancer data for classification.
    Target: diagnosis  (M=1 Malignant, B=0 Benign)
    """
    df = pd.read_csv(os.path.join("data", "data.csv"))
    df = df.dropna(axis=1, how="all")

    X = df.drop(["id", "diagnosis"], axis=1)
    y = df["diagnosis"].map({"M": 1, "B": 0})

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
        filename  = f"classification_{timestamp}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[Classification] Saved  → {filepath}")
    return filepath


def load_model(filename="classification_best.pkl"):
    """Load a saved model from models/."""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No model found at '{filepath}'. "
            "Run train_model_classification.py first."
        )
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[Classification] Loaded ← {filepath}")
    return model


def list_saved_models():
    """Return sorted list of classification .pkl files in models/."""
    _ensure_models_dir()
    return sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and f.startswith("classification")
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
    save=True,
    model_filename=None,
):
    """
    Train a Stacking Classifier and optionally pickle it.

    Base learners : Logistic Regression, Decision Tree, KNN, SVM
    Meta-learner  : Random Forest

    Returns
    -------
    model, y_pred, accuracy
    """
    base_learners = [
        ("lr",  LogisticRegression(max_iter=1000)),
        ("dt",  DecisionTreeClassifier(max_depth=tree_depth)),
        ("knn", KNeighborsClassifier(n_neighbors=knn_neighbors)),
        ("svm", SVC(probability=True)),
    ]

    meta_learner = RandomForestClassifier(
        n_estimators=rf_estimators, random_state=42
    )

    model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False,
    )

    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if save:
        save_model(model, model_filename)

    return model, y_pred, accuracy


def get_base_learner_accuracies(
    X_train, y_train, X_test, y_test, tree_depth, knn_neighbors
):
    """Train each base learner individually and return their accuracies."""
    learners = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(max_depth=tree_depth),
        "KNN":                 KNeighborsClassifier(n_neighbors=knn_neighbors),
        "SVM":                 SVC(),
    }
    results = {}
    for name, clf in learners.items():
        clf.fit(X_train, y_train)
        results[name] = accuracy_score(y_test, clf.predict(X_test))
    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data ...")
    X_train, X_test, y_train, y_test = load_classification_data(test_size=0.2)

    print("Training stacking classifier ...")
    model, y_pred, accuracy = train_model(
        X_train, y_train, X_test, y_test,
        knn_neighbors=5,
        tree_depth=5,
        rf_estimators=100,
        save=True,
        model_filename="classification_best.pkl",
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Saved to : {BEST_MODEL_PATH}")

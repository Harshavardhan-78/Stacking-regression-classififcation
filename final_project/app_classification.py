"""
app_classification.py
─────────────────────
Streamlit dashboard for Breast Cancer Classification.

Workflow
--------
1. Run once:  python train_model_classification.py
2. Then run:  streamlit run app_classification.py
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from train_model_classification import (
    load_classification_data,
    load_model,
    train_model,
    get_base_learner_accuracies,
    list_saved_models,
    BEST_MODEL_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stacking Classifier — Breast Cancer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Breast Cancer Classification — Stacking Ensemble")
st.markdown(
    "Predicts whether a tumour is **Malignant (1)** or **Benign (0)**. "
    "Train once from the terminal, then explore results here."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

test_size     = st.sidebar.slider("Test Size",                0.1, 0.5, 0.2)
knn_neighbors = st.sidebar.slider("KNN — Neighbours",          1,  20,  5)
tree_depth    = st.sidebar.slider("Decision Tree — Max Depth", 1,  20,  5)
rf_estimators = st.sidebar.slider("Meta Learner — RF Trees",  10, 300, 100)

retrain = st.sidebar.button("🔄 Retrain & Save Model")

st.sidebar.divider()
st.sidebar.subheader("💾 Saved Models")
saved = list_saved_models()
if saved:
    for f in saved:
        st.sidebar.markdown(f"• `{f}`")
else:
    st.sidebar.info("No saved models yet. Run train_model_classification.py")

# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = load_classification_data(test_size)

# ── Load or retrain model ─────────────────────────────────────────────────────
if retrain or not os.path.exists(BEST_MODEL_PATH):
    if not retrain:
        st.warning("No saved model found — training now. Run `train_model_classification.py` to pre-train.")
    with st.spinner("Training ..."):
        model, y_pred, accuracy = train_model(
            X_train, y_train, X_test, y_test,
            knn_neighbors, tree_depth, rf_estimators,
            save=True,
            model_filename="classification_best.pkl",
        )
    st.success("Model trained and saved to `models/classification_best.pkl`")
else:
    with st.spinner("Loading saved model ..."):
        model    = load_model("classification_best.pkl")
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Stacking Accuracy", f"{accuracy:.4f}")
c2.metric("Train Samples",     len(y_train))
c3.metric("Test Samples",      len(y_test))

st.divider()

# ── Confusion Matrix + Classification Report ──────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted Benign", "Predicted Malignant"],
        y=["Actual Benign",    "Actual Malignant"],
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(margin=dict(t=30, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    rows   = {k: v for k, v in report.items() if isinstance(v, dict)}
    st.dataframe(
        pd.DataFrame(rows).T.style.format("{:.3f}"),
        use_container_width=True,
    )

st.divider()

# ── Trees vs Accuracy sweep ───────────────────────────────────────────────────
st.subheader("Meta Learner Trees vs Accuracy")

with st.spinner("Running sweep ..."):
    sweep_range  = list(range(10, 201, 20))
    sweep_scores = []
    for n in sweep_range:
        _, _, acc = train_model(
            X_train, y_train, X_test, y_test,
            knn_neighbors, tree_depth, n,
            save=False,
        )
        sweep_scores.append(acc)

fig_sweep = px.line(
    x=sweep_range, y=sweep_scores,
    labels={"x": "Number of Trees", "y": "Accuracy"},
    title="Accuracy vs Number of Meta-Learner Trees",
    markers=True,
)
st.plotly_chart(fig_sweep, use_container_width=True)

st.divider()

# ── Base learner comparison ───────────────────────────────────────────────────
st.subheader("Base Learners vs Stacking — Accuracy Comparison")

with st.spinner("Evaluating base learners ..."):
    base_accs = get_base_learner_accuracies(
        X_train, y_train, X_test, y_test, tree_depth, knn_neighbors
    )
base_accs["Stacking (Ensemble)"] = accuracy

fig_bar = px.bar(
    x=list(base_accs.keys()),
    y=list(base_accs.values()),
    labels={"x": "Model", "y": "Accuracy"},
    title="Model Performance Comparison",
    color=list(base_accs.values()),
    color_continuous_scale="Blues",
    text_auto=".3f",
)
fig_bar.update_layout(coloraxis_showscale=False, yaxis_range=[0.8, 1.0])
st.plotly_chart(fig_bar, use_container_width=True)

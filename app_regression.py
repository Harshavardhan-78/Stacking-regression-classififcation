"""
app_regression.py
──────────────────
Streamlit dashboard for Tumour Radius Regression.

Workflow
--------
1. Run once:  python train_model_regression.py
2. Then run:  streamlit run app_regression.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model_regression import (
    load_regression_data,
    load_model,
    train_model,
    get_base_learner_metrics,
    list_saved_models,
    BEST_MODEL_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stacking Regressor — Tumour Radius",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📐 Tumour Radius Prediction — Stacking Regressor")
st.markdown(
    "Predicts the **mean radius** of a breast-cancer tumour (continuous target). "
    "Train once from the terminal, then explore results here."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

test_size     = st.sidebar.slider("Test Size",                  0.1,  0.5,  0.2)
knn_neighbors = st.sidebar.slider("KNN — Neighbours",             1,   20,    5)
tree_depth    = st.sidebar.slider("Decision Tree — Max Depth",    1,   20,    5)
rf_estimators = st.sidebar.slider("Meta Learner — RF Trees",     10,  300,  100)
alpha         = st.sidebar.slider("Ridge / Lasso Alpha",        0.01, 10.0,  1.0)

retrain = st.sidebar.button("🔄 Retrain & Save Model")

st.sidebar.divider()
st.sidebar.subheader("💾 Saved Models")
saved = list_saved_models()
if saved:
    for f in saved:
        st.sidebar.markdown(f"• `{f}`")
else:
    st.sidebar.info("No saved models yet. Run train_model_regression.py")

# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = load_regression_data(test_size)

# ── Load or retrain model ─────────────────────────────────────────────────────
if retrain or not os.path.exists(BEST_MODEL_PATH):
    if not retrain:
        st.warning("No saved model found — training now. Run `train_model_regression.py` to pre-train.")
    with st.spinner("Training ..."):
        model, y_pred, metrics = train_model(
            X_train, y_train, X_test, y_test,
            knn_neighbors, tree_depth, rf_estimators, alpha,
            save=True,
            model_filename="regression_best.pkl",
        )
    st.success("Model trained and saved to `models/regression_best.pkl`")
else:
    with st.spinner("Loading saved model ..."):
        model   = load_model("regression_best.pkl")
        y_pred  = model.predict(X_test)
        metrics = {
            "MAE":  mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2":   r2_score(y_test, y_pred),
        }

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² Score",     f"{metrics['R2']:.4f}")
c2.metric("MAE",          f"{metrics['MAE']:.4f}")
c3.metric("RMSE",         f"{metrics['RMSE']:.4f}")
c4.metric("Test Samples", len(y_test))

st.divider()

# ── Actual vs Predicted + Residuals ───────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Actual vs Predicted")
    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig_scatter = px.scatter(
        scatter_df, x="Actual", y="Predicted",
        title="Actual vs Predicted Radius",
        opacity=0.7,
        trendline="ols",
    )
    mn, mx = scatter_df.min().min(), scatter_df.max().max()
    fig_scatter.add_trace(
        go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                   name="Perfect Fit", line=dict(color="red", dash="dash"))
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with right:
    st.subheader("Residuals Distribution")
    residuals = y_test.values - y_pred
    fig_res = px.histogram(
        x=residuals, nbins=30,
        labels={"x": "Residual (Actual − Predicted)"},
        title="Residuals Histogram",
        color_discrete_sequence=["steelblue"],
    )
    fig_res.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_res, use_container_width=True)

st.divider()

# ── Trees vs R² sweep ─────────────────────────────────────────────────────────
st.subheader("Meta Learner Trees vs R²")

with st.spinner("Running sweep ..."):
    sweep_range  = list(range(10, 201, 20))
    sweep_scores = []
    for n in sweep_range:
        _, _, m = train_model(
            X_train, y_train, X_test, y_test,
            knn_neighbors, tree_depth, n, alpha,
            save=False,
        )
        sweep_scores.append(m["R2"])

fig_sweep = px.line(
    x=sweep_range, y=sweep_scores,
    labels={"x": "Number of Trees", "y": "R²"},
    title="R² vs Number of Meta-Learner Trees",
    markers=True,
)
st.plotly_chart(fig_sweep, use_container_width=True)

st.divider()

# ── Base learner comparison ───────────────────────────────────────────────────
st.subheader("Base Learners vs Stacking — R² Comparison")

with st.spinner("Evaluating base learners ..."):
    base_r2 = get_base_learner_metrics(
        X_train, y_train, X_test, y_test,
        tree_depth, knn_neighbors, alpha,
    )
base_r2["Stacking (Ensemble)"] = metrics["R2"]

fig_bar = px.bar(
    x=list(base_r2.keys()),
    y=list(base_r2.values()),
    labels={"x": "Model", "y": "R² Score"},
    title="R² Score Comparison Across Models",
    color=list(base_r2.values()),
    color_continuous_scale="Blues",
    text_auto=".3f",
)
fig_bar.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_bar, use_container_width=True)

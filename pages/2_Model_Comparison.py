import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Machine Learning Model Comparison")

st.markdown("---")

# Load Metrics
metrics = pd.read_csv("model_metrics.csv")

# ==========================
# BEST MODEL
# ==========================

best_model = metrics.loc[
    metrics["Accuracy"].idxmax()
]

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "🏆 Best Model",
        best_model["Model"]
    )

with c2:
    st.metric(
        "Highest Accuracy",
        f"{best_model['Accuracy']:.4f}"
    )

st.markdown("---")

# ==========================
# METRICS TABLE
# ==========================

st.subheader("Performance Metrics")

st.dataframe(
    metrics.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1"],
        color="lightgreen"
    ),
    use_container_width=True
)

# ==========================
# ACCURACY CHART
# ==========================

st.subheader("Accuracy Comparison")

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    metrics["Model"],
    metrics["Accuracy"]
)

plt.xticks(rotation=20)

ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")

st.pyplot(fig)

# ==========================
# PRECISION COMPARISON
# ==========================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Precision")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        metrics["Model"],
        metrics["Precision"]
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

with col2:

    st.subheader("Recall")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        metrics["Model"],
        metrics["Recall"]
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

# ==========================
# F1 SCORE
# ==========================

st.subheader("F1 Score Comparison")

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    metrics["Model"],
    metrics["F1"]
)

plt.xticks(rotation=20)

ax.set_ylabel("F1 Score")

st.pyplot(fig)

# ==========================
# INSIGHTS
# ==========================

st.markdown("---")

st.subheader("📌 Insights")

st.success(f"""
🏆 Best Performing Model: {best_model['Model']}

📊 Accuracy: {best_model['Accuracy']:.4f}

This model achieved the highest overall performance
for employee attrition prediction and is recommended
for deployment.
""")
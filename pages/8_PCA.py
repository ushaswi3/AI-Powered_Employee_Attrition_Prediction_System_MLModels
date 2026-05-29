import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PCA",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Principal Component Analysis (PCA)")

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

pca = joblib.load("pca.pkl")

st.markdown("---")

c1,c2 = st.columns(2)

with c1:
    st.metric(
        "Original Features",
        df.shape[1]-1
    )

with c2:
    st.metric(
        "Reduced Components",
        pca.n_components_
    )

st.markdown("---")

st.subheader("Explained Variance")

try:

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        range(
            1,
            len(cumulative_variance)+1
        ),
        cumulative_variance,
        marker="o"
    )

    ax.set_xlabel("Components")
    ax.set_ylabel("Variance Retained")

    st.pyplot(fig)

except:
    st.warning(
        "Variance graph unavailable."
    )

st.success("""
PCA reduces dimensionality while preserving
most of the important information.

Benefits:
• Faster Training
• Less Complexity
• Reduced Redundancy
""")
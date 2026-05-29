import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import export_text

st.set_page_config(
    page_title="Decision Tree",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Decision Tree")

metrics = pd.read_csv("model_metrics.csv")

row = metrics[
    metrics["Model"]=="Decision Tree"
].iloc[0]

c1,c2,c3,c4 = st.columns(4)

c1.metric("Accuracy",f"{row['Accuracy']:.4f}")
c2.metric("Precision",f"{row['Precision']:.4f}")
c3.metric("Recall",f"{row['Recall']:.4f}")
c4.metric("F1 Score",f"{row['F1']:.4f}")

st.markdown("---")

st.subheader("HR Decision Rules")

try:

    dt_model = joblib.load("dt_model.pkl")

    df = pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    X = df.drop("Attrition", axis=1)

    rules = export_text(
        dt_model,
        feature_names=list(X.columns)
    )

    st.code(rules)

except:
    st.warning(
        "Decision Tree rules could not be generated."
    )

st.subheader("Business Interpretation")

st.success("""
Decision Trees generate human-readable rules that help HR understand why employees may leave.

Example:

IF Job Satisfaction is low
AND Monthly Income is low

THEN High Attrition Risk
""")
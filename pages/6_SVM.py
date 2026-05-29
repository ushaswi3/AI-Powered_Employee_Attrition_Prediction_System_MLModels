import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SVM",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Support Vector Machine (SVM)")

metrics = pd.read_csv("model_metrics.csv")

row = metrics[
    metrics["Model"]=="SVM"
].iloc[0]

c1,c2,c3,c4 = st.columns(4)

c1.metric("Accuracy",f"{row['Accuracy']:.4f}")
c2.metric("Precision",f"{row['Precision']:.4f}")
c3.metric("Recall",f"{row['Recall']:.4f}")
c4.metric("F1 Score",f"{row['F1']:.4f}")

st.markdown("---")

st.info("""
Support Vector Machine finds the optimal boundary
between employees likely to stay and employees likely
to leave.

SVM performs particularly well on complex classification problems.
""")

st.success("""
Business Use:
HR can use SVM predictions to identify employees
at risk of attrition and take corrective actions.
""")
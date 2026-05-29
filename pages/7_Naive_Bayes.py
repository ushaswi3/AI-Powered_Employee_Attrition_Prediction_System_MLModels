import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Naive Bayes",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Naive Bayes")

metrics = pd.read_csv("model_metrics.csv")

row = metrics[
    metrics["Model"]=="Naive Bayes"
].iloc[0]

c1,c2,c3,c4 = st.columns(4)

c1.metric("Accuracy",f"{row['Accuracy']:.4f}")
c2.metric("Precision",f"{row['Precision']:.4f}")
c3.metric("Recall",f"{row['Recall']:.4f}")
c4.metric("F1 Score",f"{row['F1']:.4f}")

st.markdown("---")

st.subheader("Probability-Based Prediction")

st.info("""
Naive Bayes predicts the probability that an employee
will leave the organization.

Example:

Employee A
Leave Probability = 82%

Employee B
Leave Probability = 18%
""")

st.success("""
HR can prioritize employees with higher attrition
probabilities and provide targeted retention efforts.
""")
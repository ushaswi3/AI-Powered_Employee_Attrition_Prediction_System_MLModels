import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Logistic Regression",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Logistic Regression")

metrics = pd.read_csv("model_metrics.csv")

row = metrics[
    metrics["Model"]=="Logistic Regression"
].iloc[0]

c1,c2,c3,c4 = st.columns(4)

c1.metric("Accuracy",f"{row['Accuracy']:.4f}")
c2.metric("Precision",f"{row['Precision']:.4f}")
c3.metric("Recall",f"{row['Recall']:.4f}")
c4.metric("F1 Score",f"{row['F1']:.4f}")

st.markdown("---")

st.subheader("About Logistic Regression")

st.info("""
Logistic Regression predicts whether an employee is likely to leave or stay.

Output:

• 0 → Employee Stays

• 1 → Employee Leaves

It is a baseline classification algorithm commonly used for binary prediction problems.
""")

st.subheader("Business Use Case")

st.success("""
HR can identify employees likely to leave and take preventive actions before attrition occurs.
""")
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Employee Attrition Prediction System")

st.markdown("""
### Project Overview

This project analyzes employee attrition using the IBM HR Analytics dataset.

### Objectives

- Predict employee attrition
- Identify factors influencing attrition
- Generate HR decision rules
- Compare Machine Learning algorithms
- Perform Dimensionality Reduction using PCA
- Segment Employees using K-Means Clustering

### Dataset

IBM HR Analytics Employee Attrition & Performance Dataset

### Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- Naive Bayes
- PCA
- K-Means Clustering

Use the sidebar to navigate through the dashboard.
""")
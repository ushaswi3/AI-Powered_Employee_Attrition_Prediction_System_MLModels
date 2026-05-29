import streamlit as st

st.set_page_config(
    page_title="Project Report",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Project Report")

st.markdown("---")

# =====================================
# PROJECT OVERVIEW
# =====================================

st.header("📌 Project Overview")

st.write("""
The AI-Powered Employee Attrition Prediction System is designed
to help organizations identify employees who are likely to leave
the company and understand the factors contributing to attrition.

Using Machine Learning techniques, the system predicts employee
attrition, generates HR decision rules, identifies key influencing
factors, and segments employees into meaningful groups.
""")

# =====================================
# PROBLEM STATEMENT
# =====================================

st.header("🎯 Problem Statement")

st.info("""
Organizations lose employees every month, resulting in
increased hiring costs, training expenses, and productivity loss.

The goal is to:

• Predict employee attrition

• Identify factors influencing attrition

• Generate actionable HR insights

• Segment employees into meaningful categories

• Support employee retention strategies
""")

# =====================================
# DATASET
# =====================================

st.header("📊 Dataset Information")

st.write("""
Dataset Used:

IBM HR Analytics Employee Attrition & Performance Dataset

Dataset Characteristics:

• 1470 Employee Records

• 35 Features

• Binary Target Variable (Attrition)

Target Classes:

• Yes → Employee Leaves

• No → Employee Stays
""")

# =====================================
# TECHNOLOGIES
# =====================================

st.header("🛠 Technologies Used")

st.success("""
Programming Language:
• Python

Libraries:
• Pandas
• NumPy
• Matplotlib
• Seaborn
• Scikit-Learn
• Joblib

Deployment:
• Streamlit
""")

# =====================================
# MODELS
# =====================================

st.header("🤖 Machine Learning Models")

st.write("""
The following machine learning algorithms were implemented:

1. Logistic Regression
   • Binary Classification

2. Decision Tree
   • Rule-Based Prediction

3. Random Forest
   • Ensemble Learning

4. Support Vector Machine (SVM)
   • Margin-Based Classification

5. Naive Bayes
   • Probability-Based Prediction

6. PCA
   • Dimensionality Reduction

7. K-Means Clustering
   • Employee Segmentation
""")

# =====================================
# BUSINESS INSIGHTS
# =====================================

st.header("📈 Key Business Insights")

st.warning("""
• Employees with lower salaries exhibit higher attrition.

• Employees with lower experience levels are more likely to leave.

• Attrition differs across departments.

• Job satisfaction strongly influences employee retention.

• Employee segmentation helps identify high-risk groups.

• Machine Learning can assist HR teams in proactive decision making.
""")

# =====================================
# BENEFITS
# =====================================

st.header("✅ Benefits of the System")

st.write("""
• Early identification of at-risk employees

• Reduced employee turnover

• Improved retention strategies

• Data-driven HR decision making

• Better workforce planning

• Increased employee satisfaction
""")

# =====================================
# CONCLUSION
# =====================================

st.header("🏁 Conclusion")

st.success("""
The Employee Attrition Prediction System successfully
applies Machine Learning techniques to analyze employee
behavior and predict attrition risk.

The project demonstrates the use of:

• Exploratory Data Analysis

• Classification Algorithms

• Dimensionality Reduction

• Clustering Techniques

• Interactive Dashboard Deployment

This system can help organizations improve employee
retention and support strategic HR decision making.
""")

st.markdown("---")

st.caption(
    "Developed using Streamlit and Machine Learning"
)
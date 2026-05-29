import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="K-Means Clustering",
    page_icon="🎯",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🎯 Employee Segmentation using K-Means")

st.markdown("""
This module groups employees into similar categories
based on their characteristics and work-related factors.
""")

st.markdown("---")

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# =====================================
# PREPROCESSING
# =====================================

X = pd.get_dummies(
    df.drop("Attrition", axis=1),
    drop_first=True
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================
# TRAIN KMEANS
# =====================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# =====================================
# KPI CARDS
# =====================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.subheader("📊 Cluster Distribution")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Cluster 0",
        int(cluster_counts[0])
    )

with c2:
    st.metric(
        "Cluster 1",
        int(cluster_counts[1])
    )

with c3:
    st.metric(
        "Cluster 2",
        int(cluster_counts[2])
    )

st.markdown("---")

# =====================================
# CLUSTER SUMMARY
# =====================================

st.subheader("📈 Cluster Summary")

cluster_summary = df.groupby(
    "Cluster"
)[
    [
        "MonthlyIncome",
        "TotalWorkingYears",
        "JobSatisfaction"
    ]
].mean().round(2)

st.dataframe(
    cluster_summary,
    use_container_width=True
)

st.markdown("---")

# =====================================
# SEGMENT INTERPRETATION
# =====================================

st.subheader("👥 Employee Segments")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
### Cluster 0

🏆 High Performers

• Higher income

• More experience

• Strong performance
""")

with col2:
    st.warning("""
### Cluster 1

⚠️ At Risk

• Lower satisfaction

• Higher attrition tendency

• Requires HR attention
""")

with col3:
    st.info("""
### Cluster 2

🌱 New Employees

• Less experience

• Recently joined

• Need mentoring
""")

st.markdown("---")

# =====================================
# BUSINESS INSIGHT
# =====================================

st.subheader("📌 Business Insight")

st.info("""
K-Means clustering helps HR identify groups of employees
with similar characteristics.

Benefits:

• Identify high performers

• Detect employees at risk

• Improve retention strategies

• Design personalized HR interventions
""")

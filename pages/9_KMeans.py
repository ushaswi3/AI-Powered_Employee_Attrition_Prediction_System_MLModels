import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="K-Means",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Employee Segmentation using K-Means")

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Encode categorical columns
df_encoded = df.copy()

le = LabelEncoder()

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = le.fit_transform(df_encoded[col])

# Remove target
X = df_encoded.drop(
    "Attrition",
    axis=1
)

# Scale
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)

# KPI Cards
cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

c1,c2,c3 = st.columns(3)

c1.metric("Cluster 0", cluster_counts[0])
c2.metric("Cluster 1", cluster_counts[1])
c3.metric("Cluster 2", cluster_counts[2])

st.markdown("---")

# Summary
cluster_summary = df.groupby(
    "Cluster"
)[
    [
        "MonthlyIncome",
        "TotalWorkingYears",
        "JobSatisfaction"
    ]
].mean()

st.subheader("Cluster Summary")

st.dataframe(
    cluster_summary,
    use_container_width=True
)

st.markdown("---")

st.subheader("Employee Segments")

st.success("""
Cluster 0 → High Performers

Cluster 1 → At Risk Employees

Cluster 2 → New Employees
""")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

[data-testid="metric-container"]{
    background-color:#f0f2f6;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

df = load_data()

# =====================================
# TITLE
# =====================================

st.title("📊 Employee Attrition EDA Dashboard")

st.markdown("---")

# =====================================
# KPI CARDS
# =====================================

total_employees = len(df)

employees_left = len(
    df[df["Attrition"] == "Yes"]
)

employees_stayed = len(
    df[df["Attrition"] == "No"]
)

attrition_rate = round(
    (employees_left / total_employees) * 100,
    2
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Employees",
        total_employees
    )

with col2:
    st.metric(
        "Employees Left",
        employees_left
    )

with col3:
    st.metric(
        "Employees Stayed",
        employees_stayed
    )

with col4:
    st.metric(
        "Attrition Rate",
        f"{attrition_rate}%"
    )

st.markdown("---")

# =====================================
# ROW 1
# =====================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Attrition Distribution")

    attrition = df["Attrition"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))

    ax.pie(
        attrition,
        labels=attrition.index,
        autopct="%1.1f%%"
    )
    ax.axis('equal')
    st.pyplot(fig)

with col2:

    st.subheader("Gender-wise Attrition")

    gender = pd.crosstab(
        df["Gender"],
        df["Attrition"]
    )

    fig, ax = plt.subplots(figsize=(6,4))

    gender.plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=0)

    st.pyplot(fig)

# =====================================
# ROW 2
# =====================================

col3, col4 = st.columns(2)

with col3:

    st.subheader("Department-wise Attrition")

    dept = pd.crosstab(
        df["Department"],
        df["Attrition"]
    )

    fig, ax = plt.subplots(figsize=(6,4))

    dept.plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

with col4:

    st.subheader("Salary Impact")

    fig, ax = plt.subplots(figsize=(5,4))

    sns.boxplot(
        data=df,
        x="Attrition",
        y="MonthlyIncome",
        ax=ax
    )

    st.pyplot(fig)

# =====================================
# ROW 3
# =====================================

col5, col6 = st.columns(2)

with col5:

    st.subheader("Experience Impact")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        data=df,
        x="Attrition",
        y="TotalWorkingYears",
        ax=ax
    )

    st.pyplot(fig)

with col6:

    st.subheader("Attrition Count")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        data=df,
        x="Attrition",
        ax=ax
    )

    st.pyplot(fig)

# =====================================
# INSIGHTS
# =====================================

st.markdown("---")

st.subheader("📌 Key Business Insights")

st.success(f"""

• Overall Attrition Rate: {attrition_rate}%

• Employees with lower salaries are more likely to leave.

• Employees with fewer years of experience show higher attrition.

• Attrition varies significantly across departments.

• Understanding these patterns helps HR identify employees at risk and improve retention strategies.

""")
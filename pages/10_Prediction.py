import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🎯",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🎯 Employee Attrition Prediction System")

st.markdown("""
Use the Random Forest model to predict whether an employee
is likely to leave the organization.
""")

st.markdown("---")

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# =====================================
# ENCODE DATA
# =====================================

df_encoded = df.copy()

for col in df_encoded.columns:
    if not pd.api.types.is_numeric_dtype(df_encoded[col]):
        df_encoded[col] = LabelEncoder().fit_transform(
            df_encoded[col].astype(str)
        )

# =====================================
# LOAD MODEL
# =====================================

rf_model = joblib.load(
    "rf_model.pkl"
)

# =====================================
# EMPLOYEE SELECTION
# =====================================

employee_id = st.selectbox(
    "Select Employee",
    df.index
)

selected = df.iloc[employee_id]

# =====================================
# EMPLOYEE DETAILS
# =====================================

st.subheader("👤 Employee Details")

col1, col2 = st.columns(2)

with col1:

    st.info("Personal Information")

    st.write(f"**Age:** {selected['Age']}")
    st.write(f"**Gender:** {selected['Gender']}")
    st.write(f"**Department:** {selected['Department']}")
    st.write(f"**Job Role:** {selected['JobRole']}")
    st.write(f"**Marital Status:** {selected['MaritalStatus']}")

with col2:

    st.info("Work Information")

    st.write(f"**Monthly Income:** ₹{selected['MonthlyIncome']}")
    st.write(f"**Total Working Years:** {selected['TotalWorkingYears']}")
    st.write(f"**Job Satisfaction:** {selected['JobSatisfaction']}")
    st.write(f"**Work Life Balance:** {selected['WorkLifeBalance']}")
    st.write(f"**OverTime:** {selected['OverTime']}")

st.markdown("---")

# =====================================
# PREDICTION BUTTON
# =====================================

predict = st.button(
    "🔮 Predict Attrition Risk",
    use_container_width=True
)

# =====================================
# PREDICTION
# =====================================

if predict:

    sample = df_encoded.drop(
        "Attrition",
        axis=1
    ).iloc[[employee_id]]

    # Make sure everything is numeric
    sample = sample.astype(float)

    # TEMP DEBUG
    st.write("Feature Data Types:")
    st.write(sample.dtypes)

    prediction = rf_model.predict(sample)

    probability = rf_model.predict_proba(sample)

    risk = probability[0][1] * 100

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Prediction",
            "Leave" if prediction[0] == 1 else "Stay"
        )

    with c2:

        st.metric(
            "Attrition Probability",
            f"{risk:.2f}%"
        )

    st.markdown("---")

    # =====================================
    # RISK CATEGORY
    # =====================================

    if risk >= 70:

        st.error(
            f"⚠️ HIGH ATTRITION RISK ({risk:.2f}%)"
        )

        st.info("""
### Recommended HR Actions

• Conduct employee retention discussions

• Review salary and compensation

• Improve job satisfaction

• Monitor engagement levels

• Provide career growth opportunities
""")

    elif risk >= 40:

        st.warning(
            f"🟡 MEDIUM ATTRITION RISK ({risk:.2f}%)"
        )

        st.info("""
### Recommended HR Actions

• Monitor employee satisfaction

• Collect employee feedback

• Improve engagement activities

• Review work-life balance
""")

    else:

        st.success(
            f"🟢 LOW ATTRITION RISK ({risk:.2f}%)"
        )

        st.success("""
### HR Assessment

Employee appears stable and currently has
a low probability of attrition.

Continue normal employee engagement practices.
""")

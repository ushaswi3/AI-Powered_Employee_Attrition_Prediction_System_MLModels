import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Random Forest",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Random Forest")

metrics = pd.read_csv("model_metrics.csv")

row = metrics[
    metrics["Model"]=="Random Forest"
].iloc[0]

c1,c2,c3,c4 = st.columns(4)

c1.metric("Accuracy",f"{row['Accuracy']:.4f}")
c2.metric("Precision",f"{row['Precision']:.4f}")
c3.metric("Recall",f"{row['Recall']:.4f}")
c4.metric("F1 Score",f"{row['F1']:.4f}")

st.markdown("---")

st.subheader("Feature Importance")

try:

    rf_model = joblib.load("rf_model.pkl")

    df = pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    X = df.drop("Attrition", axis=1)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    top10 = importance.head(10)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        top10["Feature"],
        top10["Importance"]
    )

    plt.gca().invert_yaxis()

    st.pyplot(fig)

    st.dataframe(
        top10,
        use_container_width=True
    )

except:
    st.warning(
        "Feature importance could not be generated."
    )

st.subheader("Business Insight")

st.success("""
Random Forest identifies the most important factors influencing attrition.

HR can focus on these features to improve employee retention.
""")
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from streamlit_shap import st_shap
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# ==============================
# FIX PYTHON PATH FOR src/
# ==============================
ROOT = Path(__file__).resolve().parents[1]

from src.inference import (
    load_model,
    preprocess_input,
    predict_single,
    explain_single
)

# ==============================
# LOAD MODEL ONCE
# ==============================
model = load_model()

# ==============================
# STREAMLIT PAGE SETTINGS
# ==============================
st.set_page_config(page_title="Credit Default Predictor", layout="wide")
st.title("📊 Credit Card Default Risk Prediction App")
st.write("Enter customer data to predict default probability and view explanations.")

# ==============================
# FORM INPUTS
# ==============================
with st.form("customer_form"):
    st.header("Basic Information")
    
    LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", 0, 1000000, 50000)
    AGE = st.number_input("Age", 18, 100, 30)

    SEX = st.selectbox("Sex (1 = Male, 2 = Female)", [1, 2])
    EDUCATION = st.selectbox(
        "Education (1 = Graduate, 2 = University, 3 = High School, 4 = Others)",
        [1, 2, 3, 4]
    )
    MARRIAGE = st.selectbox(
        "Marriage (1 = Married, 2 = Single, 3 = Others)",
        [1, 2, 3]
    )

    st.header("Payment Status (Past 6 Months)")
    PAY_0 = st.number_input("PAY_0", -2, 8, 0)
    PAY_2 = st.number_input("PAY_2", -2, 8, 0)
    PAY_3 = st.number_number_input("PAY_3", -2, 8, 0)
    PAY_4 = st.number_input("PAY_4", -2, 8, 0)
    PAY_5 = st.number_input("PAY_5", -2, 8, 0)
    PAY_6 = st.number_input("PAY_6", -2, 8, 0)

    st.header("Bill Amounts (Past 6 Months)")
    BILL_AMT1 = st.number_input("BILL_AMT1", 0, 1000000, 20000)
    BILL_AMT2 = st.number_input("BILL_AMT2", 0, 1000000, 21000)
    BILL_AMT3 = st.number_input("BILL_AMT3", 0, 1000000, 22000)
    BILL_AMT4 = st.number_input("BILL_AMT4", 0, 1000000, 23000)
    BILL_AMT5 = st.number_input("BILL_AMT5", 0, 1000000, 24000)
    BILL_AMT6 = st.number_input("BILL_AMT6", 0, 1000000, 25000)

    st.header("Payment Amounts (Past 6 Months)")
    PAY_AMT1 = st.number_input("PAY_AMT1", 0, 1000000, 5000)
    PAY_AMT2 = st.number_input("PAY_AMT2", 0, 1000000, 4000)
    PAY_AMT3 = st.number_input("PAY_AMT3", 0, 1000000, 3000)
    PAY_AMT4 = st.number_input("PAY_AMT4", 0, 1000000, 3000)
    PAY_AMT5 = st.number_input("PAY_AMT5", 0, 1000000, 2000)
    PAY_AMT6 = st.number_input("PAY_AMT6", 0, 1000000, 1000)

    submit = st.form_submit_button("Predict Default Risk")

# ==============================
# PROCESS INPUT AND DISPLAY TABS
# ==============================
if submit:
    input_data = {
        "LIMIT_BAL": LIMIT_BAL, "SEX": SEX, "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE, "AGE": AGE,
        "PAY_0": PAY_0, "PAY_2": PAY_2, "PAY_3": PAY_3,
        "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3, "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2, "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4, "PAY_AMT5": PAY_AMT5, "PAY_AMT6": PAY_AMT6,
    }

    result = predict_single(input_data, model)

    df_final = preprocess_input(pd.DataFrame([input_data]))

    explainer, shap_values = explain_single(df_final, model)

    tab1, tab2, tab3 = st.tabs([
        "🔮 Prediction",
        "📘 SHAP Explanation",
        "📊 Global Feature Importance"
    ])

    # TAB 1 — Prediction
    with tab1:
        st.subheader("📌 Prediction Result")

        prob = result["probability"]
        pred = result["prediction"]

        if prob < 0.30:
            color = "green"
            level = "Low Risk"
        elif prob < 0.60:
            color = "orange"
            level = "Medium Risk"
        else:
            color = "red"
            level = "High Risk"

        st.markdown(f"### Probability: **{prob:.4f}**")
        st.markdown(f"### Prediction: **{'Default (1)' if pred==1 else 'No Default (0)'}**")
        st.markdown(f"### Risk Level: <span style='color:{color}; font-size:24px;'>{level}</span>", unsafe_allow_html=True)

    # TAB 2 — SHAP Local Explanation
    with tab2:
        st.subheader("🔍 SHAP Force Plot (Local Explanation)")

        shap.initjs()  # ONLY here

        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            df_final
        )

        st_shap(force_plot, height=300)

    # TAB 3 — Feature Importance
    with tab3:
        st.subheader("📊 Global Feature Importance (Ranked & Styled)")

        importance = model.feature_importances_
        features = df_final.columns

        imp_df = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(imp_df["feature"], imp_df["importance"], color="skyblue")
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Top 15 Important Features", fontsize=14)

        for bar in bars:
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}",
                va="center"
            )

        st.pyplot(fig)

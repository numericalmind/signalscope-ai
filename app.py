import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="SignalScope AI",
    page_icon="🔬",
    layout="wide"
)

#Load trained model
model = joblib.load("rf_high_model.pkl")

# High_level physics features
feature_names = [
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb"
]

# Header
st.title("🔬 SignalScope AI")
st.subheader("Higgs Signal Classification with Physics Engineered Features")

st.write(
    """
    SignalScope AI uses machine learning to estimate whether a particle
    collision event resembles a **Higgs signal** or **background event**.

    This interactive demo uses seven physics-engineered high-level features.
    """
)

st.divider()

# Input section
st.header("Collision Event Features")

inputs = {}

col1, col2 = st.columns(2)

with col1:
    inputs["m_jj"] = st.number_input("m_jj", value=1.0)
    inputs["m_jjj"] = st.number_input("m_jjj", value=1.0)
    inputs["m_lv"] = st.number_input("m_lv", value=1.0)
    inputs["m_jlv"] = st.number_input("m_jlv", value=1.0)

with col2:
    inputs["m_bb"] = st.number_input("m_bb", value=1.0)
    inputs["m_wbb"] = st.number_input("m_wbb", value=1.0)
    inputs["m_wwbb"] = st.number_input("m_wwbb", value=1.0)

st.divider()

# Prediction
if st.button("Analyze Event", type = "primary"):

    input_data =pd.DataFrame(
        [[inputs[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.header("Prediction Results")

    st.metric(
        "Higgs Signal Probability",
        f"{probability * 100:.1f}%"
    )

    if prediction == 1:
        st.success("Predicted Class: Higgs-like Signal Event")
    else:
        st.info("Predicted Class: Background Event")

    st.caption(
        "This result is a machine-learning prediction for educational"
        "and demonstration purposes."
    )

    st.divider()

    # Model performance
    st.header("Model Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Low-Level Features", "AUC 0.665")
    col2.metric("High-Level Features", "AUC 0.758")
    col3.metric("All Features", "AUC 0.793")

    st.write(

    """
    SignalScope AI uses machine learning to estimate whether a particle
    collision event resembles a **Higgs signal** or **background event**.

    This interactive demo uses seven physics-engineered high-level features.
    """
    )

    st.divider()
    st.header("Model Performance Comparison")

    auc_data = pd.DataFrame({
        "Feature Set":[
            "Low-Level Only (21)",
            "High-Level Only(7)",
            "All Features (28)"
        ],
        "ROC-AUC":[
            0.665,
            0.758,
            0.793
        ]
    })

    st.bar_chart(
        auc_data.set_index("Feature Set")
    )

    st.header("High-Level Feature Importance")

    importance_data = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_data.set_index("Feature")
    )
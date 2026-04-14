import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved ML pipeline
model = joblib.load("waterpoint_failure_model.pkl")

st.title("Waterpoint Failure Prediction System for Uganda")

st.write("Select waterpoint details to predict Waterpoint failure risk")

# Input fields

status_id = st.selectbox(
    "Water Availability Status",
    ["","Yes","No","Unknown"]
)

water_source_clean = st.selectbox(
    "Water Source",
    ["","Borehole","Rainwater Harvesting","Piped","Spring","Well", "Sand or Sub-surface Dam","Packaged Water", "Surface Water (River/Stream/Lake/Pond/Dam)", "unknown"]
)

water_tech_category = st.selectbox(
    "Technology Type",
    ["","Motorized Pump","Hand Pump","Public Tapstand","Rope and Bucket", "Unknown"]
)

management_clean = st.selectbox(
    "Management Type",
    ["","Community","Government","Private", "Institutional", "Other", "None"]
)

age_group = st.selectbox(
    "Age of Waterpoint",
    ["","0-5","6-10","11-15","16-20","21-25","26-30","31-35","36-40","46+"]
)

clean_adm1 = st.selectbox(
    "Region",
    ["","Northern","Central","Eastern","Western"]
)


population_per_waterpoint = st.number_input(
    "Population Served",
    min_value=1,
    max_value=10000
)

# Predict button
if st.button("Predict"):

    if "" in [status_id, water_source_clean, water_tech_category, management_clean, age_group, clean_adm1]:
        st.warning("Please select all fields before prediction.")

    else:

        input_df = pd.DataFrame({
            "status_id":[status_id],
            "water_source_clean":[water_source_clean],
            "water_tech_category":[water_tech_category],
            "management_clean":[management_clean],
            "age_group":[age_group],
            "clean_adm1":[clean_adm1],
            "population_per_waterpoint":[population_per_waterpoint]
        })

        prediction = model.predict(input_df)

        probs = model.predict_proba(input_df)

        prob_failed = probs[0][0]
        prob_maintenance = probs[0][1]
        prob_healthy = probs[0][2]

        risk_score = prob_failed + 0.6*prob_maintenance + 0.1*prob_healthy

        if risk_score > 0.7:
            risk_level = "High"
        elif risk_score > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        priority_score = risk_score * np.log1p(population_per_waterpoint)

        st.subheader("Prediction Result")

        status_map = {
            0:"Failed",
            1:"Needs Maintenance",
            2:"Healthy"
        }

        st.write("Predicted Status:", status_map[prediction[0]])
        st.write("Failure Probability:", round(prob_failed,3))
        st.write("Risk Score:", round(risk_score,3))
        st.write("Risk Level:", risk_level)
        st.write("Maintenance Priority Score:", round(priority_score,3))
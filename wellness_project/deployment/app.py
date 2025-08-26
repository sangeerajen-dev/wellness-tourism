import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model (pipeline already has preprocessing)
model_path = hf_hub_download(
    repo_id="sangee-huggingface/wellness-tourism-package",
    filename="wellness-tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Define features exactly as in training
numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
    "NumberOfTrips"
]

categorical_features = [
    "TypeofContact",
    "CityTier",
    "Occupation",
    "MaritalStatus",
    "Gender",
    "ProductPitched",
    "Designation"
]

st.title("Travel Lead Conversion Prediction")

st.write("### Please enter the customer details:")

with st.form("input_form"):
    user_input = {}

    st.subheader("Numeric Features")
    for feature in numeric_features:
        user_input[feature] = st.number_input(f"{feature}", min_value=0, step=1)

    st.subheader("Categorical Features")
    for feature in categorical_features:
        user_input[feature] = st.text_input(f"{feature}")

    # Submit button inside the form
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Create DataFrame with the exact schema
        input_df = pd.DataFrame([user_input])

        # Run prediction
        prediction = model.predict(input_df)[0]
        result = "Converted" if prediction == 1 else "Not Converted"

        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

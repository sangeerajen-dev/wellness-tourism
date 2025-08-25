import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sangee-huggingface/wellness-tourism-package", filename="wellness-tourism_model_v1.joblib")
model = joblib.load(model_path)



# Define features
numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'ProductPitched',
    'NumberOfFollowups',
    'DurationOfPitch',
    'NumberOfTrips'
]

categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'MaritalStatus',
    'Gender',
    'Designation'
]

st.title("Travel Lead Conversion Input Form")

st.write("### Please enter the customer details:")

# Use form for better UI
with st.form("input_form"):
    user_input = {}

    st.subheader("Numeric Features")
    for feature in numeric_features:
        user_input[feature] = st.number_input(f"{feature}", min_value=0, step=1)

    st.subheader("Categorical Features")
    for feature in categorical_features:
        user_input[feature] = st.text_input(f"{feature}")

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Prediction button
if st.button("Predict Failure"):
    try:
        prediction = model.predict(input_df)[0]   # prediction
        result = "Machine Failure" if prediction == 1 else "No Failure"

        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

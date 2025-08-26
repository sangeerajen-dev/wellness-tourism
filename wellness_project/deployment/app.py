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

categorical_features = {
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "CityTier": [1, 2, 3],
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "Gender": ["Male", "Female"],
    "ProductPitched": ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    "Designation": ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
}

st.title("🌍 Travel Lead Conversion Prediction")

st.write("### Please enter the customer details:")

with st.form("input_form"):
    user_input = {}

    st.subheader("Numeric Features")
    for feature in numeric_features:
        user_input[feature] = st.number_input(
            f"{feature}", min_value=0, step=1, value=1
        )

    st.subheader("Categorical Features")
    for feature, options in categorical_features.items():
        user_input[feature] = st.selectbox(f"{feature}", options)

    # Submit button inside the form
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Create DataFrame with the exact schema
        input_df = pd.DataFrame([user_input])

        # Run prediction
        prediction = model.predict(input_df)[0]
        result = "✅ Converted" if prediction == 1 else "❌ Not Converted"

        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

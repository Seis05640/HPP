import streamlit as st
import pandas as pd
import joblib

# Load the model, scaler, and selected feature names
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.markdown("Enter the property details below to estimate its market price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        year_built = st.number_input("📅 Year Built", min_value=1800, max_value=2025, value=2010)
        garage_cars = st.slider("🚗 Garage Capacity", 0, 5, 2)
        amenity_score = st.slider("🏘️ Amenity Score (1–5)", 1, 5, 3)
    with col2:
        gr_liv_area = st.number_input("📐 Living Area (sq ft)", min_value=300, max_value=10000, value=1800)
        social_infra = st.slider("🏥 Social Infrastructure (1–10)", 1, 10, 7)
        drinking_water = st.selectbox("💧 Drinking Water Connection", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        # Prepare the input dictionary
        user_input = {
            "YearBuilt": year_built,
            "GarageCars": garage_cars,
            "GrLivArea": gr_liv_area,
            "AmenityScore": amenity_score,
            "SocialInfra": social_infra,
            "DrinkingWater": 1 if drinking_water == "Yes" else 0
        }

        df = pd.DataFrame([user_input])

        # One-hot encode input if needed
        df_encoded = pd.get_dummies(df)

        # Ensure all training features are present
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Align feature order
        df_encoded = df_encoded[features]

        # Scale and predict
        df_scaled = scaler.transform(df_encoded)
        prediction = model.predict(df_scaled)[0]

        # Display result
        st.success(f"🏷️ Estimated House Price: ₹{int(prediction):,}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

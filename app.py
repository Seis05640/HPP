from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler


model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get raw input
        inputs = {
            "YearBuilt": int(request.form["year_built"]),
            "GarageCars": int(request.form["garage_cars"]),
            "GrLivArea": int(request.form["gr_liv_area"]),
            "AmenityScore": int(request.form["amenity_score"]),
            "SocialInfra": int(request.form["social_infra"]),
            "DrinkingWater": int(request.form["drinking_water"])
        }

        df = pd.DataFrame([inputs])

        # One-hot encode input (even if mostly numerical, include this for future categorical support)
        df_encoded = pd.get_dummies(df)

        # Add missing columns with 0
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Align column order
        df_encoded = df_encoded[features]

        # Scale and predict
        scaled = scaler.transform(df_encoded)
        prediction = model.predict(scaled)
        price = f"{int(prediction[0]):,}"  # Format with commas
        return render_template("result.html", price=price)



    except Exception as e:
        return f"Error: {e}"

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

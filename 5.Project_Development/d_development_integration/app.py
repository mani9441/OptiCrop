import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import os

from opticrop_research_analytics_endpoint import research_analytics


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static" 
    )

CORS(app)

MODEL_PATH = "models/random_forest_crop_recommendation.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# Load once during startup
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

@app.route("/", methods=["GET"])
def home():
    """Renders the single-page OptiCrop dashboard."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json()

        features = pd.DataFrame([{
            "N": data["N"],
            "P": data["P"],
            "K": data["K"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "ph": data["ph"],
            "rainfall": data["rainfall"]
        }])

        prediction = model.predict(features)

        crop = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "success": True,
            "predicted_crop": crop
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400



@app.route("/validate-crop", methods=["POST"])
def validate_crop():
    try:
        data = request.get_json()

        features = pd.DataFrame([{
            "N": data["N"],
            "P": data["P"],
            "K": data["K"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "ph": data["ph"],
            "rainfall": data["rainfall"]
        }])

        planned_crop = data["crop"]

        prediction = model.predict(features)
        recommended_crop = label_encoder.inverse_transform(prediction)[0]

        is_best = planned_crop.strip().lower() == recommended_crop.lower()

        return jsonify({
            "success": True,
            "planned_crop": planned_crop,
            "recommended_crop": recommended_crop,
            "is_best_choice": is_best,
            "message": (
                f"{planned_crop} is the recommended crop for the given soil and weather conditions."
                if is_best
                else f"{planned_crop} is not the optimal crop. We recommend growing {recommended_crop} instead."
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    

@app.route("/research-analytics", methods=["POST"])
def research_analytics_endpoint():
    """
    Endpoint that runs the complete OptiCrop research analytics pipeline.

    Supports:
    1. Uploaded CSV/XLSX
    2. Built-in sample dataset
    """

    # Get dataset name
    dataset_name = request.form.get("dataset_name", None)

    # -----------------------------
    # Check if sample dataset requested
    # -----------------------------
    use_sample = request.form.get("use_sample", "false").lower() == "true"

    if use_sample:
        sample_path = "static/sample_data/Crop_recommendation.csv"

        if not dataset_name:
            dataset_name = "Sample Crop Recommendation Dataset"

        with open(sample_path, "rb") as file:
            return research_analytics(file, dataset_name)

    # -----------------------------
    # Uploaded file
    # -----------------------------
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded. Please provide a CSV/XLSX file or use the sample dataset."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    if not dataset_name:
        dataset_name = os.path.splitext(file.filename)[0]

    return research_analytics(file, dataset_name)



if __name__ == "__main__":
    app.run(debug=True,port=7000)
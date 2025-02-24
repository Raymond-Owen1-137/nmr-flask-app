from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Debugging: Print Working Directory
print("📂 Current Working Directory:", os.getcwd())

# ✅ Ensure Model File Exists
MODEL_PATH = os.path.abspath("residue_model.h5")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ✅ Load Pretrained Model
model_residue = keras.models.load_model(MODEL_PATH)

# ✅ Ensure Training Data Exists
DATA_PATH = os.path.abspath("nmr_training_data.csv")
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: Training data not found at {DATA_PATH}")
    raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

# ✅ Load Training Data for Normalization & Encoding
df = pd.read_csv(DATA_PATH)

# ✅ Keep Only Standard 20 Amino Acids
common_residues = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
df = df[df["Residue_Name"].isin(common_residues)].reset_index(drop=True)

# ✅ Encode Residue Names
label_encoder_residue = LabelEncoder()
df["Residue_Encoded"] = label_encoder_residue.fit_transform(df["Residue_Name"])

# ✅ Standardize Data
features = ["C_Shift", "CA_Shift", "CB_Shift"]
df = df.dropna(subset=features)
scaler = StandardScaler().fit(df[features])

# ✅ Debugging Info
print("✅ Valid Residues:", list(label_encoder_residue.classes_))
print("✅ Flask Server Running...")

# -----------------------------------------------------
# ✅ ROUTES
# -----------------------------------------------------

@app.route("/")
def home():
    """Serve the Frontend (index.html)"""
    try:
        return render_template("index.html")  # Ensure /templates/index.html exists
    except Exception as e:
        return f"⚠️ Error loading index.html: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    """Handle Residue Prediction (Top 5 Residues)"""
    try:
        # ✅ Debugging - Print Incoming Request Data
        print("🔍 Incoming Request Data:", request.json)

        # ✅ Get JSON Data from Frontend
        data = request.json
        if not data:
            raise ValueError("⚠️ No data received!")

        # ✅ Ensure Required Keys Exist
        for key in ["C", "CA", "CB"]:
            if key not in data:
                raise KeyError(f"⚠️ Missing required value: {key}")

        c, ca, cb = float(data["C"]), float(data["CA"]), float(data["CB"])

        # ✅ Debugging - Print Parsed Values
        print(f"🔢 Parsed Values: C={c}, CA={ca}, CB={cb}")

        # ✅ Normalize Input Data
        test_sample = pd.DataFrame([[c, ca, cb]], columns=scaler.feature_names_in_)
        test_sample = scaler.transform(test_sample)

        # ✅ Predict Residue
        prediction_residue = model_residue.predict(test_sample)[0]  # Extract first row

        # ✅ Get Top 5 Residues & Probabilities
        top_5_indices = np.argsort(prediction_residue)[-5:][::-1]  # Sort & get top 5
        top_5_residues = label_encoder_residue.inverse_transform(top_5_indices)
        top_5_probs = prediction_residue[top_5_indices]  # Get corresponding probabilities

        # ✅ Format Response
        response = {
            "C": c,
            "CA": ca,
            "CB": cb,
            "Predictions": [
                {"Residue": res, "Probability": round(float(prob), 4)}
                for res, prob in zip(top_5_residues, top_5_probs)
            ]
        }

        print("✅ API Response:", response)  # Debugging
        return jsonify(response)

    except KeyError as ke:
        return jsonify({"error": str(ke)}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------
# ✅ Run Flask Server (Supports Render Deployment)
# -----------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides a PORT env variable
    app.run(host="0.0.0.0", port=port, debug=True)


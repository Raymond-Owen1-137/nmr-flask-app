from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Load trained model
model = keras.models.load_model("residue_model.h5")

# ✅ Load training data to recreate encoders
file_path = "nmr_training_data.csv"
df = pd.read_csv(file_path)

# ✅ Keep only standard 20 amino acids
common_residues = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
df = df[df["Residue_Name"].isin(common_residues)].reset_index(drop=True)

# ✅ Encode labels & normalize
label_encoder = LabelEncoder()
df["Residue_Encoded"] = label_encoder.fit_transform(df["Residue_Name"])
scaler = StandardScaler().fit(df[["C_Shift", "CA_Shift", "CB_Shift"]])

# ✅ Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        c = float(data["C_Shift"])
        ca = float(data["CA_Shift"])
        cb = float(data["CB_Shift"])

        # Normalize input
        test_sample = pd.DataFrame([[c, ca, cb]], columns=scaler.feature_names_in_)
        test_sample = scaler.transform(test_sample)

        # Make prediction
        prediction = model.predict(test_sample)
        predicted_residue = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        return jsonify({"Predicted_Residue": predicted_residue})

    except Exception as e:
        return jsonify({"error": str(e)})
from waitress import serve
from server import app  # Import Flask app

if __name__ == "__main__":
    print("🚀 Running on http://127.0.0.1:5000")
    serve(app, host="0.0.0.0", port=5000)


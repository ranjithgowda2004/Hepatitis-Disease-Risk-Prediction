# flask_app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

model = joblib.load("artifacts/best_model.pkl")

@app.route("/")
def home():
    return {"message": "Hepatitis Prediction API"}

@app.route("/health")
def health():
    return {"status": "OK", "timestamp": datetime.utcnow().isoformat()}

@app.route("/features")
def features():
    return {"features": list(model.named_steps["preprocessor"].feature_names_in_)}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0]
    pred = int(model.predict(df)[0])
    return jsonify({
        "prediction": "Lives" if pred else "Dies",
        "probability": float(prob[pred]),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict/batch", methods=["POST"])
def batch_predict():
    df = pd.DataFrame(request.json)
    preds = model.predict(df)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(debug=True)

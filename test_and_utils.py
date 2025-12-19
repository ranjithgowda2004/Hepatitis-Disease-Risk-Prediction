# test_and_utils.py

import joblib
import pandas as pd

class ModelTester:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, data: dict):
        df = pd.DataFrame([data])
        return self.model.predict(df)[0]

class BatchPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_csv(self, path):
        df = pd.read_csv(path)
        df["prediction"] = self.model.predict(df)
        df.to_csv("batch_predictions.csv", index=False)
        return df

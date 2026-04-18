import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent

class RiskModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            full_path = BASE_DIR / model_path
            print("Trying to load model from:", full_path)
            self.model = joblib.load(full_path)

    def train(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        return {"accuracy": accuracy_score(y_test, preds)}

    def save(self, path="risk_model.joblib"):
        full_path = BASE_DIR / path
        joblib.dump(self.model, full_path)

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
        return self.model.predict([features])[0]
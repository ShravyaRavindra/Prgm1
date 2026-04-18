import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "training_data.csv"

df = pd.read_csv(csv_path)
print("Columns:", df.columns.tolist())

X = df[["feature1", "feature2"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("n_features_in_:", model.n_features_in_)

joblib.dump(model, BASE_DIR / "risk_model.joblib")
print("Saved model to:", BASE_DIR / "risk_model.joblib")
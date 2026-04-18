import pandas as pd
from model import RiskModel

print("Loading CSV...")
df = pd.read_csv("training_data.csv")
print("Data shape:", df.shape)

rm = RiskModel()
result = rm.train(df, "target")
print("Training accuracy:", result["accuracy"])

rm.save("risk_model.joblib")
print("Saved risk_model.joblib")
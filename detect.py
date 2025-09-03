import json
import joblib
import pandas as pd

model = joblib.load("models/thought_classifier.pkl")
scalar = joblib.load("models/scaler.pkl")

with open("test.json") as f:
    new_data = json.load(f)

df_new = pd.DataFrame(new_data)
x_new_scaled = scalar.transform(df_new)
predictions = model.predict(x_new_scaled)
print(predictions)
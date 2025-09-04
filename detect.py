import json
import joblib
import pandas as pd
import requests
from time import sleep

model = joblib.load("models/thought_classifier.pkl")
scalar = joblib.load("models/scaler.pkl")

url: str = "http://localhost:3000/mindwave/data"

while True:
    response = requests.get(url)
    print(response.text)
    new_data = [json.loads(response.text)["eeg"]]
    df_new = pd.DataFrame(new_data)
    x_new_scaled = scalar.transform(df_new)
    predictions = model.predict(x_new_scaled)
    print(predictions)
    sleep(0.5)
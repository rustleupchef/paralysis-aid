import json
import joblib
import requests
import torch
from time import sleep

model = torch.load("models/modelNN.pth")
scaler = joblib.load("models/scalerNN.pkl")

url: str = "http://localhost:3000/mindwave/data"

while True:
    response = requests.get(url)
    print(response.text)

    new_data = [json.loads(response.text)["eeg"]]
    new_data = [x for x in new_data.values()]

    with torch.no_grad():
        sample = torch.tensor([new_data], dtype=torch.float32)
        sample = torch.tensor(scaler.transform(sample), dtype=torch.float32)
        output = model(sample)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted class: {predicted_class}")
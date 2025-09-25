import json
import requests
from time import sleep
import torch
import numpy as np

model = torch.load("models/thought_classifier.pth", weights_only=False)

url: str = "http://localhost:3000/mindwave/data"

x_mean = np.load("models/X_mean.npy")
x_std = np.load("models/X_std.npy")

while True:
    response = requests.get(url)
    print(response.text)
    chunk = json.load(response.text)["eeg"]

    features = np.array([chunk["delta"], chunk["theta"], chunk["loAlpha"], chunk["hiAlpha"],
                     chunk["loBeta"], chunk["hiBeta"], chunk["loGamma"], chunk["midGamma"]],
                     dtype=np.float32)
    features = (features - x_mean) / x_std
    x_tensor = torch.tensor(features).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    print(predicted_class)
    sleep(0.1)
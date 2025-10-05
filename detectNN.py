import json
import joblib
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import sleep

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = torch.load("models/modelNN.pth", weights_only=False)
scaler = joblib.load("models/scalerNN.pkl")

url: str = "http://localhost:3000/mindwave/data"

with open("Classes/Formatted/ClassKey.json", "r") as f:
    class_key = json.load(f)

while True:
    response = requests.get(url)
    print(response.text)

    new_data = json.loads(response.text)["eeg"]
    new_data = [x for x in new_data.values()]

    with torch.no_grad():
        sample = torch.tensor([new_data], dtype=torch.float32)
        sample = torch.tensor(scaler.transform(sample), dtype=torch.float32)
        output = model(sample)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted class: {predicted_class}")
        print(f"Class name: {class_key[str(predicted_class)]}")
        sleep(1)
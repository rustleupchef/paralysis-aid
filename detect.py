import json
import joblib
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import sleep
import sys

if len(sys.argv) > 1:
    version: bool = sys.argv[1] == "0"

class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1):
        super(ESN, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # Random input weights
        self.W_in = nn.Parameter(torch.randn(reservoir_size, input_size) * 0.1, requires_grad=False)

        # Random recurrent weights (reservoir)
        W = torch.randn(reservoir_size, reservoir_size)
        mask = torch.rand(reservoir_size, reservoir_size) < sparsity
        W[mask] = 0.0
        eigvals = torch.linalg.eigvals(W)
        W /= torch.max(torch.abs(eigvals)) / spectral_radius  # normalize spectral radius
        self.W = nn.Parameter(W, requires_grad=False)

        # Trainable output weights
        self.W_out = nn.Linear(reservoir_size, output_size)
    
    def forward(self, x):
        # x: [seq_len, input_size]
        h = torch.zeros(self.reservoir_size)
        for t in range(x.size(0)):
            u = x[t]
            h = torch.tanh(self.W_in @ u + self.W @ h)
        # Use final reservoir state for classification
        out = self.W_out(h)
        return out

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

with open("Classes/Formatted/Key.json", "r") as f:
    key = json.load(f)
    class_key = key["classes"]
    duration: int = key["duration"]
    divisions: int = key["divisions"]

def grab(rest, divisions) -> list[any]:
    responses = []
    for i in range(divisions):
        responses.append(json.loads(requests.get(url).text)["eeg"])
        sleep(float(rest)/float(divisions))
    return responses

while True:
    response = grab(rest=duration, divisions=divisions + 1)

    if version:
        new_data = response[0]
        new_data = [x for x in new_data.values()]

        with torch.no_grad():
            sample = torch.tensor([new_data], dtype=torch.float32)
            sample = torch.tensor(scaler.transform(sample), dtype=torch.float32)
            output = model(sample)
            predicted_class = torch.argmax(output, dim=1).item()
            print(f"Predicted class: {predicted_class}")
            print(f"Class name: {class_key[str(predicted_class)]}")
            sleep(duration)
    else:
        print(response)
        new_data = [[x for x in data.values()] for data in response]
        print(new_data)
        with torch.no_grad():
            sample = torch.tensor(new_data, dtype=torch.float32)
            output = model(sample)
            predicted_class = torch.argmax(output, dim=1).item()
            print(f"Predicted class: {predicted_class}")
            print(f"Class name: {class_key[str(predicted_class)]}")
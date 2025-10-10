import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler
import sys

version = True
epochs = 50

if len(sys.argv) > 1:
    version = sys.argv[1] == "0"
if len(sys.argv) > 2:
    epochs = int(sys.argv[2])

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SequenceDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample['features'], dtype=torch.float32)  # [seq_len, input_dim]
        y = torch.tensor(sample['label'], dtype=torch.long)
        return x, y

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no softmax here; use CrossEntropyLoss
        return x

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

if not version:
    dataset = SequenceDataset("Classes/Formatted/Formatted.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_size = len(dataset[0][0][0])  # number of features per timestep
    reservoir_size = 200
    output_size = 2  # adjust based on your number of classes

    model = ESN(input_size, reservoir_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.W_out.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x.squeeze(0))
            loss = criterion(output.unsqueeze(0), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
    torch.save(model, "models/modelNN.pth")
    sys.exit()

with open('Classes/Formatted/Formatted.json', 'r') as f:
    data = json.load(f)

x = [obj["features"] for obj in data]
y = [obj["label"] for obj in data]

scaler = StandardScaler()
x = scaler.fit_transform(x)

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TabularDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X.shape[1]
num_classes = len(set(y.tolist()))
model = MLP(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    running_loss = 0.0
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

torch.save(model, "models/modelNN.pth")
joblib.dump(scaler, "models/scalerNN.pkl")
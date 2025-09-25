import torch
import torch.nn as nn
import torch.optim as optim
import json, os
import numpy as np

# --- Load data ---
X, y = [], []
classes = {"Cat": 0, "Dog": 1}
base_path = "Classes"

for label, idx in classes.items():
    file_path = os.path.join(base_path, label, f"{label}.json")
    with open(file_path) as f:
        data = json.load(f)
    for row in data:
        X.append([row[k] for k in row])
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)
np.save("models/X_mean.npy", X.mean(axis=0))
np.save("models/X_std.npy", X.std(axis=0))

# Convert to tensors
X = torch.tensor(X)
y = torch.tensor(y)

# --- Define model ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(classes))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model, "models/thought_classifier.pth")
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler

with open('Classes/Formatted/Formatted.json', 'r') as f:
    data = json.load(f)

x = [obj["features"] for obj in data]
y = [obj["label"] for obj in data]

scaler = StandardScaler()
x = scaler.fit_transform(x)

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TabularDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

input_dim = X.shape[1]
num_classes = len(set(y.tolist()))
model = MLP(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

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

with torch.no_grad():
    sample = torch.tensor([[1573632, 8650752, 2555904, 15269888, 393216, 13238272, 14286848, 2490378]], dtype=torch.float32)
    sample = torch.tensor(scaler.transform(sample), dtype=torch.float32)
    output = model(sample)
    predicted_class = torch.argmax(output, dim=1).item()
    print("Predicted class:", predicted_class)
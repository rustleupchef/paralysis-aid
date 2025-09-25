import os
import json
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sys

ignore = sys.argv[1:]

classes: dict = {}

folders = os.listdir("Classes")
for folder in folders:
    if folder in ignore:
        continue
    path = os.path.join("Classes", folder)
    if os.path.isdir(path):
        with open(os.path.join(path, f"{folder}.json")) as f:
            classes[folder] = json.load(f)

dfs = []
for class_name, data in classes.items():
    df = pd.DataFrame(data)
    df["label"] = class_name
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(df)

x = df.drop(columns=["label"])
y = df["label"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/thought_classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")
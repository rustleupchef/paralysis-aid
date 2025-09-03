import json
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


with open("Classes/Cappybarra/Cappybarra.json") as f:
    cappy = json.load(f)

with open("Classes/Cat/Cat.json") as f:
    cat = json.load(f)

with open("Classes/Dog/Dog.json") as f:
    dog = json.load(f)

df_cappy = pd.DataFrame(cappy)
df_cat = pd.DataFrame(cat)
df_dog = pd.DataFrame(dog)

df_cappy["label"] = "cappybara"
df_cat["label"] = "cat"
df_dog["label"] = "dog"

df = pd.concat([df_cappy, df_cat, df_dog], ignore_index=True)

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
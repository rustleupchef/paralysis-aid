import json
import joblib
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import sleep
from threading import Thread
import sys
import math
import cv2 as cv

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

running = True

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

def eeg_detection():
    while running:
        response = grab(rest=duration, divisions=divisions)

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
                predicted_class = torch.argmax(output, dim=0).item()
                print(f"Predicted class: {predicted_class}")
                print(f"Class name: {class_key[str(predicted_class)]}")

def main():
    global running

    eeg = Thread(target=eeg_detection)
    eeg.start()

    capture = cv.VideoCapture(0)
    eye_cascade = cv.CascadeClassifier("models/haarcascade_eye.xml")
    face_cascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")

    while running:
        ret, frame = capture.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            eye_gray = gray[y:y+h, x:x+w]
            eye_color = frame[y:y+h, x:x+h]

            eyes = eye_cascade.detectMultiScale(eye_gray)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(eye_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255))

                inner_eye_gray = eye_gray[ey:ey + eh, ex:ex + eh]
                inner_eye_color = eye_color[ey:ey + eh, ex:ex + eh]

                _, thresh = cv.threshold(inner_eye_gray, 70, 255, cv.THRESH_BINARY_INV)

                contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key= lambda x: cv.contourArea(x) / ((cv.minEnclosingCircle(x)[1] ** 2) * math.pi), reverse=True)

                if contours:
                    (cx, cy), radius = cv.minEnclosingCircle(contours[0])
                    cv.circle(inner_eye_color, (int(cx), int(cy)), int(radius), (0, 255, 0))

        cv.imshow("Frame", frame)

        if cv.waitKey(1) == ord('q'):
            running = False
    
    capture.release()
    cv.destroyAllWindows()
    eeg.join()

if __name__ == "__main__":
    main()
import json
import joblib
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import sleep
from threading import Thread
import sys
import cv2 as cv
import dlib
import numpy as np
import math

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
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    while running:
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)            
            left_eye = []
            right_eye = []
    
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye.append((x, y))
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)        
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye.append((x, y))
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)
            
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), np.uint8)
            
            cv.fillPoly(mask, [left_eye], 255)
            cv.fillPoly(mask, [right_eye], 255)
            
            eyes = cv.bitwise_and(gray, gray, mask=mask)
            
            for eye in [left_eye, right_eye]:
                x_min = np.min(eye[:, 0])
                x_max = np.max(eye[:, 0])
                y_min = np.min(eye[:, 1])
                y_max = np.max(eye[:, 1])
                
                eye_region = eyes[y_min:y_max, x_min:x_max]
                
                _, threshold_eye = cv.threshold(eye_region, 70, 255, cv.THRESH_BINARY_INV)
                contours, _ = cv.findContours(threshold_eye, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key= lambda x: cv.contourArea(x), reverse=True)

                if contours:
                    x, y, w, h = cv.boundingRect(contours[0])
                    cv.rectangle(frame, (x_min + x, y_min + y), (x_min + x + w, y_min + y + h), (255, 0, 0), 2)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) == ord('q'):
            running = False
    
    capture.release()
    cv.destroyAllWindows()
    eeg.join()

if __name__ == "__main__":
    main()
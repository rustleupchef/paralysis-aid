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

def check_url_connectivity(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        return False


def eeg_detection():

    if not check_url_connectivity(url):
        return

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
                requests.post("http://localhost:3000/mindwave/eeg", json={"eeg_class": class_key[str(predicted_class)]})
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
                requests.post("http://localhost:3000/mindwave/eeg", json={"eeg_class": class_key[str(predicted_class)]})

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
            mouth = []
            MARKS = [0, 3, 6, 9]

            for n in range(48, 60):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                mouth.append((x, y))
                cv.circle(frame, (x, y), 2, (0, 255 if n-48 in MARKS else 0, 0 if n-48 in MARKS else 255), -1)
    
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

            previousSize = frame.shape[0] * frame.shape[1]
            dir = ""
            squint: list[bool] = []
            smirk: list[bool] = []
            open_mouth = False
            
            for eye in [left_eye, right_eye]:
                x_min = np.min(eye[:, 0])
                x_max = np.max(eye[:, 0])
                y_min = np.min(eye[:, 1])
                y_max = np.max(eye[:, 1])

                squint.append(abs(x_max-x_min)/abs(y_max-y_min) > 3.9)
                if abs(x_max-x_min)/abs(y_max-y_min) > 3.9:
                    cv.putText(frame, "SQUINT", (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                eye_region = eyes[y_min:y_max, x_min:x_max]
                
                _, threshold_eye = cv.threshold(eye_region, 70, 255, cv.THRESH_BINARY_INV)
                contours, _ = cv.findContours(threshold_eye, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key= lambda x: cv.contourArea(x), reverse=True)

                if contours:
                    x, y, w, h = cv.boundingRect(contours[0])
                    cv.rectangle(frame, (x_min + x, y_min + y), (x_min + x + w, y_min + y + h), (255, 0, 0), 2)
                    diff_x1 = x_max - (x_min + x + w)
                    diff_x2 = (x_min + x) - x_min
                    diff_y1 = y_max - (y_min + y + h)
                    diff_y2 = (y_min + y) - y_min
                    
                    horiz_pos = "CENTER"
                    if w/abs(x_max - x_min) < 0.8:
                        horiz_pos = "LEFT" if diff_x1 > diff_x2 else "RIGHT"
                    
                    vert_pos = "CENTER"
                    if h/abs(y_max - y_min) < 0.8:
                        vert_pos = "UP" if diff_y1 > diff_y2 else "DOWN"

                    cv.putText(frame, f"{horiz_pos}-{vert_pos}", (x_min + x + w, y_min + y + h), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 1)
                    if previousSize == 0 or w * h < previousSize:
                        previousSize = w * h
                        dir = f"{horiz_pos}-{vert_pos}"

            mid_mouth_x = mouth[MARKS[1]][0]

            min_eye_x = np.min(left_eye[:, 0]) - mid_mouth_x
            max_eye_x = np.max(right_eye[:, 0]) - mid_mouth_x

            min_mouth_x = mouth[MARKS[0]][0] - mid_mouth_x
            max_mouth_x = mouth[MARKS[2]][0] - mid_mouth_x

            smirk.append(min_mouth_x/min_eye_x > 0.67)
            smirk.append(max_mouth_x/max_eye_x > 0.67)
            if min_mouth_x/min_eye_x > 0.67:
                left_mouth_point = mouth[MARKS[0]]
                cv.putText(frame, "Left Smirk", (left_mouth_point[0], left_mouth_point[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if max_mouth_x/max_eye_x > 0.67:
                right_mouth_point = mouth[MARKS[2]]
                cv.putText(frame, "Right Smirk", (right_mouth_point[0], right_mouth_point[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            min_mouth_y = mouth[MARKS[3]][1]
            max_mouth_y = mouth[MARKS[1]][1]

            if abs(max_mouth_y - min_mouth_y)/abs(max_eye_x - min_eye_x) > 0.41:
                open_mouth = True
                cv.putText(frame, "Mouth Open", (mouth[MARKS[1]][0], mouth[MARKS[1]][1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            data = {
                "dir": dir,
                "squint": [str(s) for s in squint],
                "smirk": [str(s) for s in smirk],
                "open_mouth": open_mouth
            }
            response = requests.post("http://localhost:3000/mindwave/cv", json=data)
            print(response.text)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) == ord('q'):
            running = False
    
    capture.release()
    cv.destroyAllWindows()
    eeg.join()

if __name__ == "__main__":
    main()
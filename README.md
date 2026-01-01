# Paralysis Aid
This project utilizes EEG data from the neurosky mindwave, and interprets with echo state networks or multi layer perceptrons, compiling that data with visual face markers, laying it out to server endpoint.

# Usage

## Installing
for the server
```bash
cd server
npm install
```

for python scripts
```bash
pip3 install -r requirements.txt
```

## Server
You need to configure where the server will be reading the mindwave data from.

Here's some commong points
```javascript
// Linux systems
mw.connect('/dev/rfcomm0');
// Window systems
mw.connect('COM3');
```

All useful pages

- /mindwave/data 
    - Contains all the eeg data as a json
- /collect
    - a page specifically for collecting all the data
- /detect
    - a page for more consistent detections mirroring the collection process
- /mindwave/cv
    - outputs all the interpreted facial marking data to the server end json object from web request
- /mindwave/eeg
    - outputs all the interpreted eeg data to the server end json object from web request
- /mindwave/display
    - This is the html output for all the data that updates every .5 seconds
- /mindwave/detection
    - This is the endpoint that compiles all the eeg and cv data into one json, so it can be web requested

## Facial Markers

The only setup you need for this is downloading the pretrained predictors
https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat

Place this file in the models folder.

## Collecting data

To start you need to setup the config
```json
{
  "times" : 10, // number of samples you want
  "divisions" : 5, // how long the sequence for every sample is
  "rest" : 1, // the sleep between drawing of every data packet in a sequence, and for MLP training the rest between each sequence drawn
  "path" : "/home/siddhant/source/repos/paralysis-aid/Classes/", // the output folder
  "className" : "Name", // what word or class do you want this to be recognized as
  "isMerge" : true // this is the second way to run; if true it will merge all existing classes for the training software to work if false it will just start collecting data for a class
}
```

## training data

To train the data you need to decide which version are you training

MLP vs ESN

ESN is for divisions higher than one
MLP is for divisions exactly one

when running the code you write as such
```bash
python3 train.py 0 # this is for training it as an mlp
python3 train.py 1 # this is for training it as an esn
```

## detecting the code

Understanding the parameters
```json
{
    "squint" : 3.9, // the smallets value before the software detects a squint
    "eye_vertical" : 0.8, // minimum deviation from center of eye for vertical
    "eye_horizontal" : 0.8, // minimum deviation from center of eye for horizontal
    "left_smirk" : 0.67, // minimum deviation compared to left eye to be a left smirk
    "right_smirk" : 0.67, // minimum deviation compared to right eye to be a right smirk
    "mouth_open" : 0.41, // minimum size increase from eye width to have an open mouth
    "eye_contour" : [
        70, // lightest color of the pupil
        255 // darkest color of the pupil
    ]
}
```

Running the code
```bash
python3 detect.py 1 # esn detection
python3 detect.py 0 # mlp detection
```

Now you can view the data on the server endpoints

## Running

Terminal 1
```bash
node server.js
```

Terminal 2
```bash
python3 detect.py version
```

Then in your browser look at the endpoint for detections
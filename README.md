# Paralysis Aid
This project utilizes EEG data from the neurosky mindwave, and interprets with echo state networks or multi layer perceptrons, compiling that data with visual face markers, laying it out to server endpoint.

# Usage

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
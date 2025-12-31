const Mindwave = require('mindwave');
const http = require('http');
const fs = require('fs');
const path = require('path');

let latestMindwaveData = {
    eeg: null,
    attention: null,
    meditation: null,
    blink: null
};

let latestDetection = {
    dir: null,
    squint: null,
    smirk: null,
    open_mouth: null,
    eeg_class: null
};

const server = http.createServer((req, res) => {
    if (req.url === '/mindwave/data' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(latestMindwaveData));
    } else if (req.url === '/collect' && req.method === "POST") {
        const config = require("../Collecting/config.json");
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ class_name: config.className }));
    } else if (req.url === '/collect' && req.method === "GET") {
        const filePath = path.join(__dirname, "collect.html");
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-type': 'text/plain' });
                res.end("Error loading content");
                return;
            }
            res.writeHead(200, { 'Content-type': "text/html" });
            res.end(data);
        });
    } else if (req.url === "/detect" && req.method === "POST") {
        const classes = require("../Classes/Formatted/Key.json");
        const classArray = Object.values(classes.classes);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ classes: classArray }));
    } else if (req.url === "/detect" && req.method === "GET") {
        const filePath = path.join(__dirname, "detect.html");
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-type': 'text/plain' });
                res.end("Error loading content");
                return;
            }
            res.writeHead(200, { 'Content-type': "text/html" });
            res.end(data);
        });
    } else if (req.url === "/mindwave/cv" && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                latestDetection = {
                    dir: data.dir,
                    squint: data.squint,
                    smirk: data.smirk,
                    open_mouth: data.open_mouth,
                    eeg_class: latestDetection.eeg_class
                };
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'success' }));
            } catch (error) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'error', message: 'Invalid JSON' }));
            }
        });
    } else if (req.url === '/mindwave/eeg' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                latestDetection.eeg_class = data.eeg_class;
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'success' }));
            } catch (error) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'error', message: 'Invalid JSON' }));
            }
        });
    } else if (req.url === '/mindwave/detection' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(latestDetection));
    } else if (req.url === "/mindwave/display" && req.method === "GET") {
        const filePath = path.join(__dirname, "display.html");
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-type': 'text/plain' });
                res.end("Error loading content");
                return;
            }

            res.writeHead(200, { 'Content-type': "text/html" });
            res.end(data);
        });
    } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: 'Not Found' }));
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});

const mw = new Mindwave();
mw.on('eeg', data => {
    latestMindwaveData.eeg = data;
});
mw.on('attention', data => {
    latestMindwaveData.attention = data;
});
mw.on('meditation', data => {
    latestMindwaveData.meditation = data;
});
mw.on('blink', data => {
    latestMindwaveData.blink = data;
});

mw.connect('/dev/rfcomm0');
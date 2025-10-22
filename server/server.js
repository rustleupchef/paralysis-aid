const Mindwave = require('mindwave');
const http = require('http');

let latestMindwaveData = {
  eeg: null,
  attention: null,
  meditation: null,
  blink: null
};

const server = http.createServer((req, res) => {
  if (req.url === '/mindwave/data' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(latestMindwaveData));
  } else if (req.url === '/collect' && req.method === "GET") {
    const config = require("../Collecting/config.json");
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`<div style="display: flex; position: absolute; left: 0; top: 0; width: 100%; height: 100%; justify-content: center; align-items: center; flex-direction: column;"><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4><h4>${config.className}</h4></div>`);
  } else if (req.url === "/detect" && req.method === "GET") {
    const classes = require("../Classes/Formatted/Key.json");
    const classArray = Object.values(classes.classes);
    let text = `<div style="display: flex; position: absolute; left: 0; top: 0; width: 100%; height: 100%; justify-content: center; align-items: center; flex-direction: column;">`;
    for (let i = 0; i < 10; i++) {
      let innerText = `<div style="display: flex; width: 100%; justify-content: center; align-items: center; flex-direction: row;">`;
      classArray.forEach((e) => {
        innerText += `<h4 style="margin: 0; padding: 4vw; text-align: center;">${e}</h4>`;
      });
      innerText += "</div>";
      text += innerText;
    }
    text += "</div>";
    console.log(text);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(text);
  } else {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Mindwave server is running\n');
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
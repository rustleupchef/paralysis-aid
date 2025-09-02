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
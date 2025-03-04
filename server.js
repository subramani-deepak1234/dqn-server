const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const fs = require('fs');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const port = 3000;

// Create an HTTP server and WebSocket server
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(bodyParser.json());

// Configure file upload
const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('dataset'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }
    
    console.log(`File uploaded: ${req.file.path}`);
    res.send({ message: 'File uploaded successfully.', filePath: req.file.path });
});

app.post('/train', (req, res) => {
    const { urban_dataset, highway_dataset, num_episodes, batch_size } = req.body;
    if (!urban_dataset || !highway_dataset) {
        return res.status(400).json({ error: 'Missing dataset inputs' });
    }
    
    console.log('Training started with:', req.body);
    
    const pythonProcess = spawn('python3', ['train_dqn.py', urban_dataset, highway_dataset, num_episodes, batch_size]);
    
    let outputData = '';

    pythonProcess.stdout.on('data', (data) => {
        const message = data.toString();
        console.log(`Python Output: ${message}`);
        outputData += message;

        // Extract progress percentage from Python output (assumes Python prints progress like "Progress: 50%")
        const progressMatch = message.match(/Progress:\s*(\d+)%/);
        if (progressMatch) {
            const progress = parseInt(progressMatch[1]);
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({ progress }));
                }
            });
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.json({ message: 'Training completed successfully.', output: outputData });
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({ progress: 100 })); // Send final 100% progress
                }
            });
        } else {
            res.status(500).json({ error: 'Training failed.', output: outputData });
        }
    });
});

// Start server
server.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

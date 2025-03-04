const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const fs = require('fs');

const app = express();
const port = 3000;

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
        console.log(`Python Output: ${data}`);
        outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.json({ message: 'Training completed successfully.', output: outputData });
        } else {
            res.status(500).json({ error: 'Training failed.', output: outputData });
        }
    });
});

// Start server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

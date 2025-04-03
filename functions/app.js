const express = require('express');
const serverless = require('serverless-http');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const router = express.Router();

// Serve static files
app.use(express.static('public'));

// Parse JSON bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Main route
router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API route for probability calculation
router.post('/calculate', (req, res) => {
  const { problem_text } = req.body;
  
  // Spawn Python process
  const pythonProcess = spawn('python', ['app.py', problem_text]);
  
  let result = '';
  let error = '';

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    error += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      res.status(500).json({ error: error || 'An error occurred' });
    } else {
      try {
        const jsonResult = JSON.parse(result);
        res.json(jsonResult);
      } catch (e) {
        res.status(500).json({ error: 'Invalid response format' });
      }
    }
  });
});

app.use('/.netlify/functions/app', router);

module.exports.handler = serverless(app); 
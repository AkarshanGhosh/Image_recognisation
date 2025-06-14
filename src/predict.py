import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import os
import sys
import numpy as np
import argparse
import time
import cv2
import threading
from flask import Flask, request, jsonify, render_template_string
import base64
import io
import json
from werkzeug.serving import make_server

# Define the CNN model architecture (must match the training architecture)
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Second block
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third block
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fourth block
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Define the transform for prediction (must match the test transform in training)
pred_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Global variables to track statistics
prediction_stats = {
    'total': 0,
    'correct': 0,
    'incorrect': 0,
    'class_predictions': {},
    'class_correct': {}
}

# Dictionary of available models and their classes
MODEL_CONFIGS = {
    'animals': {
        'model_path': 'models/best_cnn_animals.pth',
        'classes': None  # Will be determined dynamically from test directory
    },
    'gender': {
        'model_path': 'models/best_cnn_gender.pth',
        'classes': ['men', 'women']
    }
}

# Global variables for web server
flask_app = None
web_server = None
current_model = None
current_classes = None
current_device = None
current_model_type = None

# Function to load the model
def load_model(model_type, model_path=None, test_dir=None):
    """Load a trained model from disk"""
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)  # Go up one level to the main project directory
    
    print(f"Project root: {project_root}")
    print(f"Parent directory: {parent_dir}")
    
    # Get model configuration
    if model_type not in MODEL_CONFIGS:
        print(f"❌ Unknown model type: {model_type}")
        print(f"Available model types: {list(MODEL_CONFIGS.keys())}")
        return None, None, None
    
    config = MODEL_CONFIGS[model_type]
    
    # Use provided model path or adjust default path
    if not model_path:
        model_path = os.path.join(project_root, config['model_path'])
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        # Try to find model in the src/models directory instead
        alt_model_path = os.path.join(project_root, "models", os.path.basename(model_path))
        if os.path.exists(alt_model_path):
            print(f"✓ Found model at alternative location: {alt_model_path}")
            model_path = alt_model_path
        else:
            print("❌ Could not find model file in alternative locations")
            return None, None, None
            
    # Determine classes based on model type
    classes = config['classes']
    if classes is None:
        # For models like 'animals' where classes should be determined from the test directory
        try:
            # Look in the parent directory instead
            test_dir = os.path.join(parent_dir, "real_dataset", "test", "animals")
            print(f"Looking for classes in: {test_dir}")
            
            if not os.path.exists(test_dir):
                print(f"❌ Directory not found: {test_dir}")
                # Try alternative path
                test_dir = os.path.join(parent_dir, "real_dataset", "train", "animals")
                print(f"Trying alternative path: {test_dir}")
                
                if not os.path.exists(test_dir):
                    print(f"❌ Alternative directory not found: {test_dir}")
                    # Try current directory structure
                    test_dir = os.path.join(project_root, "dataset", "test", "animals")
                    print(f"Trying current directory structure: {test_dir}")
                    
                    if not os.path.exists(test_dir):
                        # Fallback to predefined classes
                        print("❌ Could not find dataset directory, using fallback classes")
                        classes = ['cat', 'dog', 'bird', 'fish']  # Common animal classes
                    else:
                        classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                        classes.sort()
                else:
                    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                    classes.sort()
            else:
                classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                classes.sort()
            
            print(f"Found classes: {classes}")
        except Exception as e:
            print(f"❌ Error determining classes from directory: {e}")
            # Fallback classes
            classes = ['cat', 'dog', 'bird', 'fish']
            print(f"Using fallback classes: {classes}")
    
    # Ensure we have valid classes
    if not classes:
        print(f"❌ Could not determine classes for model type: {model_type}")
        return None, None, None
    
    print(f"Model type: {model_type}")
    print(f"Classes: {classes}")
    print(f"Using model file: {model_path}")
    
    # Create and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedCNN(len(classes)).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✓ Model loaded successfully")
        
        # Initialize statistics counters for each class
        for cls in classes:
            prediction_stats['class_predictions'][cls] = 0
            prediction_stats['class_correct'][cls] = 0
            
        return model, classes, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

# Function to make a prediction
def predict_image(image, model, classes, device):
    try:
        # If image is a numpy array (from camera), convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            # If it's a file path
            image = Image.open(image).convert('RGB')
        
        # Preprocess the image
        img_tensor = pred_transform(image).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # Get the top prediction and all class probabilities
            confidence_scores = {classes[i]: float(probabilities[i]) * 100 for i in range(len(classes))}
            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            top_pred_class = sorted_scores[0][0]
            
            return image, top_pred_class, confidence_scores
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, None, None

# Web application HTML template with model selection
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classification Web App</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #4285F4, #34A853);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .content {
            padding: 40px;
        }
        
        .model-section {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        
        .model-select {
            margin: 20px 0;
        }
        
        .model-select select {
            padding: 10px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 10px;
            background: white;
            margin-right: 10px;
        }
        
        .load-model-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .load-model-btn:hover {
            transform: translateY(-2px);
        }
        
        .upload-section {
            text-align: center;
            margin-bottom: 30px;
            opacity: 0.5;
            pointer-events: none;
        }
        
        .upload-section.enabled {
            opacity: 1;
            pointer-events: auto;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            margin: 20px 0;
            background: #f9f9f9;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #4285F4;
            background: #f0f7ff;
        }
        
        .upload-area.dragover {
            border-color: #34A853;
            background: #f0fff0;
        }
        
        .upload-btn {
            background: linear-gradient(45deg, #4285F4, #34A853);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .camera-section {
            margin: 30px 0;
            text-align: center;
        }
        
        .camera-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            margin: 10px;
            transition: transform 0.2s;
        }
        
        .camera-btn:hover {
            transform: translateY(-2px);
        }
        
        #cameraVideo {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        
        .result-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
            display: none;
        }
        
        .prediction-result {
            text-align: center;
            margin: 20px 0;
        }
        
        .prediction-result h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .confidence-bars {
            margin: 20px 0;
        }
        
        .confidence-item {
            margin: 10px 0;
            background: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .confidence-bar {
            background: #e0e0e0;
            border-radius: 5px;
            height: 20px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(45deg, #4285F4, #34A853);
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
        
        .error {
            color: #d32f2f;
            text-align: center;
            padding: 20px;
            background: #ffebee;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .success {
            color: #388e3c;
            text-align: center;
            padding: 20px;
            background: #e8f5e8;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        #imagePreview {
            max-width: 400px;
            max-height: 400px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            margin: 20px auto;
            display: block;
        }
        
        .status-message {
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 10px;
            font-weight: bold;
        }
        
        .status-info {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .status-success {
            background: #e8f5e8;
            color: #388e3c;
        }
        
        .status-warning {
            background: #fff3e0;
            color: #f57c00;
        }
        
        @media (max-width: 768px) {
            .content {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            #imagePreview {
                max-width: 100%;
            }
        }
        
        .hidden {
            display: none !important;
        }
        
        .disabled {
            opacity: 0.5;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Image Classification</h1>
            <p>First select a model, then upload an image or use your camera for classification</p>
        </div>
        
        <div class="content">
            <!-- Model Selection Section -->
            <div class="model-section">
                <h3>🤖 Select Classification Model</h3>
                <div class="model-select">
                    <select id="modelSelect">
                        <option value="">-- Choose a Model --</option>
                        <option value="animals">Animal Classification</option>
                        <option value="gender">Gender Classification</option>
                    </select>
                    <button class="load-model-btn" id="loadModelBtn">Load Model</button>
                </div>
                <div id="statusMessage" class="status-message status-info">
                    Please select and load a model to start classification
                </div>
            </div>
            
            <!-- File Upload Section -->
            <div class="upload-section" id="uploadSection">
                <div class="upload-area" id="uploadArea">
                    <h3>📁 Choose an Image</h3>
                    <p>Drag and drop an image here or click to browse</p>
                    <input type="file" id="imageInput" accept="image/*" style="display: none;">
                    <br><br>
                    <button class="upload-btn" onclick="document.getElementById('imageInput').click()">
                        Browse Files
                    </button>
                </div>
            </div>
            
            <!-- Camera Section -->
            <div class="camera-section disabled" id="cameraSection">
                <h3>📷 Or Use Your Camera</h3>
                <button class="camera-btn" id="startCameraBtn">Start Camera</button>
                <button class="camera-btn hidden" id="stopCameraBtn">Stop Camera</button>
                <button class="camera-btn hidden" id="captureBtn">Capture & Predict</button>
                <br>
                <video id="cameraVideo" class="hidden" autoplay playsinline></video>
                <canvas id="captureCanvas" class="hidden"></canvas>
            </div>
            
            <!-- Results Section -->
            <div class="result-section" id="resultSection">
                <div id="loadingDiv" class="loading hidden">
                    🔄 Analyzing image...
                </div>
                
                <div id="errorDiv" class="error hidden"></div>
                
                <div id="resultDiv" class="hidden">
                    <img id="imagePreview" alt="Uploaded image">
                    <div class="prediction-result">
                        <h3 id="predictionTitle">Prediction Result</h3>
                        <div class="confidence-bars" id="confidenceBars"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let stream = null;
        let isCapturing = false;
        let modelLoaded = false;
        
        // Model management
        const modelSelect = document.getElementById('modelSelect');
        const loadModelBtn = document.getElementById('loadModelBtn');
        const statusMessage = document.getElementById('statusMessage');
        const uploadSection = document.getElementById('uploadSection');
        const cameraSection = document.getElementById('cameraSection');
        
        loadModelBtn.addEventListener('click', loadModel);
        
        async function loadModel() {
            const selectedModel = modelSelect.value;
            if (!selectedModel) {
                showStatus('Please select a model first', 'warning');
                return;
            }
            
            showStatus('Loading model...', 'info');
            loadModelBtn.disabled = true;
            loadModelBtn.textContent = 'Loading...';
            
            try {
                const response = await fetch('/load_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ model_type: selectedModel })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    modelLoaded = true;
                    showStatus(`${selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} model loaded successfully! You can now classify images.`, 'success');
                    uploadSection.classList.add('enabled');
                    cameraSection.classList.remove('disabled');
                    loadModelBtn.textContent = 'Model Loaded ✓';
                    loadModelBtn.style.background = '#4CAF50';
                } else {
                    showStatus(`Failed to load model: ${result.error}`, 'warning');
                    loadModelBtn.disabled = false;
                    loadModelBtn.textContent = 'Load Model';
                }
            } catch (error) {
                showStatus(`Error loading model: ${error.message}`, 'warning');
                loadModelBtn.disabled = false;
                loadModelBtn.textContent = 'Load Model';
            }
        }
        
        function showStatus(message, type) {
            statusMessage.textContent = message;
            statusMessage.className = `status-message status-${type}`;
        }
        
        // File upload handling
        const imageInput = document.getElementById('imageInput');
        const uploadArea = document.getElementById('uploadArea');
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (modelLoaded) {
                uploadArea.classList.add('dragover');
            }
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (modelLoaded && e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (modelLoaded && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
        
        // Camera functionality
        const startCameraBtn = document.getElementById('startCameraBtn');
        const stopCameraBtn = document.getElementById('stopCameraBtn');
        const captureBtn = document.getElementById('captureBtn');
        const cameraVideo = document.getElementById('cameraVideo');
        const captureCanvas = document.getElementById('captureCanvas');
        
        startCameraBtn.addEventListener('click', startCamera);
        stopCameraBtn.addEventListener('click', stopCamera);
        captureBtn.addEventListener('click', captureAndPredict);
        
        async function startCamera() {
    if (!modelLoaded) {
        showStatus('Please load a model first', 'warning');
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Camera access not supported in this browser or insecure origin.');
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        cameraVideo.srcObject = stream;

        startCameraBtn.classList.add('hidden');
        stopCameraBtn.classList.remove('hidden');
        captureBtn.classList.remove('hidden');
        cameraVideo.classList.remove('hidden');
    } catch (error) {
        showError('Error accessing camera: ' + error.message);
    }
}

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            startCameraBtn.classList.remove('hidden');
            stopCameraBtn.classList.add('hidden');
            captureBtn.classList.add('hidden');
            cameraVideo.classList.add('hidden');
        }
        
        function captureAndPredict() {
            if (!modelLoaded) {
                showStatus('Please load a model first', 'warning');
                return;
            }
            
            const canvas = captureCanvas;
            const context = canvas.getContext('2d');
            
            canvas.width = cameraVideo.videoWidth;
            canvas.height = cameraVideo.videoHeight;
            
            context.drawImage(cameraVideo, 0, 0);
            
            // Convert canvas to blob and send for prediction
            canvas.toBlob((blob) => {
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');
                uploadImage(formData, true);
            }, 'image/jpeg', 0.8);
        }
        
        function handleFileSelect(file) {
            if (!modelLoaded) {
                showStatus('Please load a model first', 'warning');
                return;
            }
            
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file.');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            uploadImage(formData, false);
        }
        
        async function uploadImage(formData, isFromCamera) {
            showLoading();
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResult(result, isFromCamera);
                } else {
                    showError(result.error || 'Prediction failed');
                }
            } catch (error) {
                showError('Error uploading image: ' + error.message);
            }
        }
        
        function displayResult(result, isFromCamera) {
            hideLoading();
            
            const resultSection = document.getElementById('resultSection');
            const imagePreview = document.getElementById('imagePreview');
            const predictionTitle = document.getElementById('predictionTitle');
            const confidenceBars = document.getElementById('confidenceBars');
            const resultDiv = document.getElementById('resultDiv');
            
            // Show image
            if (!isFromCamera) {
                imagePreview.src = 'data:image/jpeg;base64,' + result.image;
            } else {
                // For camera captures, use the canvas content
                imagePreview.src = captureCanvas.toDataURL('image/jpeg');
            }
            
            // Show prediction
            predictionTitle.textContent = `Prediction: ${result.prediction}`;
            
            // Create confidence bars
            confidenceBars.innerHTML = '';
            const sortedScores = Object.entries(result.confidence_scores)
                .sort(([,a], [,b]) => b - a);
            
            sortedScores.forEach(([className, confidence]) => {
                const item = document.createElement('div');
                item.className = 'confidence-item';
                
                const isTopPrediction = className === result.prediction;
                const fillColor = isTopPrediction ? 
                    'linear-gradient(45deg, #4285F4, #34A853)' : 
                    'linear-gradient(45deg, #ccc, #999)';
                
                item.innerHTML = `
                    <div class="confidence-label">
                        <span>${className}</span>
                        <span>${confidence.toFixed(1)}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%; background: ${fillColor};"></div>
                    </div>
                `;
                
                confidenceBars.appendChild(item);
            });
            
            resultSection.style.display = 'block';
            resultDiv.classList.remove('hidden');
        }
        
        function showLoading() {
            document.getElementById('loadingDiv').classList.remove('hidden');
            document.getElementById('errorDiv').classList.add('hidden');
            document.getElementById('resultDiv').classList.add('hidden');
            document.getElementById('resultSection').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loadingDiv').classList.add('hidden');
        }
        
        function showError(message) {
            hideLoading();
            const errorDiv = document.getElementById('errorDiv');
            errorDiv.textContent = message;
            errorDiv.classList.remove('hidden');
            document.getElementById('resultSection').style.display = 'block';
        }
    </script>
</body>
</html>
"""

# Flask web application
def create_web_app():
    """Create Flask web application"""
    app = Flask(__name__)
    app.secret_key = 'your-secret-key'  # Replace with a secure key in production

    @app.route("/", methods=["GET"])
    def home():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/load_model", methods=["POST"])
    def load_model_route():
        global current_model, current_classes, current_device, current_model_type
        try:
            data = request.get_json()
            model_type = data.get("model_type")
            model, classes, device = load_model(model_type)
            if model is None:
                return jsonify({"success": False, "error": "Failed to load model."})
            current_model = model
            current_classes = classes
            current_device = device
            current_model_type = model_type
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    @app.route("/predict", methods=["POST"])
    def predict_route():
        global current_model, current_classes, current_device
        if current_model is None:
            return jsonify({"success": False, "error": "Model not loaded."})

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided."})

        file = request.files["image"]
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            _, top_prediction, confidence_scores = predict_image(image, current_model, current_classes, current_device)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return jsonify({
                "success": True,
                "prediction": top_prediction,
                "confidence_scores": confidence_scores,
                "image": img_str
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    return app

# Threaded server to run Flask in the background if needed
class ServerThread(threading.Thread):
    def __init__(self, app, port=5000):
        threading.Thread.__init__(self)
        self.srv = make_server("0.0.0.0", port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print("Starting server on http://localhost:5000")
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()
        print("Server stopped.")

# Run the web app directly if this file is executed
if __name__ == "__main__":
    flask_app = create_web_app()
    flask_app.run(debug=True, host="0.0.0.0", port=5000)

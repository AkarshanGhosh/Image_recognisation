# app_generator.py
import os
import zipfile
import tempfile
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CustomAppGenerator:
    def __init__(self, project: Dict[str, Any]):
        self.project = project
        self.project_id = str(project.get('_id', ''))
        self.project_name = project.get('projectName', 'MyApp')
        self.model_type = project.get('modelType', 'image_classification')
        self.classes = project.get('classes', [])
    
    async def generate(self) -> str:
        """Generate a custom app with the trained model"""
        try:
            # Create temporary directory for app files
            temp_dir = tempfile.mkdtemp()
            app_dir = os.path.join(temp_dir, f"{self.project_name}_app")
            os.makedirs(app_dir, exist_ok=True)
            
            # Generate app files
            await self._generate_html_file(app_dir)
            await self._generate_js_file(app_dir)
            await self._generate_css_file(app_dir)
            await self._generate_readme(app_dir)
            
            # Create zip file
            zip_path = os.path.join(temp_dir, f"{self.project_name}_app.zip")
            await self._create_zip(app_dir, zip_path)
            
            logger.info(f"App generated successfully for project {self.project_id}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Failed to generate app for project {self.project_id}: {e}")
            raise
    
    async def _generate_html_file(self, app_dir: str):
        """Generate the main HTML file"""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_name} - AI Prediction App</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.project_name}</h1>
            <p>AI-powered {self.model_type.replace('_', ' ').title()} App</p>
        </header>
        
        <main>
            <div class="upload-section">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-content">
                        <i class="upload-icon">📷</i>
                        <p>Click to upload an image or drag and drop</p>
                        <input type="file" id="imageInput" accept="image/*" hidden>
                    </div>
                </div>
                
                <button id="predictBtn" class="predict-btn" disabled>Analyze Image</button>
            </div>
            
            <div class="results-section" id="resultsSection" style="display: none;">
                <h3>Prediction Results</h3>
                <div id="resultsList" class="results-list"></div>
            </div>
            
            <div class="preview-section" id="previewSection" style="display: none;">
                <h3>Selected Image</h3>
                <img id="imagePreview" alt="Preview" />
            </div>
        </main>
        
        <footer>
            <p>Powered by AI Training Platform</p>
        </footer>
    </div>
    
    <script src="script.js"></script>
</body>
</html>"""
        
        with open(os.path.join(app_dir, "index.html"), "w") as f:
            f.write(html_content)
    
    async def _generate_js_file(self, app_dir: str):
        """Generate the JavaScript file"""
        js_content = f"""// {self.project_name} - AI Prediction App
const PROJECT_ID = '{self.project_id}';
const API_BASE_URL = 'http://localhost:8000'; // Change this to your API URL

class AIApp {{
    constructor() {{
        this.selectedImage = null;
        this.initializeEventListeners();
    }}
    
    initializeEventListeners() {{
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const predictBtn = document.getElementById('predictBtn');
        
        uploadArea.addEventListener('click', () => imageInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        imageInput.addEventListener('change', this.handleImageSelect.bind(this));
        predictBtn.addEventListener('click', this.makePrediction.bind(this));
    }}
    
    handleDragOver(e) {{
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('drag-over');
    }}
    
    handleDrop(e) {{
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {{
            this.processImage(files[0]);
        }}
    }}
    
    handleImageSelect(e) {{
        const file = e.target.files[0];
        if (file) {{
            this.processImage(file);
        }}
    }}
    
    processImage(file) {{
        if (!file.type.startsWith('image/')) {{
            alert('Please select a valid image file.');
            return;
        }}
        
        this.selectedImage = file;
        this.showImagePreview(file);
        document.getElementById('predictBtn').disabled = false;
    }}
    
    showImagePreview(file) {{
        const reader = new FileReader();
        reader.onload = (e) => {{
            const preview = document.getElementById('imagePreview');
            preview.src = e.target.result;
            document.getElementById('previewSection').style.display = 'block';
        }};
        reader.readAsDataURL(file);
    }}
    
    async makePrediction() {{
        if (!this.selectedImage) return;
        
        const predictBtn = document.getElementById('predictBtn');
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyzing...';
        
        try {{
            const base64Image = await this.fileToBase64(this.selectedImage);
            
            const response = await fetch(`${{API_BASE_URL}}/predict`, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer mock_token' // Add proper auth
                }},
                body: JSON.stringify({{
                    projectId: PROJECT_ID,
                    imageData: base64Image,
                    inputType: 'image'
                }})
            }});
            
            if (!response.ok) {{
                throw new Error('Prediction failed');
            }}
            
            const result = await response.json();
            this.displayResults(result.results);
            
        }} catch (error) {{
            console.error('Prediction error:', error);
            alert('Prediction failed. Please try again.');
        }} finally {{
            predictBtn.disabled = false;
            predictBtn.textContent = 'Analyze Image';
        }}
    }}
    
    fileToBase64(file) {{
        return new Promise((resolve, reject) => {{
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        }});
    }}
    
    displayResults(results) {{
        const resultsSection = document.getElementById('resultsSection');
        const resultsList = document.getElementById('resultsList');
        
        resultsList.innerHTML = '';
        
        results.forEach((result, index) => {{
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            resultItem.innerHTML = `
                <div class="result-class">${{result.className}}</div>
                <div class="result-confidence">${{(result.confidence * 100).toFixed(1)}}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${{result.confidence * 100}}%"></div>
                </div>
            `;
            resultsList.appendChild(resultItem);
        }});
        
        resultsSection.style.display = 'block';
    }}
}}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {{
    new AIApp();
}});"""
        
        with open(os.path.join(app_dir, "script.js"), "w") as f:
            f.write(js_content)
    
    async def _generate_css_file(self, app_dir: str):
        """Generate the CSS file"""
        css_content = """/* AI Prediction App Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    color: white;
    margin-bottom: 40px;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
}

header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

main {
    background: white;
    border-radius: 15px;
    padding: 40px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}

.upload-section {
    margin-bottom: 30px;
}

.upload-area {
    border: 3px dashed #667eea;
    border-radius: 10px;
    padding: 60px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-bottom: 20px;
}

.upload-area:hover, .upload-area.drag-over {
    border-color: #5a67d8;
    background-color: #f7fafc;
}

.upload-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}

.upload-icon {
    font-size: 3rem;
}

.upload-content p {
    color: #667eea;
    font-size: 1.1rem;
}

.predict-btn {
    width: 100%;
    padding: 15px;
    font-size: 1.1rem;
    font-weight: bold;
    color: white;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.predict-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
}

.predict-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.results-section, .preview-section {
    margin-top: 30px;
    padding-top: 30px;
    border-top: 2px solid #e2e8f0;
}

.results-section h3, .preview-section h3 {
    margin-bottom: 20px;
    color: #2d3748;
}

.results-list {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.result-item {
    background: #f7fafc;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.result-class {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 5px;
    color: #2d3748;
}

.result-confidence {
    font-size: 1rem;
    color: #667eea;
    margin-bottom: 10px;
}

.confidence-bar {
    width: 100%;
    height: 8px;
    background-color: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    transition: width 0.5s ease;
}

#imagePreview {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

footer {
    text-align: center;
    margin-top: 40px;
    color: white;
    opacity: 0.8;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    main {
        padding: 20px;
    }
    
    header h1 {
        font-size: 2rem;
    }
    
    .upload-area {
        padding: 40px 20px;
    }
}"""
        
        with open(os.path.join(app_dir, "style.css"), "w") as f:
            f.write(css_content)
    
    async def _generate_readme(self, app_dir: str):
        """Generate README file"""
        readme_content = f"""# {self.project_name} - AI Prediction App

This is an automatically generated web application for your trained AI model.

## Features

- **Drag & Drop Image Upload**: Easy image selection
- **Real-time Prediction**: Get instant AI predictions
- **Beautiful UI**: Modern, responsive design
- **Mobile Friendly**: Works on all devices

## Model Information

- **Project ID**: {self.project_id}
- **Model Type**: {self.model_type}
- **Classes**: {', '.join(self.classes)}

## How to Use

1. Open `index.html` in your web browser
2. Upload an image by clicking the upload area or dragging & dropping
3. Click "Analyze Image" to get predictions
4. View the results with confidence scores

## Setup

1. Make sure your AI Training Platform API is running
2. Update the `API_BASE_URL` in `script.js` if needed
3. Add proper authentication tokens if required

## Files

- `index.html` - Main application page
- `script.js` - Application logic and API calls
- `style.css` - Styling and design
- `README.md` - This file

## Support

For support or questions, contact the AI Training Platform team.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(os.path.join(app_dir, "README.md"), "w") as f:
            f.write(readme_content)
    
    async def _create_zip(self, source_dir: str, zip_path: str):
        """Create a zip file from the source directory"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arc_name)
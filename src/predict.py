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
from datetime import datetime

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

# Function to load the model
def load_model(model_type, model_path, test_dir=None):
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
                    return None, None, None
            
            classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            classes.sort()  # Ensure consistent order
            print(f"Found classes: {classes}")
        except Exception as e:
            print(f"❌ Error determining classes from directory: {e}")
            return None, None, None
    
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
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            # Load image from path
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

# Class for the enhanced prediction application GUI
class EnhancedPredictionApp:
    def __init__(self, root):
        self.root = root
        self.models = {}
        self.classes = {}
        self.device = None
        self.current_image_path = None
        self.camera_active = False
        self.cap = None
        self.camera_thread = None
        self.graph_visible = False
        
        # Set window properties
        self.root.title("Multi-Model Image Classification with Live Camera")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.header_label = tk.Label(
            self.main_frame, 
            text="Multi-Model Image Classification", 
            font=("Arial", 24, "bold"),
            bg="#f0f0f0"
        )
        self.header_label.pack(pady=(0, 10))
        
        # Load all models on startup
        self.load_all_models()
        
        # Create control buttons frame
        self.control_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # Add select image button
        self.select_button = tk.Button(
            self.control_frame,
            text="📁 Select Image",
            font=("Arial", 12),
            command=self.select_image,
            bg="#4CAF50",
            fg="white",
            padx=15,
            pady=8
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add camera button
        self.camera_button = tk.Button(
            self.control_frame,
            text="📷 Start Camera",
            font=("Arial", 12),
            command=self.toggle_camera,
            bg="#2196F3",
            fg="white",
            padx=15,
            pady=8
        )
        self.camera_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add graph toggle button
        self.graph_button = tk.Button(
            self.control_frame,
            text="📊 Show Graphs",
            font=("Arial", 12),
            command=self.toggle_graph,
            bg="#FF9800",
            fg="white",
            padx=15,
            pady=8
        )
        self.graph_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add quit button
        self.quit_button = tk.Button(
            self.control_frame,
            text="❌ Quit",
            font=("Arial", 12),
            command=self.quit_application,
            bg="#F44336",
            fg="white",
            padx=15,
            pady=8
        )
        self.quit_button.pack(side=tk.RIGHT)
        
        # Create main content frame
        self.content_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create image display frame
        self.image_frame = tk.Frame(self.content_frame, bg="white", relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image label
        self.image_label = tk.Label(
            self.image_frame,
            text="Select an image or start camera to begin",
            font=("Arial", 14),
            bg="white",
            fg="#666666"
        )
        self.image_label.pack(expand=True)
        
        # Create predictions frame
        self.predictions_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.predictions_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Predictions title
        self.pred_title = tk.Label(
            self.predictions_frame,
            text="Real-time Predictions",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        self.pred_title.pack(pady=(0, 10))
        
        # Create prediction result frames for each model
        self.prediction_frames = {}
        self.prediction_labels = {}
        
        for model_type in MODEL_CONFIGS.keys():
            # Create frame for this model's predictions
            frame = tk.Frame(self.predictions_frame, bg="white", relief=tk.RAISED, bd=2)
            frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Model title
            title_label = tk.Label(
                frame,
                text=f"{model_type.capitalize()} Model",
                font=("Arial", 14, "bold"),
                bg="white"
            )
            title_label.pack(pady=5)
            
            # Prediction label
            pred_label = tk.Label(
                frame,
                text="No prediction yet",
                font=("Arial", 12),
                bg="white",
                fg="#666666",
                wraplength=200
            )
            pred_label.pack(pady=5, padx=10)
            
            self.prediction_frames[model_type] = frame
            self.prediction_labels[model_type] = pred_label
        
        # Create graph frame (initially hidden)
        self.graph_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        
        # Create matplotlib figure for the graphs
        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create feedback frame
        self.feedback_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.feedback_frame.pack(fill=tk.X, pady=10)
        
        # Feedback title
        self.feedback_title = tk.Label(
            self.feedback_frame,
            text="Feedback (for image predictions)",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        )
        self.feedback_title.pack(pady=(0, 5))
        
        # Feedback buttons frame
        self.feedback_buttons_frame = tk.Frame(self.feedback_frame, bg="#f0f0f0")
        self.feedback_buttons_frame.pack()
        
        # Add correct button
        self.correct_button = tk.Button(
            self.feedback_buttons_frame,
            text="✓ Correct",
            font=("Arial", 12),
            command=lambda: self.record_feedback(True),
            bg="#4CAF50",
            fg="white",
            state=tk.DISABLED,
            padx=20,
            pady=8
        )
        self.correct_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add incorrect button
        self.incorrect_button = tk.Button(
            self.feedback_buttons_frame,
            text="✗ Incorrect",
            font=("Arial", 12),
            command=lambda: self.record_feedback(False),
            bg="#F44336",
            fg="white",
            state=tk.DISABLED,
            padx=20,
            pady=8
        )
        self.incorrect_button.pack(side=tk.LEFT)
        
        # Stats label
        self.stats_label = tk.Label(
            self.main_frame,
            text="Ready to start predictions",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        self.stats_label.pack(pady=10)
        
        # Status message
        self.status_label = tk.Label(
            self.main_frame,
            text="Models loaded successfully. Ready for predictions!",
            font=("Arial", 11, "italic"),
            fg="#555555",
            bg="#f0f0f0"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Set up class variables
        self.current_predictions = {}
        self.current_confidences = {}
    
    def load_all_models(self):
        """Load all available models on startup"""
        self.status_label = tk.Label(self.main_frame, text="Loading models...", bg="#f0f0f0")
        self.status_label.pack()
        
        project_root = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(project_root)
        
        loaded_models = 0
        
        for model_type, config in MODEL_CONFIGS.items():
            model_path = os.path.join(project_root, config['model_path'])
            
            # Determine test directory based on directory structure
            test_dir = None
            if model_type == 'animals':
                test_dir = os.path.join(parent_dir, 'real_dataset', 'test', 'animals')
                if not os.path.exists(test_dir):
                    test_dir = os.path.join(parent_dir, 'real_dataset', 'train', 'animals')
            
            model, classes, device = load_model(model_type, model_path, test_dir)
            
            if model:
                self.models[model_type] = model
                self.classes[model_type] = classes
                self.device = device
                loaded_models += 1
                print(f"✓ {model_type} model loaded successfully")
            else:
                print(f"❌ Failed to load {model_type} model")
        
        if loaded_models > 0:
            print(f"✓ Successfully loaded {loaded_models} out of {len(MODEL_CONFIGS)} models")
        else:
            print("❌ No models could be loaded")
    
    def select_image(self):
        """Open a file dialog to select an image"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.current_image_path = filepath
            self.predict_and_display_image(filepath)
    
    def predict_and_display_image(self, image_path):
        """Run prediction on image and display results"""
        try:
            # Load and display image
            pil_image = Image.open(image_path)
            self.display_image(pil_image)
            
            # Run predictions on all models
            self.run_all_predictions(pil_image)
            
            # Enable feedback buttons for static image predictions
            self.correct_button.config(state=tk.NORMAL)
            self.incorrect_button.config(state=tk.DISABLED)
            
            # Update status
            self.status_label.config(text="Image predictions complete. Please provide feedback.")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            self.status_label.config(text="Error processing image")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera feed"""
        try:
            self.cap = cv2.VideoCapture(0)  # Try default camera first
            if not self.cap.isOpened():
                # Try other camera indices
                for i in range(1, 5):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                
                if not self.cap.isOpened():
                    messagebox.showerror("Camera Error", "Could not access camera")
                    return
            
            self.camera_active = True
            self.camera_button.config(text="📷 Stop Camera", bg="#f44336")
            self.select_button.config(state=tk.DISABLED)
            
            # Disable feedback buttons for live camera
            self.correct_button.config(state=tk.DISABLED)
            self.incorrect_button.config(state=tk.DISABLED)
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.status_label.config(text="Camera started. Live predictions active.")
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_button.config(text="📷 Start Camera", bg="#2196F3")
        self.select_button.config(state=tk.NORMAL)
        
        # Clear image display
        self.image_label.config(image="", text="Camera stopped")
        
        self.status_label.config(text="Camera stopped. You can select images or restart camera.")
    
    def camera_loop(self):
        """Main camera loop running in separate thread"""
        while self.camera_active:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Convert frame to PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Display frame
                    self.display_image(pil_image)
                    
                    # Run predictions every few frames to avoid overload
                    if hasattr(self, 'frame_count'):
                        self.frame_count += 1
                    else:
                        self.frame_count = 0
                    
                    if self.frame_count % 10 == 0:  # Predict every 10th frame
                        self.run_all_predictions(rgb_frame)
                    
                    time.sleep(0.03)  # ~30 FPS
                else:
                    break
                    
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
        
        # Cleanup
        if self.cap:
            self.cap.release()
    
    def display_image(self, pil_image):
        """Display image in the GUI"""
        try:
            # Resize image to fit display
            display_size = (400, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def run_all_predictions(self, image):
        """Run predictions on all loaded models"""
        self.current_predictions = {}
        self.current_confidences = {}
        
        for model_type, model in self.models.items():
            try:
                _, prediction, confidence_scores = predict_image(
                    image, model, self.classes[model_type], self.device
                )
                
                if prediction:
                    self.current_predictions[model_type] = prediction
                    self.current_confidences[model_type] = confidence_scores
                    
                    # Update prediction display
                    top_conf = max(confidence_scores.values())
                    pred_text = f"Prediction: {prediction}\nConfidence: {top_conf:.1f}%"
                    
                    # Color code based on confidence
                    if top_conf > 80:
                        color = "#4CAF50"  # Green for high confidence
                    elif top_conf > 60:
                        color = "#FF9800"  # Orange for medium confidence
                    else:
                        color = "#F44336"  # Red for low confidence
                    
                    self.prediction_labels[model_type].config(
                        text=pred_text,
                        fg=color
                    )
                else:
                    self.prediction_labels[model_type].config(
                        text="Prediction failed",
                        fg="#666666"
                    )
                    
            except Exception as e:
                print(f"Error predicting with {model_type} model: {e}")
                self.prediction_labels[model_type].config(
                    text="Error in prediction",
                    fg="#F44336"
                )
        
        # Update graphs if visible
        if self.graph_visible:
            self.update_graphs()
    
    def toggle_graph(self):
        """Toggle graph visibility"""
        if not self.graph_visible:
            self.show_graphs()
        else:
            self.hide_graphs()
    
    def show_graphs(self):
        """Show the prediction graphs"""
        self.graph_visible = True
        self.graph_button.config(text="📊 Hide Graphs")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_graphs()
    
    def hide_graphs(self):
        """Hide the prediction graphs"""
        self.graph_visible = False
        self.graph_button.config(text="📊 Show Graphs")
        self.graph_frame.pack_forget()
    
    def update_graphs(self):
        """Update the prediction graphs"""
        if not self.current_confidences:
            return
        
        try:
            # Clear previous plots
            self.fig.clear()
            
            # Create subplots for each model
            num_models = len(self.current_confidences)
            if num_models == 0:
                return
            
            for i, (model_type, confidence_scores) in enumerate(self.current_confidences.items()):
                ax = self.fig.add_subplot(1, num_models, i + 1)
                
                # Prepare data for bar chart
                sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
                classes = [item[0] for item in sorted_scores]
                scores = [item[1] for item in sorted_scores]
                
                # Create bar chart
                colors = ['#4285F4' if score == max(scores) else '#A0A0A0' for score in scores]
                bars = ax.bar(classes, scores, color=colors)
                
                ax.set_ylabel('Confidence (%)')
                ax.set_title(f'{model_type.capitalize()} Model')
                ax.set_ylim([0, 100])
                
                # Add percentage labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=9)
                
                # Rotate x-axis labels if needed
                if len(classes) > 3:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating graphs: {e}")
    
    def record_feedback(self, is_correct):
        """Record user feedback (only for static image predictions)"""
        if not self.current_predictions or self.camera_active:
            return
        
        global prediction_stats
        
        # For simplicity, we'll record feedback for the first model's prediction
        # In a more complex system, you might want to ask which model's prediction to evaluate
        if self.current_predictions:
            first_model = list(self.current_predictions.keys())[0]
            prediction = self.current_predictions[first_model]
            
            prediction_stats['total'] += 1
            prediction_stats['class_predictions'].setdefault(prediction, 0)
            prediction_stats['class_predictions'][prediction] += 1
            
            if is_correct:
                prediction_stats['correct'] += 1
                prediction_stats['class_correct'].setdefault(prediction, 0)
                prediction_stats['class_correct'][prediction] += 1
                feedback_msg = "✓ Feedback recorded: Prediction was correct!"
            else:
                prediction_stats['incorrect'] += 1
                feedback_msg = "✗ Feedback recorded: Prediction was incorrect."
            
            # Update status and stats
            self.status_label.config(text=feedback_msg)
            self.update_stats_display()
            
            # Disable feedback buttons until next prediction
            self.correct_button.config(state=tk.DISABLED)
            self.incorrect_button.config(state=tk.DISABLED)
    
    def update_stats_display(self):
        """Update the statistics display"""
        global prediction_stats
        
        total = prediction_stats['total']
        correct = prediction_stats['correct']
        incorrect = prediction_stats['incorrect']
        
        if total > 0:
            accuracy = (correct / total) * 100
            stats_text = f"Feedback Stats - Total: {total} | Correct: {correct} | Incorrect: {incorrect} | Accuracy: {accuracy:.2f}%"
        else:
            stats_text = "No feedback recorded yet"
        
        self.stats_label.config(text=stats_text)
    
    def quit_application(self):
        """Exit the application"""
        # Stop camera if active
        if self.camera_active:
            self.stop_camera()
        
        # Show final statistics if any
        global prediction_stats
        
        total = prediction_stats['total']
        
        if total > 0:
            accuracy = (prediction_stats['correct'] / total) * 100
            message = f"Session Statistics:\n\n"
            message += f"Total feedback: {total}\n"
            message += f"Correct: {prediction_stats['correct']} ({(prediction_stats['correct']/total)*100:.2f}%)\n"
            message += f"Incorrect: {prediction_stats['incorrect']} ({(prediction_stats['incorrect']/total)*100:.2f}%)\n\n"
            
            # Add per-class statistics
            if prediction_stats['class_predictions']:
                message += "Class Performance:\n"
                for cls, total_preds in prediction_stats['class_predictions'].items():
                    correct_preds = prediction_stats['class_correct'].get(cls, 0)
                    class_accuracy = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
                    message += f"{cls}: {correct_preds}/{total_preds} correct ({class_accuracy:.2f}%)\n"
            
            messagebox.showinfo("Session Statistics", message)
        
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EnhancedPredictionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")
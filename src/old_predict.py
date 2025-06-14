import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image
import os
import sys
import numpy as np
import argparse
import time

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
def predict_image(image_path, model, classes, device):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
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

# Class for the prediction application GUI
class PredictionApp:
    def __init__(self, root, available_models):
        self.root = root
        self.available_models = available_models
        self.model = None
        self.classes = None
        self.device = None
        self.current_model_type = None
        self.current_image_path = None
        
        # Set window properties
        self.root.title("Image Classification")
        self.root.geometry("1000x850")  # Increased height to accommodate model selector
        self.root.configure(bg="#f0f0f0")
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.header_label = tk.Label(
            self.main_frame, 
            text="Image Classification", 
            font=("Arial", 24, "bold"),
            bg="#f0f0f0"
        )
        self.header_label.pack(pady=(0, 10))
        
        # Create model selection frame
        self.model_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.model_frame.pack(fill=tk.X, pady=10)
        
        # Add model selection label
        tk.Label(
            self.model_frame,
            text="Select Model:",
            font=("Arial", 14),
            bg="#f0f0f0"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Add model selection dropdown
        self.model_var = tk.StringVar(value=list(self.available_models.keys())[0])
        self.model_dropdown = ttk.Combobox(
            self.model_frame,
            textvariable=self.model_var,
            values=list(self.available_models.keys()),
            font=("Arial", 12),
            state="readonly",
            width=15
        )
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add model load button
        self.load_model_button = tk.Button(
            self.model_frame,
            text="Load Model",
            font=("Arial", 12),
            command=self.load_selected_model,
            bg="#4285F4",
            fg="white",
            padx=10,
            pady=5
        )
        self.load_model_button.pack(side=tk.LEFT)
        
        # Create button frame
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, pady=10)
        
        # Add select image button (initially disabled)
        self.select_button = tk.Button(
            self.button_frame,
            text="Select Image",
            font=("Arial", 14),
            command=self.select_image,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2,
            state=tk.DISABLED
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add quit button
        self.quit_button = tk.Button(
            self.button_frame,
            text="Quit",
            font=("Arial", 14),
            command=self.quit_application,
            bg="#F44336",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.quit_button.pack(side=tk.RIGHT)
        
        # Create content frame
        self.content_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a frame for the figure
        self.figure_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.figure_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for the image and predictions
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create feedback frame with a title
        self.feedback_title = tk.Label(
            self.main_frame,
            text="Was the prediction correct?",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        self.feedback_title.pack(pady=(15, 5))
        
        # Create feedback frame
        self.feedback_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.feedback_frame.pack(fill=tk.X, pady=10)
        
        # Add correct button
        self.correct_button = tk.Button(
            self.feedback_frame,
            text="Correct ✓",
            font=("Arial", 14),
            command=lambda: self.record_feedback(True),
            bg="#4CAF50",
            fg="white",
            state=tk.DISABLED,
            padx=25,
            pady=10
        )
        self.correct_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add incorrect button
        self.incorrect_button = tk.Button(
            self.feedback_frame,
            text="Incorrect ✗",
            font=("Arial", 14),
            command=lambda: self.record_feedback(False),
            bg="#F44336",
            fg="white",
            state=tk.DISABLED,
            padx=25,
            pady=10
        )
        self.incorrect_button.pack(side=tk.LEFT)
        
        # Stats label
        self.stats_label = tk.Label(
            self.main_frame,
            text="Total: 0 | Correct: 0 | Incorrect: 0 | Accuracy: 0.00%",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        )
        self.stats_label.pack(pady=10)
        
        # Status message
        self.status_label = tk.Label(
            self.main_frame,
            text="Please load a model to start",
            font=("Arial", 12, "italic"),
            fg="#555555",
            bg="#f0f0f0"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Set up class variables
        self.current_prediction = None
    
    def load_selected_model(self):
        """Load the model selected from the dropdown"""
        model_type = self.model_var.get()
        
        # Update status
        self.status_label.config(text=f"Loading {model_type} model...")
        self.root.update()
        
        # Reset prediction stats for new model
        global prediction_stats
        prediction_stats = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'class_predictions': {},
            'class_correct': {}
        }
        
        # Load model
        project_root = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(project_root)
        
        config = MODEL_CONFIGS[model_type]
        model_path = os.path.join(project_root, config['model_path'])
        
        # Determine test directory based on directory structure
        test_dir = None
        if model_type == 'animals':
            # Try to find the test directory in the parent directory (main project directory)
            test_dir = os.path.join(parent_dir, 'real_dataset', 'test', 'animals')
            if not os.path.exists(test_dir):
                test_dir = os.path.join(parent_dir, 'real_dataset', 'train', 'animals')
        
        self.model, self.classes, self.device = load_model(model_type, model_path, test_dir)
        
        if self.model:
            self.current_model_type = model_type
            self.status_label.config(text=f"{model_type.capitalize()} model loaded successfully. Please select an image.")
            self.select_button.config(state=tk.NORMAL)
            self.stats_label.config(text="Total: 0 | Correct: 0 | Incorrect: 0 | Accuracy: 0.00%")
        else:
            self.status_label.config(text=f"Error loading {model_type} model. Please check the model file or directory structure.")
            self.select_button.config(state=tk.DISABLED)
    
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
            self.predict_and_display(filepath)
    
    def predict_and_display(self, image_path):
        """Run prediction and display results"""
        # Update status
        self.status_label.config(text="Analyzing image...")
        self.root.update()
        
        # Run prediction
        image, prediction, confidence_scores = predict_image(
            image_path, self.model, self.classes, self.device
        )
        
        if image and prediction and confidence_scores:
            # Store current prediction
            self.current_prediction = prediction
            
            # Clear previous figure
            self.fig.clear()
            
            # Create two subplots - one for image, one for bar chart
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)
            
            # Display image
            ax1.imshow(image)
            ax1.set_title(f"Prediction: {prediction}")
            ax1.axis('off')
            
            # Create bar chart of confidence scores
            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            classes = [item[0] for item in sorted_scores]
            scores = [item[1] for item in sorted_scores]
            
            bars = ax2.bar(classes, scores, color=['#4285F4' if cls == prediction else '#A0A0A0' for cls in classes])
            ax2.set_ylabel('Confidence (%)')
            ax2.set_title('Class Predictions')
            ax2.set_ylim([0, 100])
            
            # Add percentage labels above bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
            
            # Rotate x-axis labels for better readability if needed
            if len(classes) > 3:
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Enable feedback buttons
            self.correct_button.config(state=tk.NORMAL)
            self.incorrect_button.config(state=tk.NORMAL)
            
            # Update status
            self.status_label.config(text=f"Prediction complete. Model says: {prediction}")
            
            # Update statistics display
            global prediction_stats
            prediction_stats['total'] += 1
            prediction_stats['class_predictions'][prediction] += 1
            self.update_stats_display()
        else:
            # Handle prediction failure
            self.status_label.config(text="Error analyzing image. Please try another image.")
            messagebox.showerror("Prediction Error", "Could not analyze the selected image.")

    def record_feedback(self, is_correct):
        """Record user feedback on prediction accuracy"""
        if self.current_prediction:
            global prediction_stats
            
            if is_correct:
                prediction_stats['correct'] += 1
                prediction_stats['class_correct'][self.current_prediction] += 1
                feedback_msg = "✓ Feedback recorded: Prediction was correct!"
            else:
                prediction_stats['incorrect'] += 1
                feedback_msg = "✗ Feedback recorded: Prediction was incorrect."
                
                # Optionally ask for correct class if prediction was wrong
                if len(self.classes) > 2:  # Only for multi-class problems
                    correct_class = simpledialog.askstring(
                        "Correct Class",
                        f"What was the correct class?\nOptions: {', '.join(self.classes)}",
                        parent=self.root
                    )
                    
                    if correct_class and correct_class in self.classes:
                        feedback_msg += f" (Correct class: {correct_class})"
            
            # Update status and stats display
            self.status_label.config(text=feedback_msg)
            self.update_stats_display()
            
            # Reset buttons for next prediction
            self.correct_button.config(state=tk.DISABLED)
            self.incorrect_button.config(state=tk.DISABLED)
    
    def update_stats_display(self):
        """Update the statistics display label"""
        global prediction_stats
        
        total = prediction_stats['total']
        correct = prediction_stats['correct']
        incorrect = prediction_stats['incorrect']
        
        if total > 0:
            accuracy = (correct / total) * 100
            stats_text = f"Total: {total} | Correct: {correct} | Incorrect: {incorrect} | Accuracy: {accuracy:.2f}%"
        else:
            stats_text = "Total: 0 | Correct: 0 | Incorrect: 0 | Accuracy: 0.00%"
        
        self.stats_label.config(text=stats_text)

    def quit_application(self):
        """Exit the application and show final statistics"""
        global prediction_stats
        
        # Create final statistics message
        total = prediction_stats['total']
        
        if total > 0:
            accuracy = (prediction_stats['correct'] / total) * 100
            message = f"Session Statistics:\n\n"
            message += f"Total predictions: {total}\n"
            message += f"Correct: {prediction_stats['correct']} ({(prediction_stats['correct']/total)*100:.2f}%)\n"
            message += f"Incorrect: {prediction_stats['incorrect']} ({(prediction_stats['incorrect']/total)*100:.2f}%)\n\n"
            
            # Add per-class statistics
            message += "Class Performance:\n"
            for cls in self.classes:
                predictions = prediction_stats['class_predictions'].get(cls, 0)
                correct = prediction_stats['class_correct'].get(cls, 0)
                
                if predictions > 0:
                    class_accuracy = (correct / predictions) * 100
                    message += f"{cls}: {correct}/{predictions} correct ({class_accuracy:.2f}%)\n"
                else:
                    message += f"{cls}: No predictions\n"
            
            messagebox.showinfo("Session Statistics", message)
        
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root, MODEL_CONFIGS)
    root.mainloop()
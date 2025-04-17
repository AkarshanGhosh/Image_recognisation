import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from torch.nn import functional as F
import argparse
import time
from pathlib import Path

# Import the model definition from train.py
try:
    from predict import ImprovedCNN
except ImportError:
    # Define the CNN model class here as backup in case train.py import fails
    import torch.nn as nn
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

# Define paths
project_root = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(project_root, 'models', 'best_cnn_animals.pth')
test_dir = os.path.join(project_root, 'real_dataset', 'test', 'animals')

# Define the transform for prediction (should match the test transform in training)
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

# Function to load the model
def load_model(model_path):
    # Determine classes from the test directory
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    classes.sort()  # Ensure consistent order
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN(len(classes)).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Initialize statistics counters for each class
        for cls in classes:
            prediction_stats['class_predictions'][cls] = 0
            prediction_stats['class_correct'][cls] = 0
            
        return model, classes, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

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

# Function to display prediction results in a GUI window
class PredictionApp:
    def __init__(self, root, model, classes, device):
        self.root = root
        self.model = model
        self.classes = classes
        self.device = device
        self.current_image_path = None
        
        # Set window properties
        self.root.title("Animal Image Classifier")
        self.root.geometry("1000x800")  # Increased height for better visibility
        self.root.configure(bg="#f0f0f0")
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.header_label = tk.Label(
            self.main_frame, 
            text="Animal Image Classifier", 
            font=("Arial", 24, "bold"),  # Increased font size
            bg="#f0f0f0"
        )
        self.header_label.pack(pady=(0, 20))
        
        # Create button frame
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, pady=10)
        
        # Add select image button
        self.select_button = tk.Button(
            self.button_frame,
            text="Select Image",
            font=("Arial", 14),  # Increased font size
            command=self.select_image,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add quit button
        self.quit_button = tk.Button(
            self.button_frame,
            text="Quit",
            font=("Arial", 14),  # Increased font size
            command=self.quit_application,  # Changed to a new method
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
            font=("Arial", 14),  # Increased font size
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
            font=("Arial", 14),  # Increased font size
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
            font=("Arial", 14, "bold"),  # Increased font size and made bold
            bg="#f0f0f0"
        )
        self.stats_label.pack(pady=10)
        
        # Status message
        self.status_label = tk.Label(
            self.main_frame,
            text="Please select an image to start",
            font=("Arial", 12, "italic"),  # Increased font size
            fg="#555555",
            bg="#f0f0f0"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Set up class variables
        self.current_prediction = None
        
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
            
            # Clear previous figure content
            self.fig.clear()
            
            # Set up the figure with two subplots
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)
            
            # Display the image
            ax1.imshow(image)
            ax1.set_title("Selected Image")
            ax1.axis('off')
            
            # Sort scores and create bar chart
            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            cls_names = [cls for cls, _ in sorted_scores]
            cls_scores = [score for _, score in sorted_scores]
            
            bars = ax2.barh(range(len(cls_names)), cls_scores, align='center')
            ax2.set_yticks(range(len(cls_names)))
            ax2.set_yticklabels(cls_names)
            ax2.set_xlabel('Confidence (%)')
            ax2.set_title('Prediction Confidence')
            
            # Color the highest confidence bar differently
            bars[0].set_color('green')
            
            # Add percentage annotations to the bars
            for i, (score, bar) in enumerate(zip(cls_scores, bars)):
                if score > 5:  # Only show percentage for bars with significant values
                    ax2.text(
                        bar.get_width() + 0.5, 
                        bar.get_y() + bar.get_height()/2, 
                        f"{score:.1f}%", 
                        ha='left', 
                        va='center'
                    )
            
            # Update the figure
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Show prediction results
            filename = os.path.basename(image_path)
            self.status_label.config(
                text=f"Prediction: {prediction} (from '{filename}')"
            )
            
            # Make feedback title visible
            self.feedback_title.config(text=f"Was the prediction '{prediction}' correct?")
            
            # Enable feedback buttons
            self.correct_button.config(state=tk.NORMAL)
            self.incorrect_button.config(state=tk.NORMAL)
        else:
            # Handle prediction error
            self.status_label.config(text="Error analyzing image. Please try another.")
            messagebox.showerror("Prediction Error", "Could not analyze the selected image.")
    
    def record_feedback(self, is_correct):
        """Record user feedback on prediction accuracy"""
        if self.current_prediction:
            # Update stats
            prediction_stats['total'] += 1
            prediction_stats['class_predictions'][self.current_prediction] += 1
            
            if is_correct:
                prediction_stats['correct'] += 1
                prediction_stats['class_correct'][self.current_prediction] += 1
                feedback_msg = "Thanks! Prediction marked as correct."
            else:
                prediction_stats['incorrect'] += 1
                feedback_msg = "Thanks! Prediction marked as incorrect."
                
                # Ask for the correct class if user marked it as incorrect
                correct_class = simpledialog.askstring(
                    "Correct Class",
                    f"What is the correct class for this image?",
                    parent=self.root
                )
                
                if correct_class and correct_class in self.classes:
                    # Update stats for the correct class if it exists
                    prediction_stats['class_predictions'][correct_class] = \
                        prediction_stats['class_predictions'].get(correct_class, 0) + 1
                    prediction_stats['class_correct'][correct_class] = \
                        prediction_stats['class_correct'].get(correct_class, 0) + 1
                    
                    feedback_msg += f" You indicated the correct class is '{correct_class}'."
            
            # Calculate accuracy
            accuracy = 0
            if prediction_stats['total'] > 0:
                accuracy = (prediction_stats['correct'] / prediction_stats['total']) * 100
            
            # Update stats display
            self.stats_label.config(
                text=f"Total: {prediction_stats['total']} | "
                     f"Correct: {prediction_stats['correct']} | "
                     f"Incorrect: {prediction_stats['incorrect']} | "
                     f"Accuracy: {accuracy:.2f}%"
            )
            
            # Update status
            self.status_label.config(text=feedback_msg)
            
            # Ask if the user wants to continue
            self.ask_continue()
    
    def ask_continue(self):
        """Ask if the user wants to continue with another image"""
        # Disable feedback buttons
        self.correct_button.config(state=tk.DISABLED)
        self.incorrect_button.config(state=tk.DISABLED)
        
        # Show dialog
        result = messagebox.askyesno(
            "Continue?", 
            "Do you want to continue with another image?",
            icon=messagebox.QUESTION
        )
        
        if result:
            # Reset for next prediction
            self.status_label.config(text="Please select another image")
            self.current_prediction = None
        else:
            # Show final stats and exit
            self.show_final_stats()
    
    def show_final_stats(self):
        """Show final statistics and exit"""
        try:
            # Calculate final statistics
            total = prediction_stats['total']
            correct = prediction_stats['correct']
            incorrect = prediction_stats['incorrect']
            
            if total > 0:
                accuracy = (correct / total) * 100
            else:
                accuracy = 0
            
            # Prepare class statistics
            class_stats = []
            for cls in self.classes:
                preds = prediction_stats['class_predictions'][cls]
                correct_preds = prediction_stats['class_correct'][cls]
                
                if preds > 0:
                    class_acc = (correct_preds / preds) * 100
                else:
                    class_acc = 0
                    
                class_stats.append({
                    'class': cls,
                    'predictions': preds,
                    'correct': correct_preds,
                    'accuracy': class_acc
                })
            
            # Sort by number of predictions
            class_stats.sort(key=lambda x: x['predictions'], reverse=True)
            
            # Create statistics window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Prediction Statistics")
            stats_window.geometry("500x600")
            stats_window.configure(bg="#f0f0f0")
            
            # Add header
            tk.Label(
                stats_window,
                text="Final Prediction Statistics",
                font=("Arial", 18, "bold"),
                bg="#f0f0f0"
            ).pack(pady=(20, 10))
            
            # Add overall stats
            overall_frame = tk.Frame(stats_window, bg="#f0f0f0", relief=tk.RIDGE, bd=2)
            overall_frame.pack(fill=tk.X, padx=20, pady=10)
            
            tk.Label(
                overall_frame,
                text=f"Total Predictions: {total}",
                font=("Arial", 14),
                bg="#f0f0f0"
            ).pack(anchor=tk.W, padx=10, pady=(10, 5))
            
            tk.Label(
                overall_frame,
                text=f"Correct Predictions: {correct}",
                font=("Arial", 14),
                bg="#f0f0f0"
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            tk.Label(
                overall_frame,
                text=f"Incorrect Predictions: {incorrect}",
                font=("Arial", 14),
                bg="#f0f0f0"
            ).pack(anchor=tk.W, padx=10, pady=5)
            
            tk.Label(
                overall_frame,
                text=f"Overall Accuracy: {accuracy:.2f}%",
                font=("Arial", 14, "bold"),
                bg="#f0f0f0"
            ).pack(anchor=tk.W, padx=10, pady=(5, 10))
            
            # Add class stats
            class_label = tk.Label(
                stats_window,
                text="Class-wise Statistics:",
                font=("Arial", 16, "bold"),
                bg="#f0f0f0"
            )
            class_label.pack(anchor=tk.W, padx=20, pady=(15, 5))
            
            # Frame for class stats
            class_frame = tk.Frame(stats_window, bg="#f0f0f0")
            class_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
            
            # Create a canvas with scrollbar for class stats
            canvas = tk.Canvas(class_frame, bg="#f0f0f0", highlightthickness=0)
            scrollbar = tk.Scrollbar(class_frame, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
            
            # Configure the canvas
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Pack canvas and scrollbar
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add headers for class stats
            header_frame = tk.Frame(scrollable_frame, bg="#e0e0e0")
            header_frame.pack(fill=tk.X, pady=(0, 5))
            
            tk.Label(header_frame, text="Class", width=10, font=("Arial", 12, "bold"), bg="#e0e0e0").pack(side=tk.LEFT, padx=5, pady=5)
            tk.Label(header_frame, text="Predictions", width=10, font=("Arial", 12, "bold"), bg="#e0e0e0").pack(side=tk.LEFT, padx=5, pady=5)
            tk.Label(header_frame, text="Correct", width=10, font=("Arial", 12, "bold"), bg="#e0e0e0").pack(side=tk.LEFT, padx=5, pady=5)
            tk.Label(header_frame, text="Accuracy", width=10, font=("Arial", 12, "bold"), bg="#e0e0e0").pack(side=tk.LEFT, padx=5, pady=5)
            
            # Add class stats rows
            for i, stat in enumerate(class_stats):
                row_bg = "#f8f8f8" if i % 2 == 0 else "#f0f0f0"
                row_frame = tk.Frame(scrollable_frame, bg=row_bg)
                row_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(row_frame, text=stat['class'], width=10, anchor=tk.W, bg=row_bg).pack(side=tk.LEFT, padx=5, pady=5)
                tk.Label(row_frame, text=str(stat['predictions']), width=10, bg=row_bg).pack(side=tk.LEFT, padx=5, pady=5)
                tk.Label(row_frame, text=str(stat['correct']), width=10, bg=row_bg).pack(side=tk.LEFT, padx=5, pady=5)
                tk.Label(row_frame, text=f"{stat['accuracy']:.2f}%", width=10, bg=row_bg).pack(side=tk.LEFT, padx=5, pady=5)
            
            # Add close button
            tk.Button(
                stats_window,
                text="Close",
                font=("Arial", 14),
                command=lambda: self.finalize_and_close(stats_window),
                bg="#4CAF50",
                fg="white",
                padx=20,
                pady=8
            ).pack(pady=20)
            
            # Wait for this window to close before destroying main window
            self.root.wait_window(stats_window)
            
            # Make sure to destroy the root window
            if self.root.winfo_exists():
                self.root.destroy()
                
        except Exception as e:
            print(f"Error in show_final_stats: {e}")
            # If we encounter an error, make sure to destroy the root window
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.destroy()
                
    def finalize_and_close(self, stats_window):
        """Safely close the stats window and then the main window"""
        try:
            stats_window.destroy()
            if self.root.winfo_exists():
                self.root.destroy()
        except Exception as e:
            print(f"Error in finalize_and_close: {e}")
            # If we encounter an error, make sure to destroy the root window
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.destroy()
                
    def quit_application(self):
        """Safely quit the application"""
        if prediction_stats['total'] > 0:
            # Show final stats before quitting
            self.show_final_stats()
        else:
            # If no predictions were made, just quit
            self.root.destroy()

# Main function
def main():
    # Load the model
    print("🔄 Loading model...")
    model, classes, device = load_model(model_path)
    print(f"✅ Model loaded successfully with {len(classes)} classes: {classes}")
    
    # Create and start the GUI
    root = tk.Tk()
    app = PredictionApp(root, model, classes, device)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"Error in mainloop: {e}")
    
    # Print final stats to console before exiting
    print("\n📊 Final Prediction Statistics:")
    print(f"Total predictions: {prediction_stats['total']}")
    print(f"Correct predictions: {prediction_stats['correct']}")
    print(f"Incorrect predictions: {prediction_stats['incorrect']}")
    
    if prediction_stats['total'] > 0:
        accuracy = (prediction_stats['correct'] / prediction_stats['total']) * 100
        print(f"Overall accuracy: {accuracy:.2f}%")
    
    print("\nClass-wise statistics:")
    for cls in classes:
        preds = prediction_stats['class_predictions'][cls]
        if preds > 0:
            class_acc = (prediction_stats['class_correct'][cls] / preds) * 100
            print(f"- {cls}: {preds} predictions, {prediction_stats['class_correct'][cls]} correct ({class_acc:.2f}%)")
        else:
            print(f"- {cls}: 0 predictions")

if __name__ == "__main__":
    main()
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model_utils import ImprovedCNN

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.pred_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Updated model configurations to match your actual file structure
        self.MODEL_CONFIGS = {
            'animals': {
                'model_path': 'models/best_cnn_animals.pth',  # Will be adjusted based on actual location
                'classes': None  # To be populated dynamically
            },
            'gender': {
                'model_path': 'models/best_cnn_gender.pth',   # Will be adjusted based on actual location
                'classes': ['men', 'women']
            }
        }
        
        # Load models on initialization
        self.load_all_models()
    
    def find_model_files(self):
        """Find the actual model files in your project structure"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up to webapp
        main_project_root = os.path.dirname(project_root)  # Go up to Image_recognisation
        
        # Look for model files in different possible locations
        possible_locations = [
            os.path.join(current_dir, 'models'),                    # webapp/backend/models/
            os.path.join(project_root, 'models'),                   # webapp/models/
            os.path.join(main_project_root, 'models'),              # Image_recognisation/models/
            os.path.join(main_project_root, 'src', 'models'),       # Image_recognisation/src/models/
            current_dir,                                             # webapp/backend/
            project_root,                                            # webapp/
            main_project_root                                        # Image_recognisation/
        ]
        
        found_models = {}
        
        for location in possible_locations:
            if os.path.exists(location):
                files = os.listdir(location)
                for file in files:
                    if file.endswith('.pth'):
                        if 'animal' in file.lower():
                            found_models['animals'] = os.path.join(location, file)
                        elif 'gender' in file.lower():
                            found_models['gender'] = os.path.join(location, file)
        
        return found_models
    
    def get_classes_from_dataset(self, model_name):
        """Get class names from dataset directory structure"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # webapp
        main_project_root = os.path.dirname(project_root)  # Image_recognisation
        
        # Look for dataset directories
        possible_dataset_locations = [
            os.path.join(main_project_root, 'real_dataset'),
            os.path.join(main_project_root, 'dataset'),
            os.path.join(main_project_root, 'data'),
            os.path.join(project_root, 'real_dataset'),
            os.path.join(project_root, 'dataset'),
            os.path.join(project_root, 'data')
        ]
        
        for dataset_root in possible_dataset_locations:
            if os.path.exists(dataset_root):
                # Try test first, then train
                for split in ['test', 'train']:
                    test_dir = os.path.join(dataset_root, split, model_name)
                    if os.path.exists(test_dir):
                        classes = sorted([d for d in os.listdir(test_dir) 
                                        if os.path.isdir(os.path.join(test_dir, d))])
                        if classes:
                            return classes
        
        # Fallback defaults
        if model_name == 'animals':
            return ['cat', 'dog', 'bird', 'fish']  # Common animal classes
        elif model_name == 'gender':
            return ['men', 'women']
        
        return []
    
    def load_all_models(self):
        """Loads all available models into memory."""
        print("🔄 Loading models...")
        
        # Find actual model files
        found_models = self.find_model_files()
        
        if not found_models:
            print("❌ No model files found! Please check your model file locations.")
            return
        
        print(f"📁 Found model files: {found_models}")
        
        for model_name, model_path in found_models.items():
            try:
                # Get classes for this model
                if model_name == 'animals':
                    classes = self.get_classes_from_dataset('animals')
                elif model_name == 'gender':
                    classes = ['men', 'women']
                else:
                    classes = self.get_classes_from_dataset(model_name)
                
                if not classes:
                    print(f"⚠️  Could not determine classes for {model_name}")
                    continue
                
                # Load model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ImprovedCNN(num_classes=len(classes)).to(device)
                
                # Load state dict
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                
                # Store loaded model
                self.loaded_models[model_name] = {
                    'model': model,
                    'classes': classes,
                    'device': device,
                    'path': model_path
                }
                
                print(f"✅ Loaded model: {model_name} ({len(classes)} classes)")
                print(f"   📍 Path: {model_path}")
                print(f"   🏷️  Classes: {classes}")
                
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def predict_all(self, image_input):
        """Runs prediction using all loaded models."""
        if not self.loaded_models:
            return {"error": "No models loaded"}
        
        try:
            # Handle different input types
            if hasattr(image_input, 'read'):
                # File-like object
                image_input.seek(0)
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, bytes):
                # Bytes
                import io
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            else:
                # PIL Image or path
                if isinstance(image_input, str):
                    image = Image.open(image_input).convert('RGB')
                else:
                    image = image_input.convert('RGB')
            
            # Apply transforms
            img_tensor = self.pred_transform(image).unsqueeze(0)
            
        except Exception as e:
            return {"error": f"Invalid image input: {e}"}
        
        results = {}
        for model_name, meta in self.loaded_models.items():
            try:
                model = meta['model']
                classes = meta['classes']
                device = meta['device']
                
                with torch.no_grad():
                    output = model(img_tensor.to(device))
                    probabilities = F.softmax(output, dim=1)[0]
                    top_index = torch.argmax(probabilities).item()
                    
                    results[model_name] = {
                        "prediction": classes[top_index],
                        "confidence": round(probabilities[top_index].item() * 100, 2),
                        "all_probabilities": {
                            classes[i]: round(probabilities[i].item() * 100, 2) 
                            for i in range(len(classes))
                        }
                    }
            except Exception as e:
                results[model_name] = {"error": f"Prediction failed: {e}"}
        
        return results
    
    def get_available_models(self):
        """Returns list of available model names"""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        if model_name not in self.loaded_models:
            return None
        
        meta = self.loaded_models[model_name]
        return {
            "name": model_name,
            "classes": meta['classes'],
            "num_classes": len(meta['classes']),
            "device": str(meta['device']),
            "path": meta['path']
        }

# Create the model manager instance that main.py expects
model_manager = ModelManager()
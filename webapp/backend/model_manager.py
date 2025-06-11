# model_manager.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
from database import db
from models.improved_cnn import ImprovedCNN

class ModelManager:
    def __init__(self):
        self.system_models = {}
        self.user_models = {}
        self.load_system_models()

    def load_system_models(self):
        try:
            animals_model = ImprovedCNN(num_classes=10)
            animals_model.load_state_dict(torch.load('models/best_cnn_animals.pth', map_location='cpu'))
            animals_model.eval()

            self.system_models['animals'] = {
                'model': animals_model,
                'classes': ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
                'transform': transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            }

            gender_model = ImprovedCNN(num_classes=2)
            gender_model.load_state_dict(torch.load('models/best_cnn_gender.pth', map_location='cpu'))
            gender_model.eval()

            self.system_models['gender'] = {
                'model': gender_model,
                'classes': ['men', 'women'],
                'transform': transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            }

            print("✅ System models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading system models: {e}")

    async def load_user_model(self, project_id: str):
        if project_id in self.user_models:
            return self.user_models[project_id]

        try:
            project = await db.projects.find_one({"_id": project_id})
            if not project or project['status'] != 'completed':
                return None

            model = ImprovedCNN(num_classes=len(project['classes']))
            model.load_state_dict(torch.load(project['training']['modelPath'], map_location='cpu'))
            model.eval()

            self.user_models[project_id] = {
                'model': model,
                'classes': project['classes'],
                'transform': transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            }
            return self.user_models[project_id]
        except Exception as e:
            print(f"❌ Error loading user model {project_id}: {e}")
            return None

    async def predict(self, image_data: str, project_id: str = None):
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            results = []

            if project_id:
                model_info = await self.load_user_model(project_id)
                if model_info:
                    results.extend(await self._run_single_prediction(image, model_info))
            else:
                for model_name, model_info in self.system_models.items():
                    predictions = await self._run_single_prediction(image, model_info)
                    for pred in predictions:
                        pred['modelType'] = model_name
                    results.extend(predictions)

            return results
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return []

    async def _run_single_prediction(self, image: Image.Image, model_info: dict):
        try:
            img_tensor = model_info['transform'](image).unsqueeze(0)

            with torch.no_grad():
                outputs = model_info['model'](img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            results = []
            for i, class_name in enumerate(model_info['classes']):
                confidence = float(probabilities[i]) * 100
                if confidence > 10:
                    results.append({
                        'className': class_name,
                        'confidence': round(confidence, 2)
                    })

            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results
        except Exception as e:
            print(f"❌ Single prediction error: {e}")
            return []

model_manager = ModelManager()
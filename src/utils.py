import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

# Define the ImprovedCNN model architecture
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)

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

# Preprocessing: same as in training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load image and apply transform
image_path = "C:\\Users\\user\\Desktop\\mini project\\Image_recognisation\\real_dataset\\test\\animals\\cat\\anete-lusina-609858-unsplash.jpg"

img = Image.open(image_path).convert('RGB')
image_tensor = transform(img)  # shape: [3, 64, 64]

# Load model and weights
model_path = "C:\\Users\\user\\Desktop\\mini project\\Image_recognisation\\models\\final_cnn_animals.pth"
num_classes = 9  # Replace with actual number of classes
model = ImprovedCNN(num_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Visualize feature maps from a given layer
def visualize_feature_maps(model, image, layer_name):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    # Attach hook to specified layer
    layer = getattr(model, layer_name)
    layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(image.unsqueeze(0))  # [1, 3, 64, 64]

    # Get feature map
    feature_maps = activations[0][0].cpu().numpy()  # shape: [C, H, W]
    print(f"✅ Feature map shape from {layer_name}: {feature_maps.shape}")

    # Plot the first 16 feature maps
    plt.figure(figsize=(15, 10))
    for i in range(min(16, feature_maps.shape[0])):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.axis('off')

    plt.suptitle(f'Feature Maps from Layer: {layer_name}', fontsize=16)
    plt.savefig(f'feature_maps_{layer_name}.png')
    plt.show()

# Example usage (you can try 'conv1', 'conv2', 'conv3', 'conv4')
visualize_feature_maps(model, image_tensor, 'conv4')

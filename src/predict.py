import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
from tkinter import Tk, filedialog

# --- CNN Model (Improved Architecture) ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_cifar10.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- CIFAR-10 Classes ---
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# --- Image Picker & Prediction ---
def select_and_predict_image():
    Tk().withdraw()  # Hide tkinter root window
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not image_path:
        print("❌ No image selected.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f"\n🖼️ Prediction: {classes[predicted.item()]}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")

# --- Run ---
if __name__ == "__main__":
    select_and_predict_image()
    print("✅ Prediction complete.")

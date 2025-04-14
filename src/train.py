import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#1 define trasnformations

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    
])

#Load CIFAR-10 datasets

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# CNN-model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128,10)
        

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # ReLU activation function
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # flatten 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)    
        return x

# Instantiate the model, define the loss function and the optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model

for epoch in range (5):
    running_loss=0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels =data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99: #print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
            
            
print("Finished Training")

# Get the path to the project root and create 'models/' if it doesn't exist
project_root = os.path.dirname(os.path.dirname(__file__))  # One level up from 'src'
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

# Save the model
model_path = os.path.join(models_dir, 'cnn_cifar10.pth')
torch.save(model.state_dict(), model_path)

print(f"✅ Model saved at {model_path}")
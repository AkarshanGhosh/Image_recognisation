import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Define the CNN Model for Gender Detection
class GenderCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderCNN, self).__init__()
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

# Define transformations for training and testing
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomCrop(64, padding=8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to visualize a few sample images
def visualize_samples(dataset, classes, n=5):
    fig, axes = plt.subplots(len(classes), n, figsize=(15, 5*len(classes)))
    for i, c in enumerate(classes):
        idx = dataset.class_to_idx[c]
        class_samples = [j for j, (_, label) in enumerate(dataset.samples) if label == idx]
        
        for j in range(min(n, len(class_samples))):
            if j < len(class_samples):
                img, _ = dataset[class_samples[j]]
                img = img.numpy().transpose((1, 2, 0))
                img = img * 0.5 + 0.5  # Denormalize
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{c}")
                axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'gender_samples.png'))
    plt.close()

# Function to train the model with validation
def train_model(model, criterion, optimizer, trainloader, testloader, scheduler, epochs=30, early_stop_patience=5):
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(testloader)
        val_losses.append(epoch_val_loss)
        
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        time_elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} | Time: {time_elapsed:.1f}s | Train Loss: {epoch_loss:.3f} | Val Loss: {epoch_val_loss:.3f} | Accuracy: {accuracy:.2f}%')
        
        # Learning rate scheduler step
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_cnn_gender.pth'))
            print(f"✓ New best model saved (Accuracy: {best_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
                break
    
    # Final model save
    torch.save(model.state_dict(), os.path.join(models_dir, 'final_cnn_gender.pth'))
    print(f"✓ Final model saved")
    print(f"Best accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
    
    # Plot the training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'gender_training_history.png'))
    plt.close()
    
    return model, best_acc

# Function to evaluate and visualize model performance
def evaluate_model(model, testloader, classes):
    model.eval()
    
    # Collect predictions and ground truth
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                confusion_matrix[label][pred] += 1
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Print class accuracies
    print("\nClass-wise Accuracy:")
    for i in range(len(classes)):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'- {classes[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    # Calculate overall accuracy
    overall_accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    plt.savefig(os.path.join(project_root, 'gender_confusion_matrix.png'))
    plt.close()
    
    return overall_accuracy, class_correct, class_total

# Main execution
if __name__ == "__main__":
    # Setup paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    train_path = os.path.join(project_root, '..', 'real_dataset', 'train', 'humans')
    test_path = os.path.join(project_root, '..', 'real_dataset', 'test', 'humans')

    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Gender detection classes
    classes = ['men', 'women']
    
    # Check if dataset directories exist
    if not os.path.exists(train_path):
        print(f"❌ Training directory not found: {train_path}")
        print("Please create the following directory structure before training:")
        print(f"  {os.path.join(train_path, 'men')}")
        print(f"  {os.path.join(train_path, 'women')}")
        print(f"  {os.path.join(test_path, 'men')}")
        print(f"  {os.path.join(test_path, 'women')}")
        exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"Loading datasets from {train_path} and {test_path}...")
    try:
        trainset = datasets.ImageFolder(root=train_path, transform=train_transform)
        testset = datasets.ImageFolder(root=test_path, transform=test_transform)
        
        # Calculate class weights to handle imbalanced data
        class_counts = [0] * len(trainset.classes)
        for _, label in trainset.samples:
            class_counts[label] += 1
        
        print(f"Class distribution: {trainset.classes}")
        print(f"Class counts: {class_counts}")
        
        # Create weighted sampler for imbalanced classes
        class_weights = [1.0 / count for count in class_counts]
        sample_weights = [class_weights[label] for _, label in trainset.samples]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Create data loaders
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=32, sampler=sampler, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False, num_workers=2
        )
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        exit(1)
    
    # Visualize sample images
    visualize_samples(trainset, classes)
    
    # Create the model, loss function, optimizer, and scheduler
    model = GenderCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)
    
    print(f"🔍 Starting training for {len(classes)} classes: {classes}")
    
    # Train the model
    model, best_acc = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        trainloader=trainloader,
        testloader=testloader,
        scheduler=scheduler,
        epochs=30,
        early_stop_patience=5
    )
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_cnn_gender.pth')))
    
    # Evaluate the model
    print("\n📊 Evaluating model performance...")
    accuracy, class_correct, class_total = evaluate_model(model, testloader, classes)
    
    print("\n✅ Training and evaluation complete.")
    print(f"🏅 Best accuracy: {best_acc:.2f}%")
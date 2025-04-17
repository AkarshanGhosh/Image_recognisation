import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import WeightedRandomSampler

# Define improved transformations for training data
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),  # Increased from 32x32 to 64x64
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomCrop(64, padding=8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Simpler transformations for testing data
transform_test = transforms.Compose([
    transforms.Resize((64, 64)),  # Increased to match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    # Dataset base path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'real_dataset'))
    
    # Only use animals and humans folders
    train_dirs = [os.path.join(base_dir, 'train', folder) for folder in ['animals', 'humans']]
    test_dirs = [os.path.join(base_dir, 'test', folder) for folder in ['animals', 'humans']]
    
    # Load datasets separately
    train_datasets = [datasets.ImageFolder(root=dir_path, transform=transform_train) for dir_path in train_dirs if os.path.exists(dir_path)]
    test_datasets = [datasets.ImageFolder(root=dir_path, transform=transform_test) for dir_path in test_dirs if os.path.exists(dir_path)]
    
    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    # Calculate class weights to handle imbalanced data
    # First, need to get the class counts from train_datasets
    class_counts = {}
    for dataset in train_datasets:
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
    
    print(f"Class distribution: {class_counts}")
    
    # Create weighted sampler for the training dataset
    # We need to convert ConcatDataset back to single dataset for this
    # For simplicity, we'll use the first dataset's class structure
    # This approach works if the classes are consistent across datasets
    if train_datasets:
        all_samples = []
        all_labels = []
        for dataset in train_datasets:
            all_samples.extend(dataset.samples)
            for _, label in dataset.samples:
                all_labels.append(label)
        
        # Calculate weights for each sample
        class_weights = [1.0 / class_counts[train_datasets[0].classes[label]] for label in all_labels]
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)
        
        # Create data loaders with the sampler for balanced training
        trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=32, 
            sampler=sampler, 
            num_workers=2
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=2
        )
    
    testloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2
    )
    
    # Function to visualize some images
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    # Visualize samples function (similar to the one in your document)
    def visualize_samples(dataset, n=5):
        if not hasattr(dataset, 'classes'):
            # If using ConcatDataset, access the first dataset for classes
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                dataset = dataset.datasets[0]
        
        fig, axes = plt.subplots(len(dataset.classes), n, figsize=(15, 10))
        for i, c in enumerate(dataset.classes):
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
        plt.savefig(os.path.join(os.path.dirname(base_dir), 'samples.png'))
        plt.show()
    
    # Try to visualize samples if possible
    if train_datasets:
        visualize_samples(train_datasets[0])
    
    # Get a batch of training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # Show images
    imshow(torchvision.utils.make_grid(images[:8]))
    print("Labels:", labels[:8])
    
    return trainloader, testloader

if __name__ == "__main__":
    trainloader, testloader = main()
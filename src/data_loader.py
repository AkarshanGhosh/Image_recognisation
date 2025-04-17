import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os

# Define transformations for training and testing
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    # Dataset base path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'real_dataset'))
    
    # Only use animals and humans folders
    train_dirs = [os.path.join(base_dir, 'train', folder) for folder in ['animals', 'humans']]
    test_dirs = [os.path.join(base_dir, 'test', folder) for folder in ['animals', 'humans']]

    # Load datasets separately and merge them
    train_datasets = [datasets.ImageFolder(root=dir_path, transform=transform_train) for dir_path in train_dirs if os.path.exists(dir_path)]
    test_datasets = [datasets.ImageFolder(root=dir_path, transform=transform_test) for dir_path in test_dirs if os.path.exists(dir_path)]

    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    # Define DataLoaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Function to visualize some images
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Get a batch of training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    imshow(torchvision.utils.make_grid(images[:8]))
    print("Labels:", labels[:8])

if __name__ == "__main__":
    main()

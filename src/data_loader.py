import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize between -1 and 1
])

def main():
    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Define DataLoaders for batch processing
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)  # Change num_workers=0 for Windows
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    # Define class names for CIFAR-10 categories
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Function to visualize some images from the dataset
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize (convert back to 0-1 range)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert to (H, W, C) format
        plt.show()

    # Get a batch of training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show some images
    imshow(torchvision.utils.make_grid(images[:8]))  # Show first 8 images
    print(' '.join(classes[labels[j]] for j in range(8)))  # Print their labels
    


    
if __name__ == "__main__":
    main()

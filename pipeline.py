import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Global config
DATASET_PATH = 'train_dataset'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def get_transforms():
    """Define transformations for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    return train_transform, val_transform


def load_datasets(dataset_path=DATASET_PATH):
    """Load train and validation datasets."""
    train_transform, val_transform = get_transforms()

    full_dataset = datasets.ImageFolder(dataset_path)

    # Split into train/val (80/20)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    return train_dataset, val_dataset, full_dataset.classes


def create_loaders(train_dataset, val_dataset):
    """Create DataLoader objects for training and validation."""
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def visualize_batch(data_loader, class_names):
    """Visualize a batch of 9 images with labels."""
    images, labels = next(iter(data_loader))
    images = images[:9]
    labels = labels[:9]

    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].permute(1, 2, 0)  # CHW to HWC
        plt.imshow((img * 0.5 + 0.5).numpy())  # Unnormalize
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

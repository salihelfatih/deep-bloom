import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(batch_size=32):
    """Creates dataloaders for the Flowers102 dataset."""

    # 🖼️ Define image size and normalization parameters
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # 👟 Define transforms for training data (with augmentation)
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )

    # ☑️ Define transforms for validation and test data (no augmentation)
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )

    # 📦 Load the Flowers102 dataset
    train_dataset = datasets.Flowers102(
        root="./data", split="train", download=True, transform=train_transforms
    )
    val_dataset = datasets.Flowers102(
        root="./data", split="val", download=True, transform=test_transforms
    )
    test_dataset = datasets.Flowers102(
        root="./data", split="test", download=True, transform=test_transforms
    )

    # 🔃 Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

import torch.nn as nn
from torchvision import models


def create_deep_bloom_model():
    """Creates a ResNet18 model fine-tuned for the Flowers102 dataset."""

    # 📦 Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)

    # ⛄ Freeze all the model's layers
    for param in model.parameters():
        param.requires_grad = False

    # 🌷 Replace the final layer for our 102 flower classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)

    return model

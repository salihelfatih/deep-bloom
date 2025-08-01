import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def create_fulla_model():
    """Creates a ResNet18 model fine-tuned for the Flowers102 dataset."""

    # 📦 Load a pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # ⛄ Freeze all the model's layers
    for param in model.parameters():
        param.requires_grad = False

    # 🌷 Replace the final layer for our 102 flower classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)

    return model

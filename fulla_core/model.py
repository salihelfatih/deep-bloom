import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# 🌷 Creating a ResNet18 model fine-tuned for the Flowers102 dataset
def create_fulla_model():
    """Creates a ResNet18 model fine-tuned for the Flowers102 dataset."""

    # Loading the pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freezing the layers of the model
    for param in model.parameters():
        param.requires_grad = False

    # Replacing the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)

    return model

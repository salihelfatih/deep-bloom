import torch
import torch.nn as nn
import torch.optim as optim
from model import create_fulla_model
from utils import create_dataloaders

# 🌼 Main Training Script
if __name__ == "__main__":
    # Setting device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Preparing data and model
    train_loader, val_loader, _ = create_dataloaders()
    model = create_fulla_model()
    model.to(device)

    # Defining training tools
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Setting training loop
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train()  # Setting model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Setting model to evaluation mode for validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} -> Validation Accuracy: {val_acc:.2f}%")
        scheduler.step()

    # Saving the trained model
    torch.save(model.state_dict(), "../fulla_model.pth")
    print("\nFinished Training! Model saved to fulla_model.pth")

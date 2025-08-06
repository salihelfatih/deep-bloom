import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from model import create_fulla_model
from utils import create_dataloaders

# 🌼 Flower Class Names
if __name__ == "__main__":
    # Setting the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    _, _, test_loader = create_dataloaders()

    # 🌸 Loading the model
    model = create_fulla_model()
    # Load the saved weights from your .pth file
    model.load_state_dict(torch.load("../fulla_model.pth"))
    model.to(device)
    model.eval()  # Setting model to evaluation mode

    # Running inference on the test set
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Finished inference on the test set.")

    # Calculating accuracy
    accuracy = (
        100 * sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
    )
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

    # Calculating F1 Score
    f1 = f1_score(
        y_true, y_pred, average="weighted"
    )  # 'weighted' accounts for any imbalance in the number of samples per class
    print(f"Final F1 Score: {f1:.4f}")

    # Plotting the confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 20))  # Increasing figure size for better visibility
    sns.heatmap(
        conf_matrix,
        annot=False,  # Setting annottation to False for cleaner output
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(
        "../confusion_matrix.png"
    )  # Saving the confusion matrix plot as an image
    print("Confusion matrix saved to confusion_matrix.png")

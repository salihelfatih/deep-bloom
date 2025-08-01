import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from model import create_fulla_model
from utils import create_dataloaders

# 📊 Main Evaluation Script
if __name__ == "__main__":
    # 📦 Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🧪 Load the test data
    _, _, test_loader = create_dataloaders()

    # 🔃 Load the trained model
    model = create_fulla_model()
    # Load the saved weights from your .pth file
    model.load_state_dict(torch.load("../fulla_model.pth"))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # 🏃‍♂️ Run Inference and Collect Prediction
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

    # 🔍 Calculate accuracy
    accuracy = (
        100 * sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
    )
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

    # 🧮 Calculate F1 Score
    f1 = f1_score(
        y_true, y_pred, average="weighted"
    )  # 'weighted' accounts for any imbalance in the number of samples per class
    print(f"Final F1 Score: {f1:.4f}")

    # 🎨 Plot Confusion Matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 20))  # Increase figure size for 102 classes
    sns.heatmap(
        conf_matrix, annot=False
    )  # Annotations off for clarity with many classes
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("../confusion_matrix.png")  # Save the plot as a file
    print("Confusion matrix saved to confusion_matrix.png")

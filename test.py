import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import BrainTumorCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "brain_tumor_cnn.pth"
TEST_DIR = "test_dataset"
BATCH_SIZE = 32

def load_test_loader():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # must match training
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes
    return test_loader, class_names

def test():
    # Load model
    model = BrainTumorCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_loader, class_names = load_test_loader()
    print("Testing on", len(test_loader.dataset), "images.")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy())

    # === Metrics ===
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.show()

    visualize_predictions(model, test_loader, ['No Tumor', 'Tumor'], DEVICE)


def unnormalize(img_tensor, mean, std):
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor

def visualize_predictions(model, loader, class_names, device, num_images=8):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(16, 8))

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                # img = images[i].cpu().permute(1, 2, 0).numpy()
                img = images[i].cpu().clone()
                img = unnormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img = img.permute(1, 2, 0).numpy()
                true_label = class_names[labels[i]]
                pred_label = class_names[preds[i][0]]

                plt.subplot(2, num_images // 2, images_shown + 1)
                plt.imshow(img)
                plt.title(f"True: {true_label}\nPred: {pred_label}", color='green' if true_label == pred_label else 'red')
                plt.axis('off')
                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.suptitle("Test Predictions", fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()


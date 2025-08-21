import torch
import torch.nn as nn
import torch.optim as optim
from model import BrainTumorCNN
from pipeline import load_datasets, create_loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

# ====== Hyperparameters ======
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1
BEST_MODEL_PATH = "Best_brain_tumor_model.pth"
FINAL_MODEL_PATH = "brain_tumor_cnn.pth"


# ====== Training Utilities ======
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / len(loader.dataset)
    return val_loss, val_acc

# ====== Main Training Function ======
def main():
    print(f"Using device: {DEVICE}")

    # Load data
    train_dataset, val_dataset, _ = load_datasets()
    train_loader, val_loader = create_loaders(train_dataset, val_dataset)

    # Initialize model, loss, optimizer
    model = BrainTumorCNN().to(DEVICE)
    # Compute pos_weight from training labels
    train_labels = [label for _, label in train_dataset]
    label_counts = Counter(train_labels)
    neg = label_counts[0]
    pos = label_counts[1]
    pos_weight_value = (neg / pos) * 0.5  # or even 0.3
    print(f"Using pos_weight = {pos_weight_value:.2f} (neg={neg}, pos={pos})")
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE)

    # Define loss with class weighting
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    train_losses, train_accuracies = [], []

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Saved new best model to {BEST_MODEL_PATH}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

    # === 1. Plot Loss vs Accuracy ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_accuracy_plot.png")
    plt.show()

    # === 2. Confusion Matrix + Classification Report ===
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.3).int().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # === 3. Save the final model ===
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nModel saved to: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()

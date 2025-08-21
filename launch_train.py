from pipeline import load_datasets, create_loaders, visualize_batch
from train import main as train_model

def main():
    # Load datasets and visualize sample batch
    train_dataset, val_dataset, class_names = load_datasets()
    train_loader, val_loader = create_loaders(train_dataset, val_dataset)
    visualize_batch(train_loader, class_names)

    # Train and evaluate the model
    train_model()

if __name__ == "__main__":
    main()

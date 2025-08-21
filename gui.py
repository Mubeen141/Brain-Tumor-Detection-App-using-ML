import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from torchvision import transforms
from PIL import Image
from model import BrainTumorCNN
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Constants
MODEL_PATH = "brain_tumor_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)

# Load model
model = BrainTumorCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class TumorDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Detector")
        self.setGeometry(200, 200, 400, 400)

        # Widgets
        self.label = QLabel("Upload a brain MRI image", self)
        self.image_label = QLabel(self)
        self.result_label = QLabel("", self)
        self.button = QPushButton("Upload Image", self)
        self.button.clicked.connect(self.upload_image)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            # Show image
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.predict(file_path)

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).item()
            label = "Tumor" if pred > 0.5 else "No Tumor"
            confidence = pred if pred > 0.5 else 1 - pred
            confidence = max(0.0, min(confidence, 1.0))
            self.result_label.setText(f"Prediction: {label} ({confidence*100:.2f}%)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TumorDetectionApp()
    window.show()
    sys.exit(app.exec_())

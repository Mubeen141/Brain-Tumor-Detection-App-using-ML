from gui import TumorDetectionApp
from PyQt5.QtWidgets import QApplication
import sys

def main():
    # Launch GUI
    print("\nLaunching BrainMRI Checker")
    app = QApplication(sys.argv)
    window = TumorDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

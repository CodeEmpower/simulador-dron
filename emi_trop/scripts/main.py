#Este archivo será el punto de entrada principal para iniciar la aplicación.
#!/usr/bin/env python
import os
import sys
import cv2
import yaml

from PyQt5.QtWidgets import QApplication
from ui_drone_control import DroneControlApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneControlApp()
    window.show()
    sys.exit(app.exec_())

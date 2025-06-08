#!/usr/bin/env python
import os
import sys
import rospy
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QSlider
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

class ImageProcessingThread(QThread):
    image_processed = pyqtSignal(QImage)

    def __init__(self, bridge, net, output_layers, person_class_id):
        super().__init__()
        self.bridge = bridge
        self.net = net
        self.output_layers = output_layers
        self.person_class_id = person_class_id
        self.latest_image = None
        self.frame_skip = 0
        self.is_thermal_mode = False
        self.brightness_factor = 1.0  # Control de brillo

    def run(self):
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                self.process_image(self.latest_image)

    def process_image(self, cv_image):
        self.frame_skip += 1
        if self.frame_skip % 5 != 0:
            return

        if self.is_thermal_mode:
            cv_image = self.convert_to_thermal(cv_image)

        # Aplicar el efecto de brillo
        cv_image = cv2.convertScaleAbs(cv_image, alpha=self.brightness_factor, beta=0)

        resized_image = cv2.resize(cv_image, (420, 420))

        blob = cv2.dnn.blobFromImage(resized_image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        height, width, _ = cv_image.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == self.person_class_id and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"Persona {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_processed.emit(q_image)

    def convert_to_thermal(self, cv_image):
        return cv2.applyColorMap(cv_image, cv2.COLORMAP_JET)

    def set_image(self, image):
        self.latest_image = image

    def set_brightness(self, value):
        # Ajustar el factor de brillo
        self.brightness_factor = value / 100.0

class DroneControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drone Control Interface')
        self.setGeometry(100, 100, 700, 600)

        rospy.init_node('drone_control_gui', anonymous=True)
        self.bridge = CvBridge()
        self.load_yolo_model()
        self.init_ui()

        self.processing_thread = ImageProcessingThread(self.bridge, self.net, self.output_layers, self.person_class_id)
        self.processing_thread.image_processed.connect(self.update_camera_feed)
        self.processing_thread.start()

        self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, self.image_cb)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_state)
        self.timer.start(1000)

    def init_ui(self):
        layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)

        self.toggle_button = QPushButton('Cambiar a modo térmico')
        self.toggle_button.clicked.connect(self.toggle_mode)
        self.toggle_button.setFixedHeight(50)
        layout.addWidget(self.toggle_button)

        self.capture_button = QPushButton('Capturar Imagen')
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setFixedHeight(50)
        layout.addWidget(self.capture_button)

        # Slider para ajustar brillo
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)  # Valor inicial del brillo
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        layout.addWidget(self.brightness_slider)

        self.setLayout(layout)

    def load_yolo_model(self):
        yolo_path = "/home/niwre21/catkin_ws/src/emi_trop/config"
        try:
            self.net = cv2.dnn.readNet(os.path.join(yolo_path, "yolov3.weights"), 
                                       os.path.join(yolo_path, "yolov3.cfg"))
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

            with open(os.path.join(yolo_path, "yolov3.yaml"), 'r') as file:
                config = yaml.safe_load(file)
                self.classes = config['yolo_model']['detection_classes']['names']

            self.person_class_id = self.classes.index("person")
        except Exception as e:
            rospy.logerr(f"Error cargando modelo YOLO: {e}")
            sys.exit(1)

    def image_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error en CvBridge: {e}")
            return

        self.processing_thread.set_image(cv_image)

    def update_camera_feed(self, processed_image):
        pixmap = QPixmap.fromImage(processed_image)
        self.camera_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def update_state(self):
        pass

    def toggle_mode(self):
        self.processing_thread.is_thermal_mode = not self.processing_thread.is_thermal_mode
        mode = "normal" if self.processing_thread.is_thermal_mode else "térmico"
        self.toggle_button.setText(f'Cambiar a modo {mode}')

    def capture_image(self):
        if self.processing_thread.latest_image is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_path = "/home/niwre21/captured_images"
            os.makedirs(folder_path, exist_ok=True)
            image_path = os.path.join(folder_path, f"image_{timestamp}.png")
            cv2.imwrite(image_path, self.processing_thread.latest_image)
            rospy.loginfo(f"Imagen guardada en {image_path}")
        else:
            rospy.logwarn("No hay imagen para capturar.")

    def adjust_brightness(self, value):
        # Actualizar el brillo basado en el valor del slider
        self.processing_thread.set_brightness(value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneControlApp()
    window.show()
    sys.exit(app.exec_())

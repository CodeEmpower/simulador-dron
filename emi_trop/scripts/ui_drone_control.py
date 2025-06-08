#Este archivo contiene la clase DroneControlApp que maneja la interfaz gráfica. Aquí también agregarás un nuevo botón.
import sys
import rospy
import cv2
import os
import yaml

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from image_processing import ImageProcessingThread
from mavros_msgs.msg import BatteryStatus

class DroneControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drone Control Interface')
        self.setGeometry(100, 100, 800, 600)

        rospy.init_node('drone_control_gui', anonymous=True)
        self.bridge = CvBridge()
        self.load_yolo_model()
        self.init_ui()
        self.show()
        self.battery_sub = rospy.Subscriber('/mavros/battery', BatteryStatus, self.battery_cb)



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

        self.battery_label = QLabel("Estado de la Batería: Desconocido")
        self.battery_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.battery_label)

        self.setLayout(layout)
    def battery_cb(self, msg):
        # Actualizar la etiqueta de la batería con el estado actual
        voltage = msg.voltage  # Voltaje de la batería
        current = msg.current  # Corriente de la batería
        percentage = msg.percentage  # Porcentaje de la batería
        status = msg.status  # Estado de la batería (1 = buena, 0 = mala)

        # Mostrar los datos en la interfaz gráfica
        battery_status = f"Voltaje: {voltage}V\nCorriente: {current}A\nPorcentaje: {percentage}%"
        self.battery_label.setText(battery_status)

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
        mode = "térmico" if self.processing_thread.is_thermal_mode else "normal"
        self.toggle_button.setText(f'Cambiar a modo {mode}')

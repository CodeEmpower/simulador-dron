#!/usr/bin/env python
import rospy
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QGroupBox, QFormLayout, QPushButton, QApplication
from sensor_msgs.msg import NavSatFix, Image
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge
from PyQt5.QtGui import QImage, QPixmap
import cv2
import sys

class DroneControlGUI(QWidget):
    def __init__(self):
        super().__init__()

        # ROS initialization
        rospy.init_node('drone_control_gui', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.update_gps)
        rospy.Subscriber('/mavros/global_position/rel_alt', Float64, self.update_altitude)
        rospy.Subscriber('/mavros/state', State, self.update_state)
        rospy.Subscriber('/webcam/image_raw', Image, self.update_camera)

        # Publishers
        self.local_pos_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        
        # Services for arming and mode changing
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # OpenCV bridge
        self.bridge = CvBridge()

        # UI setup
        self.setWindowTitle("Drone Control GUI")
        self.layout = QVBoxLayout()

        # GPS and Altitude GroupBox
        gps_alt_group = QGroupBox("Información del Dron")
        gps_alt_layout = QFormLayout()
        self.gps_label = QLabel("GPS: Lat: 0.0, Lon: 0.0")
        self.altitude_label = QLabel("Altitud: 0.0m")
        self.state_label = QLabel("Estado: Desconocido")
        gps_alt_layout.addRow("GPS:", self.gps_label)
        gps_alt_layout.addRow("Altitud:", self.altitude_label)
        gps_alt_layout.addRow("Estado:", self.state_label)
        gps_alt_group.setLayout(gps_alt_layout)
        self.layout.addWidget(gps_alt_group)

        # Camera GroupBox
        camera_group = QGroupBox("Cámara")
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(500, 500)
        self.toggle_button = QPushButton("Cambiar a detección de infrarrojos")
        self.toggle_button.clicked.connect(self.toggle_detection_mode)
        
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.toggle_button)
        camera_group.setLayout(camera_layout)
        self.layout.addWidget(camera_group)

        # Set layout
        self.setLayout(self.layout)

        # Mode
        self.detection_mode = False

    def toggle_detection_mode(self):
        self.detection_mode = not self.detection_mode
        mode_text = "Normal" if not self.detection_mode else "Detección de infrarrojos"
        self.toggle_button.setText(f"Cambiar a modo {mode_text}")

    def update_gps(self, data):
        self.gps_label.setText(f"GPS: Lat: {data.latitude:.6f}, Lon: {data.longitude:.6f}")

    def update_altitude(self, data):
        self.altitude_label.setText(f"Altitud: {data.data:.2f}m")

    def update_state(self, data):
        state_text = f"Estado: {'Armado' if data.armed else 'Desarmado'}, Modo: {data.mode}"
        self.state_label.setText(state_text)

    def update_camera(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            if self.detection_mode:
                # Apply infrared simulation and person detection
                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                _, thresholded = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                cv_image = cv2.merge((thresholded, thresholded, thresholded))
                
                # Detect people (simulated with contours)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 30 and h > 30:  # Filter out small contours
                        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.camera_label.setPixmap(QPixmap.fromImage(q_image))
        except Exception as e:
            rospy.logerr(f"Error al procesar la imagen: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DroneControlGUI()
    gui.show()
    sys.exit(app.exec_())

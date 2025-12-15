import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO

class YOLOWebcamApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model = YOLO(r"C:/Users/Amir/Desktop/Yolo8 Guitar/guitar_detector/weights/best.pt")
        self.cap = cv2.VideoCapture(0)

        self.setWindowTitle("YOLOv8 Webcam Detection")
        self.image_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Timer for reading webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)   # ~33 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model.predict(frame, conf=0.5)
        annotated = results[0].plot()

        # Convert to RGB → QImage
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOWebcamApp()
    window.show()
    sys.exit(app.exec_())

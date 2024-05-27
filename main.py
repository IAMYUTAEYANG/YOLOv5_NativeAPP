import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from PIL import Image
import pathlib  # PosixPath 오류 해결
pathlib.PosixPath = pathlib.WindowsPath  # PosixPath 오류 해결

# 모델을 로드하는 함수
def load_model():
    model = torch.hub.load(repo_or_dir='ultralytics/yolov5', model='yolov5s', pretrained=True)
    return model

# 이미지를 전처리하는 함수
def process_image(image):
    return image

# 추론을 수행하는 함수
def infer(model, image):
    with torch.no_grad():  # 그래디언트 계산을 비활성화
        predictions = model(image)  # 모델을 사용하여 추론 수행
    return predictions

# 추론 결과를 처리하는 함수
def process_predictions(frame, predictions):
    if not predictions.xyxy[0].size(0):
        return frame

    class_names = predictions.names
    bboxes = predictions.xyxy[0].cpu().numpy()  # bounding boxes
    for bbox in bboxes:
        if len(bbox) < 6:
            continue  # 유효한 예측 결과가 아니면 건너뜀
        class_index = int(bbox[5])
        class_name = class_names[class_index]
        confidence = bbox[4]
        if confidence > 0.5:  # confidence threshold 설정
            xmin, ymin, xmax, ymax = bbox[:4].astype(int)  # 변경된 부분
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

class FallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fall Detection App")
        self.setGeometry(100, 100, 1200, 600)  # 가로 크기를 넓힘

        # 모델 로드
        self.model = load_model()

        self.initUI()
        self.webcam = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image")
        self.image_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")

        self.result_label = QLabel(self)
        self.result_label.resize(640, 480)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setText("No Image")
        self.result_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.start_webcam_button = QPushButton("Start Webcam", self)
        self.start_webcam_button.clicked.connect(self.start_webcam)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.start_webcam_button)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.result_label)

        layout = QVBoxLayout()
        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        self.timer.stop()  # 웹캠 타이머 중지
        self.clear_labels()  # 화면 초기화

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Image Files (*.png;*.jpg;*.jpeg)", options=options)
        if fileName:
            image = Image.open(fileName).convert('RGB')
            image = np.array(image)[:, :, ::-1]  # RGB -> BGR 변환

            # 원본 이미지를 왼쪽 화면에 표시
            self.display_image(image, self.image_label)

            processed_image = process_image(image)
            predictions = infer(self.model, processed_image)
            result_image = process_predictions(image.copy(), predictions)
            # 감지 결과 이미지를 오른쪽 화면에 표시
            self.display_image(result_image, self.result_label)

    def start_webcam(self):
        self.clear_labels()  # 화면 초기화
        self.timer.start(30)  # 웹캠 타이머 시작

    def update_frame(self):
        ret, frame = self.webcam.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_image = process_image(frame_rgb)
            predictions = infer(self.model, processed_image)
            result_frame = process_predictions(frame_rgb.copy(), predictions)
            # 원본 웹캠 프레임을 왼쪽 화면에 표시
            self.display_image(frame_rgb, self.image_label)  # RGB 형식으로 표시
            # 감지 결과 웹캠 프레임을 오른쪽 화면에 표시
            self.display_image(result_frame, self.result_label)  # RGB 형식으로 표시

    def clear_labels(self):
        self.image_label.clear()
        self.result_label.clear()
        self.image_label.setText("No Image")
        self.result_label.setText("No Image")

    def display_image(self, image, label):
        if image is None or image.size == 0:
            return
        # Convert the image to RGB before creating QImage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec_())

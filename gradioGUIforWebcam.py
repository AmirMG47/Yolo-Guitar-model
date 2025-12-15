import cv2
from ultralytics import YOLO
import gradio as gr

model = YOLO(r"C:/Users/---model path---/best.pt")

def predict_frame(frame):
    if frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, conf=0.5)
    annotated = results[0].plot()

    return annotated

iface = gr.Interface(
    fn=predict_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=gr.Image(),
    live=True,
    title="YOLOv8 Webcam Detection"
)

iface.launch()

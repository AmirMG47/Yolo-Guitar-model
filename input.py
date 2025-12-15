from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import shutil
import uuid
import os
import cv2

app = FastAPI()

# مدل را اینجا بارگذاری می‌کنیم
model = YOLO("model/best.pt")


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    temp_path = f"temp_{uid}.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "class_id": int(box.cls[0]),
            "class_name": results.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    os.remove(temp_path)
    return {"detections": detections}


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    temp_path = f"temp_{uid}.mp4"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(temp_path)

    frame_detections = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame_result = []

        for box in results.boxes:
            frame_result.append({
                "class_id": int(box.cls[0]),
                "class_name": results.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

        frame_detections.append({
            "frame": frame_index,
            "detections": frame_result
        })

        frame_index += 1

    cap.release()
    os.remove(temp_path)

    return {"video_frames": frame_detections}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

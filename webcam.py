from ultralytics import YOLO
import cv2

# مدل را لود کن
model = YOLO(r"C:/Users/---model path---/best.pt")

# وب‌کم را فعال کن (0 یعنی وب‌کم اصلی لپ‌تاپ/سیستم)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("وب‌کم باز نشد. بررسی کن که دوربین آزاد باشد.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("فریم دریافت نشد.")
        break

    # اجرای YOLO روی فریم
    results = model.predict(frame, conf=0.5)

    # رسم خروجی روی فریم
    annotated_frame = results[0].plot()

    # نمایش تصویر
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # خروج با کلید q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

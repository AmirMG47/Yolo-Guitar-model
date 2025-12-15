# تصویر تستی
from ultralytics import YOLO

model = YOLO('C:/Users/Amir/Desktop/Yolo8 Guitar/guitar_detector/weights/best.pt')
test_image = r"C:/Users/Amir/Desktop/Yolo8 Guitar/1.jpg"
predict_results = model.predict(
    source=test_image,
    save=True,          # خروجی را در runs ذخیره می‌کند
    conf=0.5,
)

print(predict_results)

# تصویر تستی
from ultralytics import YOLO

model = YOLO('C:/Users/---your path---/best.pt')
test_image = r"C:/Users/---image path---/1.jpg"
predict_results = model.predict(
    source=test_image,
    save=True,          # خروجی را در runs ذخیره می‌کند
    conf=0.5,
)

print(predict_results)

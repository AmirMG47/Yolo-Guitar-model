from ultralytics import YOLO


data_yaml = r'C:/Users/---path---/Guitar_v8.yaml'

model = YOLO('yolov8s.pt')

results = model.train(
    data=data_yaml,
    epochs =  5,
    imgsz = 416,
    batch=8,
    plots = True,
    device = 'cpu',
    save = True,
)
print(' Training complete')




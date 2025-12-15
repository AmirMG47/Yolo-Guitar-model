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



#model.train(data="Guitar_v8.yaml",epoch = 5, imgsz=640,dryrun=True)



#Amuzeshe model
#yolo task=detect mode=train model=yolov8s.pt data=C:/Users/Amir/Desktop/Yolo8Guitar/Guitar_dataset/Guitar_v8.yaml epochs=5 imgsz=640 batch=16


#arzyabi model
#yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=C:/Users/Amir/Desktop/Yolo8Guitar/Guitar_dataset/Guitar_v8.yaml

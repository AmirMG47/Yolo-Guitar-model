from ultralytics import YOLO
import torch

# این شرط الزامی است برای ویندوز تا multiprocessing درست کار کند
if __name__ == '__main__':

    # بررسی GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # مسیر دیتاست YAML
    data_yaml = r'C:/Users/Amir/Desktop/Yolo8 Guitar/Guitar_v8.yaml'

    # بارگذاری مدل پایه YOLO
    model = YOLO('yolov8s.pt')

    # آموزش مدل (بهینه برای کارت 1650 با 4GB VRAM)
    results = model.train(
        data=data_yaml,
        epochs=50,                     # تعداد epoch برای دیتاست کوچک
        imgsz=416,                     # اندازه تصویر کمتر → مصرف VRAM کمتر
        batch=4,                       # batch کوچک برای جلوگیری از OOM
        device=device,                 # GPU یا CPU بسته به شناسایی
        project=r'C:/Users/Amir/Desktop',  # مسیر دلخواه ذخیره
        name='guitar_detector',        # اسم پروژه
        save=True,                     # ذخیره مدل
        plots=True                      # نمایش نمودار آموزش
    )

    print("Training complete.")
    #print("Model saved at:", 
          #r'C:/Users/Amir/Desktop/yolo/output/guitar_detector/weights/best.pt')

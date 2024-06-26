from ultralytics import YOLO

#YOLOv8 model
model = YOLO('yolov8n.pt')

model.train(data='/Users/parmarparth/ultralytics/ultralytics/data/coco128.yaml', 
            epochs=50, 
            batch=16, 
            imgsz=640, 
            name='train4')

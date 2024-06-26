import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train43/weights/best.pt') 

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model(frame)
    for result in results:
        result_frame = result.plot()

    cv2.imshow('YOLOv8 Live Object Detection', result_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

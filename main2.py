import cv2
import torch
from ultralytics import YOLO

 
model = YOLO("yolov8n.pt")


coco_classes = ["airplane", "bird"]  
coco_class_ids = [4, 14]  

 
video_path = "input2.mp4"  
cap = cv2.VideoCapture(video_path)

 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

 
output_path = "output_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

     
    results = model(frame)

    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   
            class_id = int(box.cls[0])   
            conf = box.conf[0].item()  

            
            detected_class = model.names[class_id]
            print(f"Detected: {detected_class} ({conf:.2f})")

             
            if class_id in coco_class_ids:
                label = f"{detected_class}: {conf:.2f}"
                color = (0, 255, 0)  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    cv2.imshow("Aerial Object Detection", frame)
    out.write(frame)   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
out.release()
cv2.destroyAllWindows()

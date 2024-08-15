import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

cap = cv2.VideoCapture(0) 

smooth_factor = 0.7
prev_coords = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    detections = results.xyxy[0].cpu().numpy()

    if prev_coords is None:
        prev_coords = detections[:, :4]

    if len(detections) > 0:
        current_coords = detections[:, :4]
        if prev_coords.shape == current_coords.shape:
            smoothed_coords = smooth_factor * current_coords + (1 - smooth_factor) * prev_coords
            detections[:, :4] = smoothed_coords

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.3:
            try:
                label = results.names[int(cls)]
            except IndexError:
                label = f"Unknown_{int(cls)}"

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prev_coords = detections[:, :4]

    cv2.imshow('YOLOv5 Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
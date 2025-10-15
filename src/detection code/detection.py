import cv2
from ultralytics import YOLO  # Works for YOLOv8/YOLOv9

# === Define paths ===
MODEL_PATH = r'C:\Users\ABHISHEK ARUN RAJA\Documents\Coding Projects\pothole-detection\models\best.pt'    # your trained YOLO model
VIDEO_PATH = r'C:\Users\ABHISHEK ARUN RAJA\Documents\Coding Projects\pothole-detection\test_videos\Pothole video.mp4'      # your .mp4 file path

# === Load model ===
model = YOLO(MODEL_PATH)

# === Load video ===
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# === Process each frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Display frame
    cv2.imshow("Pothole Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# 1. Load Pose and Object Detection models
pose_model = YOLO("yolov8n-pose.pt")
ball_model = YOLO("yolov8n.pt")

# 2. Open input video clip
input_video_path = "data/Lionel_Messi.mp4"
cap = cv2.VideoCapture(input_video_path)

# Extract video dimensions and FPS to set up video saving
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# 3. Video Writer setup to export the final processed video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_analytics.mp4", fourcc, fps, (width, height))

print("Processing full video frame-by-frame...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video finished

    # 4. Detect player skeletons (Pose Estimation) on current frame
    pose_results = pose_model(frame, verbose=False)[0]
    annotated_frame = pose_results.plot()

    # 5. Detect ball (Class 32 = sports ball) on current frame
    ball_results = ball_model(frame, classes=[32], verbose=False)[0]
    for box in ball_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])

        # Draw yellow box and label around the ball
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated_frame,
            f"ball {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    # 6. Write annotated frame to the output video file
    out.write(annotated_frame)

    # Display live preview window
    cv2.imshow("PitchIntel AI - Video Analytics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Success! Processed video saved as 'output_analytics.mp4'.")

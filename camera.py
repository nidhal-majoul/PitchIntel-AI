import cv2
import numpy as np
from ultralytics import YOLO
from team_classifier import (
    extract_jersey_pixels, 
    get_dominant_color, 
    fit_team_classifier, 
    assign_team
)

# Load pose and object detection models
pose_model = YOLO("yolov8n-pose.pt")
ball_model = YOLO("yolov8n.pt")

# Set up video capture and export parameters
input_video_path = "data/Lionel_Messi.mp4"
cap = cv2.VideoCapture(input_video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_team_analytics.mp4", fourcc, fps, (width, height))

# -------------------------------------------------------------------
# Phase 1: Warmup — Sample initial frames to train global team model
# -------------------------------------------------------------------
print("[+] Sampling initial frames for team color clustering...")
collected_colors = []
sample_frames = 45  # Process first 45 frames for clustering data

while cap.isOpened() and len(collected_colors) < 100 and sample_frames > 0:
    ret, frame = cap.read()
    if not ret:
        break
    sample_frames -= 1
    
    pose_results = pose_model(frame, verbose=False)[0]
    if pose_results.boxes is not None:
        for box in pose_results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            jersey_pixels = extract_jersey_pixels(frame, bbox)
            
            # Keep valid crops with enough non-grass pixels
            if jersey_pixels is not None and len(jersey_pixels) > 50:
                dom_color = get_dominant_color(jersey_pixels)
                collected_colors.append(dom_color)

# Fit 2-team KMeans classifier
team_kmeans = fit_team_classifier(collected_colors)
print("[+] Global team classifier trained successfully.")

# Rewind capture back to frame 0 for main output processing
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -------------------------------------------------------------------
# Phase 2: Pipeline loop — Predict teams, annotate, and write video
# -------------------------------------------------------------------
print("[+] Rendering final visual team analytics video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # Detect players and predict teams
    pose_results = pose_model(frame, verbose=False)[0]
    if pose_results.boxes is not None:
        for box in pose_results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)

            jersey_pixels = extract_jersey_pixels(frame, bbox)
            
            if jersey_pixels is not None and len(jersey_pixels) > 0 and team_kmeans is not None:
                player_color = get_dominant_color(jersey_pixels)
                team_id, team_bgr = assign_team(player_color, team_kmeans)
                
                # Cast BGR floats to standard integers for OpenCV drawing
                color = (int(team_bgr[0]), int(team_bgr[1]), int(team_bgr[2]))
                label = f"Team {team_id + 1}"
            else:
                color = (255, 255, 255)
                label = "Player"

            # Draw team bounding box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Detect ball (Class 32)
    ball_results = ball_model(frame, classes=[32], verbose=False)[0]
    for box in ball_results.boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
        cv2.putText(
            annotated_frame, 
            "Ball", 
            (bx1, by1 - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            2
        )

    # Output frame to file and live display window
    out.write(annotated_frame)
    cv2.imshow("PitchIntel AI - Team Analytics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("[+] Processing complete. Output saved to 'output_team_analytics.mp4'.")

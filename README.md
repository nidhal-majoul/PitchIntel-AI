# PitchIntel AI ⚽🤖
> **Real-Time Football Computer Vision & Player Pose Analytics**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)

**PitchIntel AI** is a computer vision pipeline built to analyze broadcast football match clips frame-by-frame. It utilizes deep learning neural networks to perform real-time player pose estimation and ball tracking.

---

## 📋 Table of Contents
- [Overview \& Key Features](#-overview--key-features)
- [🛠️ Tech Stack \& Tools](#️-tech-stack--tools)
- [🧠 How the AI Works](#-how-the-ai-works)
- [📁 Repository Structure](#-repository-structure)
- [🔍 Detailed Code Breakdown](#-detailed-code-breakdown)
- [🐍 Source Code (`camera.py`)](#-source-code-camerapy)
- [🚀 How to Run the Project](#-how-to-run-the-project)

---

## 🌟 Overview & Key Features

- **Skeletal Pose Estimation:** Detects players on the pitch and extracts 17 anatomical body keypoints (eyes, shoulders, elbows, hips, knees, ankles) to visualize posture and movement.
- **Ball Detection:** Scans every frame for the football (Class 32) and displays bounding boxes along with real-time model confidence scores.
- **Live Video Pipeline:** Streams the video frame-by-frame, overlays computer vision visual annotations, displays a live interactive window, and exports the final result as an `.mp4` file.

---

## 🛠️ Tech Stack & Tools

| Tool / Library | Category | Description |
| :--- | :--- | :--- |
| **Python 3.9+** | Programming Language | Core execution environment. |
| **Ultralytics YOLOv8** | Computer Vision Framework | Pre-trained deep learning neural network architecture for pose and object detection. |
| **OpenCV (`opencv-python`)** | Image Processing | Handles video streaming, frame extraction, bounding box drawing, text rendering, and video saving. |
| **PyTorch** | Deep Learning Backend | Powering neural network tensor calculations on CPU/GPU. |

---

## 🧠 How the AI Works

Unlike traditional computer vision algorithms that rely on manual color masks or fixed shapes, PitchIntel AI relies on **Convolutional Neural Networks (CNNs)** trained on massive datasets.

```text
Broadcast Input Video (data/Lionel_Messi.mp4)
       │
       ├──► 1. Frame Capture (OpenCV cv2.VideoCapture)
       │
       ├──► 2. Pose Estimation Inference (yolov8n-pose.pt)
       │       └── Detects humans & renders 17 skeletal joint keypoints
       │
       ├──► 3. Object Detection Inference (yolov8n.pt)
       │       └── Scans for Class 32 (sports ball) & extracts bounding boxes + confidence
       │
       ├──► 4. OpenCV Annotation Layer
       │       └── Draws yellow bounding box & text label for detected ball
       │
       └──► 5. Video File Export (cv2.VideoWriter)
               └── Saves annotated video to output_analytics.mp4

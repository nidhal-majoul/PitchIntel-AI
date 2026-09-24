# PitchIntel AI ⚽ | Computer Vision Pipeline for Football Analytics

PitchIntel AI is an end-to-end computer vision pipeline designed to extract tactical analytics from broadcasting camera feeds using object detection, pose estimation, and color-space clustering.

---

## 🚀 Features & Milestones

### Milestone 1: Player & Ball Detection (Pose Estimation)
* Detects players on the pitch using YOLOv8 pose models (`yolov8n-pose.pt`).
* Tracks ball position frame-by-frame using targeted class filtering (`yolov8n.pt`, Class 32).

### Milestone 2: Unsupervised Team Classification
Instead of retraining heavy detection models for custom kit colors, PitchIntel AI uses a custom two-phase computer vision pipeline to separate players into **Team 1** and **Team 2** in real time.

```text
[Frame Input] ──► [Player BBox] ──► [Torso Crop] ──► [HSV Grass Masking] ──► [K-Means (k=1)] ──► [Global Cluster (k=2)] ──► [Team Assignment]
```

#### Pipeline Architecture
1. **Torso ROI Isolation:** Cuts the top 50% of each player bounding box to isolate shirt pixels and remove shorts, boots, and shadows.
2. **HSV Grass Removal:** Converts torso crops from BGR to HSV color space and applies a green hue mask ($35 \le H \le 85$) to filter out background pitch grass.
3. **Dominant Color Extraction:** Runs single-cluster K-Means ($k=1$) on non-grass jersey pixels to derive a clean BGR color vector for each player.
4. **Global Team Clustering:** Collects initial frame samples and fits a 2-cluster K-Means ($k=2$) model to automatically identify primary kit colors for both teams without manual tuning.

---

## 📁 Repository Structure

```text
PitchIntel-AI/
├── data/
│   └── Lionel_Messi.mp4          # Source video clip
├── output_team_analytics.mp4     # Generated analytics output
├── team_classifier.py            # Helper module for HSV masking & KMeans clustering
├── camera.py                     # Main pipeline orchestration script
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/PitchIntel-AI.git](https://github.com/nidhal-majoul/PitchIntel-AI.git)
   cd PitchIntel-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the classification pipeline:**
   ```bash
   python camera.py
   ```

---

## 🛠️ Tech Stack
* **Language:** Python
* **Vision & AI:** OpenCV, Ultralytics YOLOv8, Scikit-Learn (KMeans), NumPy, PyTorch

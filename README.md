# PitchIntel AI ⚽🤖
> **Real-Time Football Computer Vision & Player Pose Analytics**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)

**PitchIntel AI** es una solución de visión por computador diseñada para analizar retransmisiones de fútbol en tiempo real. Utiliza redes neuronales profundas (Deep Learning) para extraer la pose corporal de los jugadores y rastrear la trayectoria del balón cuadro por cuadro.

---

## 🧠 ¿Cómo funciona la Inteligencia Artificial?

El script principal (`camera.py`) no utiliza reglas de programación tradicionales (como filtros de color o formas fijas), sino dos modelos de **Redes Neuronales Convolucionales (CNN)** previamente entrenadas con la arquitectura **YOLOv8**:

```text
Entrada (Video .mp4)
       │
       ├──► 1. Extracción de fotogramas (OpenCV)
       │
       ├──► 2. Inferencia de Pose (yolov8n-pose.pt)
       │       └── Detecta jugadores y dibuja 17 puntos clave (esqueleto)
       │
       ├──► 3. Detección de Objetos (yolov8n.pt)
       │       └── Filtra Clase 32 (sports ball) y calcula la confianza
       │
       ├──► 4. Renderizado (OpenCV cv2.rectangle / cv2.putText)
       │
       └──► Salida: Exportación de Video (output_analytics.mp4)

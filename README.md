---
title: TrafficVision
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# TrafficVision — Real-Time Road Traffic Detection and Tracking

**Group 10** | AIMS Senegal | Computer Vision Project 2025–2026  
Members: Moute Jean-Baptiste, Okere Rafiatou, Gaolatlhe Angelah Kgato  
Supervisor: Dr. Jordan Felicien Masakuna

---

## Overview

TrafficVision is a real-time computer vision system for detecting, tracking, and counting road-traffic objects across multiple video scenes. It uses a fine-tuned YOLOv11 model combined with the ByteTrack multi-object tracking algorithm to assign persistent identities to detected objects across frames, enabling accurate unique-object counting rather than simple per-frame detection counts.

The system is part of a collaborative project at AIMS Senegal where each group contributes traffic data in a shared format to build a global multi-scene traffic monitoring dataset and dashboard.

---

## Features

- **Object detection** using fine-tuned YOLOv11n across six traffic classes: person, bicycle, car, motorcycle, bus, truck
- **Multi-object tracking** with ByteTrack — each object receives a unique, persistent ID across frames
- **Unique counting** — each object is counted only once per session, regardless of how many frames it appears in
- **Speed estimation** in pixels per second from centroid displacement between frames
- **Direction detection** — up or down movement relative to the frame
- **Line-crossing events** — detects when an object crosses the horizontal counting line at mid-frame
- **Three video input modes** — local file upload, YouTube URL (direct stream), or webcam
- **Live annotated stream** — real-time bounding boxes, class labels, and track IDs displayed in the browser
- **19-column CSV logs** following the shared schema for global dashboard integration
- **Analytics dashboard** — per-scene object counts, class distribution charts, traffic intensity timeline, and dataset download

---

## Project Structure

```
.
├── app.py                  # Flask web application
├── utils/
│   └── tracker.py          # TrafficTracker class (detection, tracking, CSV logging)
├── templates/
│   ├── home.html           # Landing page
│   ├── index.html          # Live detection interface
│   └── dashboard.html      # Analytics dashboard
├── models/
│   └── yolo11n.pt          # Fine-tuned YOLOv11n weights
├── data/
│   ├── traffic.mp4         # Test video — Scene 1
│   ├── traffic-sign-test.mp4
│   └── group_10_traffic_log.csv  # Sample detection log
├── requirements.txt
└── Dockerfile
```

---

## CSV Log Schema

Every detection is logged with the following 19 columns:

| Column | Description |
|---|---|
| `frame` | Frame index (0-based) |
| `timestamp_sec` | Time in seconds from video start |
| `scene_name` | Name of the scene |
| `group_id` | Student group identifier (`group_10`) |
| `video_name` | Original video filename |
| `track_id` | Persistent ByteTrack object ID |
| `class_name` | Detected class (car, person, etc.) |
| `confidence` | YOLOv11 confidence score |
| `bbox_x1` | Bounding box left edge (px) |
| `bbox_y1` | Bounding box top edge (px) |
| `bbox_x2` | Bounding box right edge (px) |
| `bbox_y2` | Bounding box bottom edge (px) |
| `cx` | Centroid x coordinate (px) |
| `cy` | Centroid y coordinate (px) |
| `frame_width` | Frame width (px) |
| `frame_height` | Frame height (px) |
| `crossed_line` | Whether the object crossed the counting line |
| `direction` | Direction of movement (`up` or `down`) |
| `speed_px_s` | Estimated speed in pixels per second |

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/okere-rafiatou/trafficvision-group10
cd trafficvision-group10
pip install -r requirements.txt
python app.py
```

The application starts on `http://localhost:7860`.

---

## Usage

1. Open the application in your browser
2. Go to the **App** tab
3. Select an input source: upload a video file, paste a YouTube URL, or use the webcam
4. Choose which object classes to detect
5. Click **Start** and watch the live detection stream
6. When processing is complete, open the **Dashboard** tab to view statistics and download the CSV log

**Note:** The webcam input is only available when running the application locally. It is not supported in the hosted cloud version.

---

## Deployment

The application is deployed on Hugging Face Spaces using Docker:

[https://huggingface.co/spaces/Rafiatou/trafficvision-group10](https://huggingface.co/spaces/Rafiatou/trafficvision-group10)

---

## Model

The detection model is YOLOv11n (nano variant) fine-tuned on traffic-specific data covering the six target classes. The base weights were taken from the Ultralytics pre-trained COCO checkpoint and further trained to improve performance on real-world traffic footage.

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- [Ultralytics YOLOv11](https://docs.ultralytics.com) — detection and tracking framework  
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — multi-object tracking algorithm  
- [Pexels](https://www.pexels.com/search/videos/traffic/) — source for test videos  
- AIMS Senegal — academic supervision and project framework

# 🚦 Traffic Analytics System

## Overview

An end-to-end Traffic Analytics System built using **YOLOv8l**, **ByteTrack**, **FastAPI**, **Streamlit**, and **Docker** for intelligent traffic monitoring and analysis.

The system automatically detects, tracks, and analyzes vehicles from traffic surveillance videos and generates actionable traffic insights including vehicle counts, speed estimation, lane utilization, traffic flow direction, congestion monitoring, and vehicle type distribution.

---
## 🎥 Demo

### Vehicle Detection and Tracking

![Traffic Analytics Demo](images/demo.gif)

---

## Features

| Feature                       | Status |
| ----------------------------- | ------ |
| Vehicle Detection             | ✅      |
| Vehicle Tracking              | ✅      |
| Speed Estimation              | ✅      |
| Lane Utilization Analysis     | ✅      |
| Vehicle Counting              | ✅      |
| Vehicle Type Classification   | ✅      |
| Inbound / Outbound Analysis   | ✅      |
| Traffic Congestion Monitoring | ✅      |
| CSV Report Generation         | ✅      |
| Annotated Video Generation    | ✅      |
| FastAPI Backend               | ✅      |
| Streamlit Dashboard           | ✅      |
| Docker Deployment             | ✅      |

---

## System Architecture

```text
Traffic Video
      │
      ▼
 YOLOv8l Detector
      │
      ▼
 ByteTrack Tracker
      │
      ▼
 Analytics Engine
      │
      ├── Vehicle Counting
      ├── Speed Estimation
      ├── Lane Utilization
      ├── Traffic Flow Analysis
      ├── Congestion Monitoring
      └── Vehicle Type Statistics
      │
      ▼
 FastAPI Backend
      │
      ▼
 Streamlit Dashboard
```

---

## Dataset

### UA-DETRAC Dataset

The system was evaluated using the UA-DETRAC traffic surveillance dataset.

Dataset Characteristics:

* Urban traffic surveillance videos
* Multiple weather conditions
* Different traffic densities
* Various vehicle categories

Vehicle Categories:

* Car
* Bus
* Van
* Truck
* Motorcycle

---

## Technology Stack

### Computer Vision

* YOLOv8l
* OpenCV
* ByteTrack

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Deployment

* Docker
* AWS EC2

---

## Analytics Generated

The system automatically computes:

* Total Vehicle Count
* Unique Vehicle Count
* Average Vehicle Speed
* Maximum Vehicle Speed
* Lane-wise Vehicle Distribution
* Peak Traffic Density
* Inbound Traffic Count
* Outbound Traffic Count
* Vehicle Type Distribution

---

## Sample Output

```json
{
  "video_name": "MVI_20011",
  "unique_vehicles": 91,
  "avg_speed": 15.27,
  "max_speed": 220.15,
  "avg_vehicle_count": 5.98,
  "peak_vehicle_count": 12,
  "lane_1": 69,
  "lane_2": 52,
  "lane_3": 30,
  "lane_4": 25,
  "inbound": 7,
  "outbound": 10
}
```

---

## Project Structure

```text
TRAFFIC_ANALYTICS_APP
│
├── backend
│   ├── app.py
│   ├── tracker.py
│   ├── inference.py
│   ├── config.py
│   ├── models
│   │   └── yolov8l.pt
│   ├── uploads
│   └── output
│
├── frontend
│   └── streamlit_app.py
│
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## API Endpoints

### POST /predict

Upload a traffic video and receive:

* Traffic Analytics Summary
* Annotated Video
* Analytics CSV
* Vehicle Type CSV

### GET /

Health Check Endpoint

---

## Local Installation

### Clone Repository

```bash
git clone https://github.com/vishwnayani/traffic-analytics-system.git

cd traffic-analytics-system
```

### Create Virtual Environment

```bash
python -m venv venv

venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
cd backend

uvicorn app:app --reload
```

### Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

## Docker Deployment

### Build Image

```bash
docker build -t traffic-analytics .
```

### Run Container

```bash
docker run -p 8501:8501 -p 8000:8000 traffic-analytics
```

---

## Results

The system successfully performs:

* Multi-vehicle detection and tracking
* Traffic density monitoring
* Lane utilization analysis
* Vehicle speed estimation
* Traffic flow direction analysis
* Automated report generation

---

## Future Enhancements

* Real-time CCTV stream processing
* Multi-camera traffic monitoring
* License Plate Recognition (ANPR)
* Traffic Violation Detection
* Vehicle Re-identification
* Traffic Congestion Prediction using Deep Learning
* Edge Deployment on NVIDIA Jetson

---

## Author

**Vishwa Nayani**

M.Tech Artificial Intelligence & Data Science

Computer Vision | Deep Learning | GEN AI | MLOps

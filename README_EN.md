# Traffic Flow Detection Project

An intelligent traffic flow detection system based on YOLO and OpenCV for vehicle detection, counting, and traffic analysis.

---

## Introduction

This project is a computer vision based traffic monitoring system that can detect and analyze vehicles from road surveillance videos in real time.

The system mainly uses:

- YOLO object detection
- OpenCV video processing
- Python deep learning frameworks

Main features include:

- Vehicle detection
- Real-time vehicle counting
- Traffic flow statistics
- Video stream analysis
- Visualization of detection results

---

## Features

### 1. Vehicle Detection

Automatically detects different types of vehicles, including:

- Cars
- Trucks
- Buses
- Motorcycles

---

### 2. Real-Time Counting

The system can count vehicles crossing a specified region in real time.

Functions include:

- One-way counting
- Multi-vehicle statistics
- Traffic density analysis

---

### 3. Video Analysis

Supports:

- Local video file detection
- Real-time webcam detection
- Output video saving

---

### 4. Visualization

Detection results are displayed visually with:

- Bounding boxes
- Vehicle labels
- Real-time counts
- FPS information

---

## Project Structure

```bash
Traffic-Flow-Detection-Project/
│
├── data/                # Dataset
├── model/               # Model files
├── output/              # Output results
├── videos/              # Test videos
├── main.py              # Main program
├── detect.py            # Detection module
├── requirements.txt     # Dependencies
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jiantianbujiaban/Traffic-Flow-Detection-Project.git
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Detect Vehicles from Video

```bash
python main.py
```

---

### Real-Time Camera Detection

After modifying the input source:

```bash
python main.py
```

---

## Tech Stack

- Python
- OpenCV
- YOLO
- NumPy
- Deep Learning

---

## Application Scenarios

- Smart Cities
- Intelligent Transportation Systems
- Road Monitoring
- Traffic Statistics
- AI Video Analytics

---

## Future Improvements

Possible future upgrades include:

- Multi-object tracking
- Vehicle speed estimation
- License plate recognition
- Traffic light detection
- Web dashboard
- GPU deployment optimization

---

## License

This project is licensed under the MIT License.

---

## Author

GitHub: https://github.com/jiantianbujiaban

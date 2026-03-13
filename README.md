
# Project Prompt: Edge AI YOLO11n Vision Server for Raspberry Pi

### Project Goal
Build a real-time Edge AI object detection system running on a Raspberry Pi 4B (8GB). The system will ingest a live MJPEG stream from a phone (IP Webcam app), perform inference using an **OpenVINO-optimized YOLO11n** model, and serve the annotated video back to a web browser using **FastAPI**.

### Target Hardware/OS
- **Device:** Raspberry Pi 4B (ARM64 architecture)
- **OS:** Ubuntu with ROS2 Humble
- **Model Optimization:** OpenVINO (Required for Pi CPU performance)

---

### 1. Project Folder Structure
```text
yoloEDGE/
├── models/
│   └── yolo11n_openvino_model/    # Optimized OpenVINO model folder
├── src/
│   ├── __init__.py
│   ├── camera.py                  # Threaded IP Webcam capture logic
│   ├── detector.py                # YOLO11 OpenVINO inference class
│   └── server.py                  # FastAPI streaming server logic
├── main.py                        # Entry point (Orchestrator)
├── .env                           # Configuration (IPs, Ports, Confidence)
└── requirements.txt               # Dependencies
```

---

### 2. Technical Requirements for the Coding Agent

#### A. Dependency Management (`requirements.txt`)
Ensure the following versions are used to avoid "Illegal Instruction" errors on ARM64 and use a venv if needed:
- `ultralytics`
- `openvino>=2024.0.0`
- `fastapi`
- `uvicorn`
- `opencv-python-headless` (Avoids GUI dependency issues on headless Ubuntu)
- `numpy<2.0.0` (Critical for Raspberry Pi compatibility)
- `python-dotenv`

#### B. Camera Module (`src/camera.py`)
- Implement a `ThreadedCamera` class using `cv2.VideoCapture`.
- It must continuously read frames from the `IP_WEBCAM_URL` in a background thread to prevent frame buffering/latency.
- Provide a `get_frame()` method that returns the most recent frame.

#### C. Detection Module (`src/detector.py`)
- Load the YOLO11n model using the OpenVINO framework.
- The class should initialize the model once.
- The `predict()` method should accept a raw frame, perform inference, and return an annotated frame using `results[0].plot()`.
- **Optimization:** Use `half=True` or `int8` if available, and set `task='detect'`.

#### D. Server Module (`src/server.py`)
- Use **FastAPI** with `StreamingResponse`.
- Create a route `/video_feed` that yields JPEG-encoded frames in a multipart/x-mixed-replace format.
- Ensure the server is hosted on `0.0.0.0` so it's accessible from other devices on the network.

#### E. Main Orchestrator (`main.py`)
- Use `python-dotenv` to load configurations.
- Coordinate the Camera thread, the Detector, and the FastAPI server.
- **Frame Skipping Logic:** Since the Pi 4 handles ~3-5 FPS, implement logic to drop frames so the "Live" feed doesn't lag minutes behind.

---

### 3. Environment Variables (`.env`)
```bash
IP_WEBCAM_URL=http://<phone_ip>:8080/video
PORT=8000
MODEL_PATH=models/yolo11n_openvino_model/
CONFIDENCE=0.25
```

---

### 4. Implementation Steps for the Agent
1.  **Code the Camera Streamer:** Handle the connection to the Phone's IP.
2.  **Code the Detector:** Implement the YOLO11 logic specifically utilizing the OpenVINO exported folder.
3.  **Code the FastAPI Server:** Create the streaming endpoint.
4.  **Integration:** Connect them in `main.py` using a producer-consumer pattern (or simple shared global variables with Locks).
5.  **Optimization:** Ensure `numpy` and `opencv` imports are handled safely to avoid ARM-specific crashes.

---

### 5. Final Output Expectation
The user should be able to run `python3 main.py` on the Raspberry Pi and open `http://<pi_ip>:8000/video_feed` on any computer to see the real-time detections.

***

**Instruction for the Agent:** *“Please generate the complete Python code for the structure above. Prioritize thread safety when sharing frames between the camera thread and the FastAPI server. Assume the model has already been exported to OpenVINO format on a PC and moved to the `/models` folder.”*
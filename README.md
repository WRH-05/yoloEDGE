
# YOLO EDGE on Raspberry Pi (Beginner Guide)

This project runs real-time object detection on a Raspberry Pi using:
- a phone camera stream (MJPEG via IP Webcam)
- an OpenVINO YOLO model
- a FastAPI video server

If you are new to Raspberry Pi or Python, follow the steps in order.

## What You Need

- Raspberry Pi 4 (recommended 8GB), 64-bit OS
- Phone and Pi on the same Wi-Fi network
- Internet access on the Pi (for package install)
- This repo cloned on the Pi
- OpenVINO model files already in this folder:
  - models/yolo11n_openvino_model/
  - models/yolo11n_openvino_model/*.xml

## 1. Open Terminal on the Pi

Go into the project folder:

	cd ~/yoloEDGE

If you have not cloned the repo yet:

	git clone https://github.com/WRH-05/yoloEDGE.git
	cd yoloEDGE

## 2. Install System Packages (one-time)

	sudo apt update
	sudo apt install -y python3 python3-venv python3-pip git

## 3. Create and Activate a Virtual Environment

	python3 -m venv .venv
	source .venv/bin/activate

After activation, your prompt usually shows (.venv).

## 4. Install Python Dependencies

	pip install --upgrade pip
	pip install -r requirements.txt

## 5. Set Up Phone Camera Stream

1. Install IP Webcam on your Android phone.
2. Start the server in the app.
3. Find your phone stream URL. It usually looks like:

	   http://192.168.1.50:8080/video
4. i also would highly recommend going into this ip address on a browser to change the video resolution to 640x480 or less.
5. Keep the phone app running.

## 6. Create .env File

In the project root, create a file named .env:

	nano .env

Paste this (change the phone IP):

	IP_WEBCAM_URL=http://192.168.1.50:8080/video
	PORT=8000
	MODEL_PATH=models/yolo11n_openvino_model/
	CONFIDENCE=0.25

Optional performance tuning for Raspberry Pi:

	TARGET_DETECT_FPS=2.5
	OV_NUM_THREADS=4
	OV_NUM_STREAMS=2
	OV_PERFORMANCE_HINT=THROUGHPUT
	MAX_CANDIDATES=300
	STREAM_JPEG_QUALITY=70

Save and exit in nano:
- Ctrl+O, Enter
- Ctrl+X

## 7. Run the App

Make sure your venv is active, then run:

	python3 main.py

You should see output similar to:
- Video feed (local): http://127.0.0.1:8000/video_feed
- Video feed (network): http://<pi_ip>:8000/video_feed

## 8. View the Stream

From another device on the same network, open:

	http://<pi_ip>:8000/video_feed

Example:

	http://192.168.1.20:8000/video_feed

If you open just the root URL, you will get a message telling you to open /video_feed.

## 9. Stop the App

In the terminal running the server:

	Ctrl+C

## Troubleshooting (Most Common)

1. Error: IP_WEBCAM_URL is required
- Your .env file is missing or not in the project root.

2. No video appears
- Check phone and Pi are on same Wi-Fi.
- Open the phone stream URL directly in a browser first.
- Keep IP Webcam app running in foreground.

3. Port not reachable
- Confirm you are using Pi IP from the startup log.
- Confirm nothing else is using port 8000.
- Try changing PORT in .env to 8001 and restart.

4. Slow FPS on Pi
- Lower stream resolution/FPS in the phone app.
- Reduce STREAM_JPEG_QUALITY (for example 60).
- Lower TARGET_DETECT_FPS (for example 2.0).

## Optional: Export OpenVINO Model on PC

Do this only if you need to regenerate model files:

	pip install -r requirements-export.txt
	python export_model.py

Then copy the generated OpenVINO model folder into models/ on the Pi.
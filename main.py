from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np
import uvicorn
from dotenv import load_dotenv

from src.camera import ThreadedCamera
from src.detector import YOLODetector
from src.server import create_app


def _get_env(primary_key: str, fallback_key: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(primary_key)
    if (value is None or value == "") and fallback_key:
        value = os.getenv(fallback_key)
    if value is None or value == "":
        return default
    return value.strip()


def _get_int_env(primary_key: str, fallback_key: Optional[str], default: int) -> int:
    raw = _get_env(primary_key, fallback_key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float_env(primary_key: str, fallback_key: Optional[str], default: float) -> float:
    raw = _get_env(primary_key, fallback_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def main() -> None:
    load_dotenv()

    stream_url = _get_env("IP_WEBCAM_URL")
    if not stream_url:
        raise RuntimeError("IP_WEBCAM_URL is required")

    port = _get_int_env("PORT", "SERVER_PORT", 8000)
    confidence = _get_float_env("CONFIDENCE", "CONFIDENCE_THRESHOLD", 0.25)
    model_path = _get_env("MODEL_PATH", default="models/yolo11n_openvino_model/")
    frame_skip = max(1, _get_int_env("FRAME_SKIP", None, 2))

    camera = ThreadedCamera(stream_url)
    detector = YOLODetector(model_path=model_path, confidence=confidence)

    frame_lock = threading.Lock()
    stop_event = threading.Event()
    latest_frame: Optional[np.ndarray] = None

    def get_latest_frame() -> Optional[np.ndarray]:
        with frame_lock:
            if latest_frame is None:
                return None
            return latest_frame.copy()

    def processing_loop() -> None:
        nonlocal latest_frame
        frame_count = 0

        while not stop_event.is_set():
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            try:
                processed = detector.predict(frame)
            except Exception:
                processed = frame

            with frame_lock:
                latest_frame = processed

    camera.start()
    worker = threading.Thread(target=processing_loop, name="detector-worker", daemon=True)
    worker.start()

    app = create_app(get_latest_frame)

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        camera.stop()


if __name__ == "__main__":
    main()

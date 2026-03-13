from __future__ import annotations

import os
import socket
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


def _get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> None:
    load_dotenv()

    stream_url = _get_env("IP_WEBCAM_URL")
    if not stream_url:
        raise RuntimeError("IP_WEBCAM_URL is required")

    port = _get_int_env("PORT", "SERVER_PORT", 8000)
    confidence = _get_float_env("CONFIDENCE", "CONFIDENCE_THRESHOLD", 0.25)
    model_path = _get_env("MODEL_PATH", default="models/yolo11n_openvino_model/")
    target_detect_fps = max(0.5, _get_float_env("TARGET_DETECT_FPS", None, 2.5))
    detect_interval = 1.0 / target_detect_fps

    ov_num_threads = max(1, _get_int_env("OV_NUM_THREADS", None, 4))
    ov_num_streams = max(1, _get_int_env("OV_NUM_STREAMS", None, 2))
    ov_performance_hint = _get_env("OV_PERFORMANCE_HINT", default="THROUGHPUT")
    max_candidates = max(50, _get_int_env("MAX_CANDIDATES", None, 300))
    stream_jpeg_quality = max(40, min(95, _get_int_env("STREAM_JPEG_QUALITY", None, 70)))

    camera = ThreadedCamera(stream_url)
    detector = YOLODetector(
        model_path=model_path,
        confidence=confidence,
        performance_hint=ov_performance_hint,
        num_threads=ov_num_threads,
        num_streams=ov_num_streams,
        max_candidates=max_candidates,
    )

    frame_lock = threading.Lock()
    stop_event = threading.Event()
    latest_raw_frame: Optional[np.ndarray] = None
    latest_boxes = np.empty((0, 4), dtype=np.float32)
    latest_scores = np.empty((0,), dtype=np.float32)
    latest_class_ids = np.empty((0,), dtype=np.int32)
    latest_det_fps: float = 0.0

    def get_latest_frame() -> Optional[np.ndarray]:
        with frame_lock:
            if latest_raw_frame is None:
                return None
            frame = latest_raw_frame.copy()
            boxes = latest_boxes.copy()
            scores = latest_scores.copy()
            class_ids = latest_class_ids.copy()
            det_fps = latest_det_fps
        return detector.annotate(frame, boxes, scores, class_ids, fps=det_fps)

    def ingest_loop() -> None:
        nonlocal latest_raw_frame

        while not stop_event.is_set():
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            with frame_lock:
                latest_raw_frame = frame

    def inference_loop() -> None:
        nonlocal latest_boxes, latest_scores, latest_class_ids, latest_det_fps
        last_infer_time = 0.0

        while not stop_event.is_set():
            with frame_lock:
                frame = None if latest_raw_frame is None else latest_raw_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            now = time.perf_counter()
            elapsed_since_last = now - last_infer_time
            if elapsed_since_last < detect_interval:
                time.sleep(0.001)
                continue

            start = time.perf_counter()
            try:
                boxes, scores, class_ids = detector.infer(frame)
            except Exception:
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
                class_ids = np.empty((0,), dtype=np.int32)

            infer_elapsed = max(time.perf_counter() - start, 1e-6)
            instant_fps = 1.0 / infer_elapsed
            latest_det_fps = instant_fps if latest_det_fps <= 0.0 else (0.85 * latest_det_fps + 0.15 * instant_fps)
            last_infer_time = time.perf_counter()

            with frame_lock:
                latest_boxes = boxes
                latest_scores = scores
                latest_class_ids = class_ids

    camera.start()
    ingest_worker = threading.Thread(target=ingest_loop, name="frame-ingest-worker", daemon=True)
    infer_worker = threading.Thread(target=inference_loop, name="detector-worker", daemon=True)
    ingest_worker.start()
    infer_worker.start()

    app = create_app(get_latest_frame, jpeg_quality=stream_jpeg_quality)
    local_ip = _get_local_ip()
    print(f"Video feed (local): http://127.0.0.1:{port}/video_feed")
    print(f"Video feed (network): http://{local_ip}:{port}/video_feed")
    print(
        "Tuning: "
        f"OV_NUM_THREADS={ov_num_threads}, "
        f"OV_NUM_STREAMS={ov_num_streams}, "
        f"OV_PERFORMANCE_HINT={ov_performance_hint}, "
        f"MAX_CANDIDATES={max_candidates}, "
        f"STREAM_JPEG_QUALITY={stream_jpeg_quality}, "
        f"TARGET_DETECT_FPS={target_detect_fps:.1f}"
    )

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        stop_event.set()
        ingest_worker.join(timeout=2.0)
        infer_worker.join(timeout=2.0)
        camera.stop()


if __name__ == "__main__":
    main()

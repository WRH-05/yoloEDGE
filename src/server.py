from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

FrameProvider = Callable[[], Optional[np.ndarray]]


def create_app(frame_provider: FrameProvider, jpeg_quality: int = 80) -> FastAPI:
    app = FastAPI()
    boundary = "frame"

    def frame_generator():
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        fps_value = 0.0
        previous_time = time.perf_counter()
        while True:
            frame = frame_provider()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            delta = max(now - previous_time, 1e-6)
            instant_fps = 1.0 / delta
            fps_value = instant_fps if fps_value <= 0.0 else (0.9 * fps_value + 0.1 * instant_fps)
            previous_time = now

            cv2.putText(
                frame,
                f"Stream FPS: {fps_value:.1f}",
                (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            success, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            yield (
                b"--" + boundary.encode("ascii") + b"\r\n"
                + b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

    @app.get("/video_feed")
    def video_feed() -> StreamingResponse:
        media_type = f"multipart/x-mixed-replace; boundary={boundary}"
        return StreamingResponse(frame_generator(), media_type=media_type)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Open /video_feed for the live stream."}

    return app

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
        while True:
            frame = frame_provider()
            if frame is None:
                time.sleep(0.01)
                continue

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

    return app

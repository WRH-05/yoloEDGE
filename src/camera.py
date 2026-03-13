from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class ThreadedCamera:
    def __init__(self, stream_url: str, reconnect_delay: float = 1.0) -> None:
        self.stream_url = stream_url
        self.reconnect_delay = max(reconnect_delay, 0.1)

        self._frame_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None and not self._open_capture():
                time.sleep(self.reconnect_delay)
                continue

            if self._capture is None:
                time.sleep(self.reconnect_delay)
                continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._release_capture()
                time.sleep(self.reconnect_delay)
                continue

            with self._frame_lock:
                self._frame = frame

    def _open_capture(self) -> bool:
        self._release_capture()

        capture = cv2.VideoCapture(self.stream_url)
        if not capture.isOpened():
            capture.release()
            return False

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        return True

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

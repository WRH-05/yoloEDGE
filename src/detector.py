from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str, confidence: float = 0.25, device: str = "cpu") -> None:
        self.confidence = confidence
        self.device = device
        self.model = YOLO(model_path, task="detect")
        self._use_half = True

    def predict(self, frame: np.ndarray) -> np.ndarray:
        predict_kwargs = {
            "source": frame,
            "task": "detect",
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }

        if self._use_half:
            predict_kwargs["half"] = True

        try:
            results = self.model.predict(**predict_kwargs)
        except Exception:
            if self._use_half:
                self._use_half = False
                predict_kwargs.pop("half", None)
                results = self.model.predict(**predict_kwargs)
            else:
                raise

        if not results:
            return frame

        return results[0].plot()

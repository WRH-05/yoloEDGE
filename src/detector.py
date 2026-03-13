from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
import openvino as ov


class YOLODetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        device: str = "CPU",
        iou_threshold: float = 0.45,
    ) -> None:
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device.upper()

        xml_path = self._resolve_model_xml(model_path)

        core = ov.Core()
        model = core.read_model(xml_path)
        self.compiled_model = core.compile_model(model, self.device, {"PERFORMANCE_HINT": "LATENCY"})

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        input_shape = list(self.input_layer.shape)
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        self.class_names = self._load_class_names(Path(xml_path).with_name("metadata.yaml"))

    def predict(self, frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame.shape[:2]
        input_tensor, ratio, dwdh = self._preprocess(frame)
        raw_output = self.compiled_model([input_tensor])[self.output_layer]

        boxes, scores, class_ids = self._postprocess(raw_output, ratio, dwdh, (orig_h, orig_w))
        if boxes.shape[0] == 0:
            return frame

        return self._draw_detections(frame, boxes, scores, class_ids)

    @staticmethod
    def _resolve_model_xml(model_path: str) -> str:
        path = Path(model_path)
        if path.is_file() and path.suffix.lower() == ".xml":
            return str(path)
        if path.is_dir():
            candidates = sorted(path.glob("*.xml"))
            if candidates:
                return str(candidates[0])
        raise FileNotFoundError(f"OpenVINO .xml model not found in: {model_path}")

    @staticmethod
    def _load_class_names(metadata_path: Path) -> dict[int, str]:
        names: dict[int, str] = {}
        if not metadata_path.exists():
            return names

        in_names_block = False
        with metadata_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.rstrip("\n")
                if line.startswith("names:"):
                    in_names_block = True
                    continue
                if in_names_block and line and not line.startswith("  "):
                    break
                if not in_names_block:
                    continue

                match = re.match(r"^\s*(\d+)\s*:\s*(.+)$", line)
                if not match:
                    continue

                index = int(match.group(1))
                label = match.group(2).strip().strip("'\"")
                names[index] = label
        return names

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        image, ratio, dwdh = self._letterbox(frame, (self.input_h, self.input_w))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        return image, ratio, dwdh

    @staticmethod
    def _letterbox(
        frame: np.ndarray,
        new_shape: tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        shape = frame.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return frame, ratio, (dw, dh)

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        dwdh: tuple[float, float],
        original_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = output[0]
        if prediction.ndim == 2 and prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T

        if prediction.shape[1] < 5:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        boxes_xywh = prediction[:, :4]
        class_scores = prediction[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        valid = scores >= self.confidence
        if not np.any(valid):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        boxes_xywh = boxes_xywh[valid]
        scores = scores[valid]
        class_ids = class_ids[valid].astype(np.int32)

        boxes_xyxy = np.empty_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        dw, dh = dwdh
        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= ratio

        orig_h, orig_w = original_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

        nms_boxes = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            nms_boxes.append([float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))])

        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.confidence, self.iou_threshold)
        if len(indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        selected = np.array(indices).reshape(-1)
        return boxes_xyxy[selected], scores[selected], class_ids[selected]

    def _draw_detections(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        annotated = frame.copy()

        for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            class_name = self.class_names.get(int(class_id), str(int(class_id)))
            label = f"{class_name} {float(score):.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text_top = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(annotated, (x1, y_text_top), (x1 + text_w + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

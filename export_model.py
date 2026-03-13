from ultralytics import YOLO

# Load the smallest YOLO11 model
model = YOLO("yolo11n.pt")

# Export to OpenVINO (optimized for CPU inference)
model.export(format="openvino")
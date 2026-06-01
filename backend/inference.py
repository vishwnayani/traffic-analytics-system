from ultralytics import YOLO

from backend.config import MODEL_PATH


print("Loading YOLOv8l...")

model = YOLO(
    MODEL_PATH
)

print("YOLOv8l Loaded")


def get_model():

    return model
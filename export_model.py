"""Export the YOLO model to engine files for GPU use.

This script is not used by the main program. Run it only when you need to
export the models on your computer. It requires a different PyTorch
environment, so use a separate virtual environment for this script.
"""
from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")

model.export(
    format="engine",
    half=True,
    imgsz=320,
    device=0,
    workspace=2)

# People Detector

This project detects and tracks people in a camera feed. It uses YOLO for
person detection and pose estimation, then displays the tracked people and
their estimated face regions in an OpenCV window.

## What it does

- Captures frames from one camera.
- Detects and tracks people using YOLO.
- Estimates face regions from pose keypoints.
- Matches a person to a previous track using a clothing-appearance histogram.
- Periodically verifies detected faces against the configured reference image
  using DeepFace.
- Displays bounding boxes, face regions, person identities, and processing FPS.

The application uses separate threads for frame capture, detection, face
verification, and display. Queues between the threads keep the newest frames
available when processing cannot keep up with the camera.

## Project structure

```text
vectra/
├── decetion_display_class.py  # Capture, detection, tracking, verification, and display classes
├── export_model.py             # Optional YOLO engine export script
├── main.py                     # Application entry point
├── requirements.txt            # Main application dependencies
├── requirements-trt.txt        # Model export dependencies
└── known_faces/
    ├── name_1.jpg             # Reference images used for face verification
```

The YOLO pose model is a local runtime dependency. The TensorRT engine used by
the application is generated from `yolo11m-pose.pt` with `export_model.py` and
must be available in the project directory before running the application.

## Requirements

- Python 3.10 or newer
- A camera
- An NVIDIA GPU with a TensorRT-compatible setup for the configured YOLO models
- Python packages:
  - OpenCV
  - Ultralytics
  - DeepFace

## Setup

From the `vectra` directory, create and activate a virtual
environment for the main application, then install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`export_model.py` can use a separate virtual environment because exporting
TensorRT engines may require different CUDA, TensorRT, and Python package
versions. For example:

```powershell
python -m venv .venv-trt
.\.venv-trt\Scripts\Activate.ps1
pip install -r requirements-trt.txt
python export_model.py
```

The TensorRT and CUDA runtime/development packages are not installed by
`requirements-trt.txt`. Install versions compatible with the target NVIDIA
GPU and operating system before running the export.

Add one or more reference images for verification to:

```text
Vectra/known_faces/
```

Place the required model files in the `vectra` directory:

```text
yolo11m-pose.engine
```

`export_model.py` exports `yolo11m-pose.pt` to a TensorRT engine. The
TensorRT and CUDA runtime/development packages are not installed by
`requirements-trt.txt`; install compatible versions before running the export.

## Running

From the `vectra` directory:

```powershell
python main.py
```

Press `q` in the display window to stop the application.

## Configuration

Camera and processing settings are configured in `main.py`, including:

- Camera index
- Capture queue size
- Detection confidence and IoU thresholds
- Face-verification queue size

Appearance histograms are refreshed periodically in
`decetion_display_class.py` to reduce unnecessary CPU work while tracking.

# People Detector

This project detects and tracks people from multiple camera feeds. It uses
YOLO for person detection and pose estimation, then displays the tracked
people and their estimated face regions in an OpenCV window.

## What it does

- Captures frames from multiple cameras.
- Detects and tracks people using YOLO.
- Estimates face regions from pose keypoints.
- Matches a person to a previous track using a clothing-appearance histogram.
- Periodically verifies detected faces against the configured reference image
  using DeepFace.
- Uses a stereo camera pair to match keypoints and determine whether detections
  belong to the same person.
- Displays bounding boxes, face regions, person identities, and processing FPS.

The application uses separate threads for frame capture, detection, face
verification, stereo vision, and display. Queues between the threads keep the
newest frames available when processing cannot keep up with the cameras.

## Project structure

```text
vectra/
├── detection_display_class.py  # Capture, detection, tracking, verification, and display classes
├── find_cameras.py             # Finds camera indexes that OpenCV can open and read
├── main.py                     # Configures and starts the application
├── export_model.py             # Exports the YOLO pose model to TensorRT
├── requirements.txt            # Main application dependencies
├── requirements-trt.txt        # Model export dependencies
└── known_faces/
    ├── name_1.jpg             # Reference images used for face verification
└── calibration/
    ├── calibration.py         # Calculates the calibration data from images
    ├── calibration_images_taker.py  # Captures an image from both cameras and saves it
    ├── calibration_images/    # Holds the images used for calibration
        ├── left_camera/
            ├── 001.png
        ├── right_camera/
            ├── 001.png
    └── print-board/           # Contains the files for a web application that displays a printable board
        ├── index.html
        ├── main.js
        ├── style.css
```

## Main files

### `detection_display_class.py`

This file contains the main application classes:

- `FrameCapture` opens the configured cameras and places captured frames in a
  queue.
- `Detection` runs YOLO pose detection, updates one ByteTrack tracker per
  camera, extracts appearance and face regions, and sends face crops for
  verification.
- `PeopleManager` keeps track of persistent people identities, track IDs,
  appearance histograms, and face-verification results.
- `FaceDetector` loads reference face embeddings and verifies face crops in a
  separate thread.
- `StereoVision` uses stereo calibration data to match detections between the
  left and right cameras.
- `Display` draws person and face bounding boxes, names, camera indexes, and
  FPS information.

### `find_cameras.py`

This diagnostic script lists the camera devices reported by Windows and tests
each device with the configured OpenCV backend. It reports whether each camera
can be opened and whether it returns a frame.

Run this before `main.py` when camera indexes are uncertain. Use the indexes
reported as working in `main.py`.

### `main.py`

This is the application entry point. It:

1. Defines the camera indexes and queue settings.
2. Creates the people manager, face detector, frame capture, detection, and
   display objects.
3. Starts the processing components.
4. Keeps the display running until the user presses `q`.
5. Stops the components and releases resources.

### `export_model.py`

This optional script converts `yolo11m-pose.pt` into the
`yolo11m-pose.engine` TensorRT file used by `Detection`. It is not run when
starting the application and normally requires a separate environment with
compatible CUDA, TensorRT, PyTorch, and Ultralytics versions.

Run it only when the engine file needs to be created or regenerated.

The YOLO pose model is a local runtime dependency. The TensorRT engine used by
the application is generated from `yolo11m-pose.pt` with `export_model.py` and
must be available in the project directory before running `main.py`.

### `calibration/`

The stereo calibration workflow is optional unless `StereoVision` is enabled.

1. Print or display the board from `print-board/`.
2. Capture synchronized calibration images with
   `calibration_images_taker.py`.
3. Run `calibration.py` to generate `stereo_calibration.npz`.
4. Configure `calibration_data` in `main.py` with the generated file.

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

If you use stereo vision, serve the printable board first:

```powershell
npm install
npm run dev
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
python find_cameras.py
python main.py
```

Press `q` in the display window to stop the application.

## Configuration

Camera and processing settings are configured in `main.py`, including:

- Camera indexes
- Capture queue size
- Detection confidence and IoU thresholds
- Face-verification queue size

Appearance histograms are refreshed periodically in
`detection_display_class.py` to reduce unnecessary CPU work while tracking.

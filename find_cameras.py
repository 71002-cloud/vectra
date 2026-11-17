import cv2
from pygrabber.dshow_graph import FilterGraph

BACKEND = cv2.CAP_MSMF

def find_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    cameras = []

    print("=== Connected cameras ===")

    for camera_index, camera_name in enumerate(devices):
        print(f"Index {camera_index}: {camera_name}")

        camera = cv2.VideoCapture(camera_index, BACKEND)

        if not camera.isOpened():
            print(" Status: Failed to open")
            camera.release()
            continue

        success, frame = camera.read()
        camera.release()

        if not success or frame is None:
            print(" Status: No frame received")
            continue

        print(" Status: Working")
        cameras.append({
            "index": camera_index,
            "model": camera_name
        })

    return cameras

def main():
    cameras = find_cameras()

    print("\n=== Working cameras ===")

    for camera in cameras:
        print(f"Index {camera['index']} | Model {camera['model']}")


if __name__ == "__main__":
    main()

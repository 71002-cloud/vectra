import cv2
import threading
from pathlib import Path

from detection_display_class import Detection, Display, FaceDetector, FrameCapture, PeopleManager, StereoVision

camera_index = [0, 1, 2]
stereo_cameras = [(1, 2)]
camera_amount = len(camera_index)
queue_size = 2
appearance_refresh = 5

class App:
    def __init__(self, detection, display, face_detector, capture, stereo_vision):
        self.detection = detection
        self.display = display
        self.face_detector = face_detector
        self.capture = capture
        self.stereo_vision = stereo_vision

        self.stop_event = threading.Event()

    def start(self):
        self.capture.start()
        self.detection.start()
        self.stereo_vision.start()
        self.face_detector.start()
        self.display.start()

    def stop(self):
        self.display.stop()
        self.detection.stop()
        self.stereo_vision.stop()
        self.face_detector.stop()
        self.capture.stop()
        cv2.destroyAllWindows()

def main():
    photo_folder = str(Path(__file__).parent / "known_faces")
    if photo_folder is None:
        print("No known faces found")

    calibration_data = str(Path(__file__).parent / "calibration" / "stereo_calibration.npz")

    people_manager = PeopleManager(camera_amount=camera_amount)
    face_detector = FaceDetector(queue_size=queue_size, photo_folder=photo_folder, people_manager=people_manager)
    capture = FrameCapture(camera_index=camera_index, camera_amount=camera_amount, queue_size=queue_size, stereo_list=stereo_cameras)
    detection = Detection(conf=0.6, iou=.5, queue_size=queue_size, capture=capture, people_manager=people_manager, face_detector=face_detector, appearance_refresh=appearance_refresh)
    stereo_vision = StereoVision(detection=detection, people_manager=people_manager, queue_size=queue_size, calibration_data=calibration_data)
    display = Display(last_frame_owner=stereo_vision, people_manager=people_manager, camera_amount=camera_amount)

    app = App(detection=detection, display=display, face_detector=face_detector, capture=capture, stereo_vision=stereo_vision)

    try:
        app.start()
        while display.running:
            display.display_loop()

    finally:
        app.stop()


if __name__ == "__main__":
    main()

import cv2
import threading
from pathlib import Path

from detection_display_class import Detection, Display, FaceDetector, FrameCapture, PeopleManager

camera_index = [0, 1, 2]
camera_amount = len(camera_index)
queue_size = 2
appearance_refresh = 5

class App:
    def __init__(self, detection, display, face_detector, capture):
        self.detection = detection
        self.display = display
        self.face_detector = face_detector
        self.capture = capture

        self.stop_event = threading.Event()

    def start(self):
        self.capture.start()
        self.detection.start()
        self.face_detector.start()
        self.display.start()

    def stop(self):
        self.display.stop()
        self.detection.stop()
        self.face_detector.stop()
        self.capture.stop()
        cv2.destroyAllWindows()

def main():
    photo_folder = str(Path(__file__).parent / "known_faces")
    if photo_folder is None:
        print("No known faces found")

    people_manager = PeopleManager(camera_amount=camera_amount)
    face_detector = FaceDetector(queue_size=queue_size, photo_folder=photo_folder, people_manager=people_manager)
    capture = FrameCapture(camera_index=camera_index, camera_amount=camera_amount, queue_size=queue_size)
    detection = Detection(conf=0.6, iou=.5, queue_size=queue_size, capture=capture, people_manager=people_manager, face_detector=face_detector, appearance_refresh=appearance_refresh)
    display = Display(last_frame_owner=detection, people_manager=people_manager, camera_amount=camera_amount)

    app = App(detection=detection, display=display, face_detector=face_detector, capture=capture)

    try:
        app.start()
        while display.running:
            display.display_loop()

    finally:
        app.stop()


if __name__ == "__main__":
    main()

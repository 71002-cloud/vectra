import cv2
import threading
from pathlib import Path

from detection_display_class import Detection, Display, FaceDetector, FrameCapture, PeopleManager

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

    people_manager = PeopleManager()
    face_detector = FaceDetector(2, photo_folder, people_manager)
    capture = FrameCapture(0, 2)
    detection = Detection(0.6, 0.5, 2, capture, people_manager, face_detector, 5)
    display = Display(detection, people_manager)

    app = App(detection, display, face_detector, capture)

    try:
        app.start()
        while display.running:
            display.display_loop()

    finally:
        app.stop()


if __name__ == "__main__":
    main()
